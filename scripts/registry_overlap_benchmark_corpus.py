# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a credential-free ClawHub retrieval workload from pinned Parquet and ONNX files.

Use the project environment with NumPy, PyArrow, ONNX Runtime, and tokenizers.
Pass local copies of the pinned files. The generator never executes skill text
or calls an embedding provider. The resulting matrix measures the ANN backend
with a proxy embedding space and cannot qualify a production provider's recall.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from skillevaluator.embedding.extractor import MAX_FRONTMATTER_FIELD_CHARS
from skillevaluator.utils.structured_data import StructuredDataLimitError, StructuredDataSyntaxError, load_bounded_yaml
from skillevaluator.validators.frontmatter_parser import FRONTMATTER_PATTERN

DATASET = "OpenClaw/clawhub-security-signals-live"
DATASET_REVISION = "e59ffc97c03080af95f38e992be79dab652c8d93"
PARQUET_REVISION = "23a1a8aa3e3cf7359ce81c74d6280b02a3910672"
# LFS SHA-256 values of default/latest at the pinned conversion revision.
PARQUET_SHA256 = {
    "0000.parquet": "1097bfd4c3d951f2e520536b45e1a4e23fd0850c0cee4e802185bf3893e5ae76",
    "0001.parquet": "f53dfb8459fb55634146344d41c365354db28c30a92c8c759d9f73bbac4150bb",
    "0002.parquet": "4ff41571e6ee98056ffec1c1cce154fb1fd2c9634c0fe0823f192133347c63a0",
    "0003.parquet": "8a1b777a8e8d49752048d430c637114dd03f48088efdecc329702a4c2a4f3836",
    "0004.parquet": "6d244fae9341a40c86db95b6da7c8aa7db180c96ad77109e1f83abedcc6360e8",
    "0005.parquet": "8037e71585a9a4fadce41488e0a45231b0f828242ee979570d820ffce595a441",
}
PARQUET_SHARDS = tuple(PARQUET_SHA256)
PARQUET_ROWS = 86_441
MODEL = "Xenova/all-MiniLM-L6-v2"
MODEL_REVISION = "751bff37182d3f1213fa05d7196b954e230abad9"
MODEL_FILE = "model_int8.onnx"
MODEL_SHA256 = "afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1"
TOKENIZER_SHA256 = "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0"
MODEL_DIMENSION = 384
MAX_TOKENS = 128
MAX_FRONTMATTER_CHARS = 64 * 1024
PAD_BATCH_SIZE = 2048
SOURCE_COLUMNS = ("id", "skill_slug", "skill_version", "skill_md_content")


@dataclass(frozen=True)
class CorpusRow:
    source_id: str
    slug: str
    version: str
    embedding_text: str


def _bounded_text(value: object, limit: int) -> str | None:
    if isinstance(value, str) and 0 < len(value) <= limit and value.strip() and "\x00" not in value:
        return value.strip()
    return None


def project_row(value: Mapping[str, object]) -> CorpusRow | None:
    """Extract the embedded name and description, plus version identity."""
    source_id = _bounded_text(value.get("id"), 128)
    slug = _bounded_text(value.get("skill_slug"), 256)
    version = _bounded_text(value.get("skill_version"), 128)
    markdown = value.get("skill_md_content")
    if not (source_id and slug and version and isinstance(markdown, str)):
        return None
    match = FRONTMATTER_PATTERN.match(markdown[:MAX_FRONTMATTER_CHARS].lstrip("\ufeff"))
    if match is None:
        return None
    try:
        frontmatter = load_bounded_yaml(match[1], last_key_wins=True)
    except (StructuredDataLimitError, StructuredDataSyntaxError):
        return None
    if not isinstance(frontmatter, dict):
        return None
    name = _bounded_text(frontmatter.get("name"), MAX_FRONTMATTER_FIELD_CHARS)
    description = _bounded_text(frontmatter.get("description"), MAX_FRONTMATTER_FIELD_CHARS)
    if name is None or description is None:
        return None
    return CorpusRow(source_id, slug, version, f"{name}: {description}")


def _load_rows(path: Path, *, allow_source_sample: bool) -> tuple[list[CorpusRow], dict[str, int], dict[str, str]]:
    import pyarrow.parquet as pq

    if not allow_source_sample:
        names = {file.name for file in path.iterdir() if file.suffix.lower() == ".parquet"}
        if names != set(PARQUET_SHARDS):
            raise ValueError("Parquet shard names must match the pinned source files")
    rows = []
    slugs = set()
    source_hashes = {}
    seen = malformed = duplicate = 0
    for shard in PARQUET_SHARDS:
        source = path / shard
        if allow_source_sample and not source.is_file():
            continue
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if not allow_source_sample and digest != PARQUET_SHA256[shard]:
                raise ValueError(f"Parquet shard {shard} does not match the pinned SHA-256 checksum")
            source_hashes[shard] = digest
            stream.seek(0)
            parquet = pq.ParquetFile(stream)
            for batch in parquet.iter_batches(batch_size=256, columns=list(SOURCE_COLUMNS)):
                for value in batch.to_pylist():
                    seen += 1
                    row = project_row(value)
                    if row is None:
                        malformed += 1
                        continue
                    slug = row.slug.casefold()
                    if slug in slugs:
                        duplicate += 1
                        continue
                    slugs.add(slug)
                    rows.append(row)
        print(f"Projected {shard}: {len(rows):,} cumulative retained records", flush=True)
    if seen != PARQUET_ROWS and (not allow_source_sample or not 0 < seen < PARQUET_ROWS):
        raise ValueError(f"Expected {PARQUET_ROWS} pinned source rows, got {seen}")
    counts = {"source": seen, "malformed": malformed, "duplicate_slug": duplicate, "retained": len(rows)}
    return rows, counts, source_hashes


def _embed(rows: list[CorpusRow], model_dir: Path, *, threads: int, batch_size: int) -> np.ndarray:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(
        str(model_dir / MODEL_FILE), sess_options=options, providers=["CPUExecutionProvider"]
    )
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=MAX_TOKENS)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    order = sorted(range(len(rows)), key=lambda index: len(rows[index].embedding_text))
    vectors = np.empty((len(rows), MODEL_DIMENSION), dtype=np.float32)
    inputs = [input_.name for input_ in session.get_inputs()]
    for start in range(0, len(rows), batch_size):
        indices = order[start : start + batch_size]
        encodings = tokenizer.encode_batch([rows[index].embedding_text for index in indices])
        arrays = {
            "input_ids": np.array([encoding.ids for encoding in encodings], dtype=np.int64),
            "attention_mask": np.array([encoding.attention_mask for encoding in encodings], dtype=np.int64),
            "token_type_ids": np.array([encoding.type_ids for encoding in encodings], dtype=np.int64),
        }
        hidden = session.run(None, {name: arrays[name] for name in inputs})[0]
        weights = arrays["attention_mask"].astype(np.float32)[..., None]
        vectors[indices] = (hidden * weights).sum(axis=1) / weights.sum(axis=1)
        if start % (batch_size * 128) == 0:
            print(f"Embedded {start + len(indices):,} / {len(rows):,} records", flush=True)
    return vectors


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms == 0):
        raise ValueError("Embeddings must be finite nonzero vectors")
    return (vectors / norms).astype(np.float32)


def write_matrix(real_vectors: np.ndarray, output: Path, *, size: int, dimension: int, seed: int) -> None:
    """Write normalized real vectors followed by seeded perturbations of real clusters."""
    if real_vectors.ndim != 2 or not 0 < len(real_vectors) <= size or not 0 < real_vectors.shape[1] <= dimension:
        raise ValueError("Matrix dimensions must fit a nonempty real-vector prefix")
    real = _normalize(real_vectors)
    span = real.shape[1]
    projection_seed, padding_seed = np.random.SeedSequence(seed).spawn(2)
    projection_rng = np.random.default_rng(projection_seed)
    padding_rng = np.random.default_rng(padding_seed)
    projection, _ = np.linalg.qr(projection_rng.standard_normal((dimension, span)))
    projection = projection.T.astype(np.float32)

    descriptor, temp_name = tempfile.mkstemp(suffix=".npy", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temp_name)
    try:
        matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype="<f4", shape=(size, dimension))
        for start in range(0, len(real), PAD_BATCH_SIZE):
            block = real[start : start + PAD_BATCH_SIZE] @ projection
            matrix[start : start + len(block)] = _normalize(block)
        for start in range(len(real), size, PAD_BATCH_SIZE):
            sources = padding_rng.integers(len(real), size=PAD_BATCH_SIZE)
            strength = padding_rng.uniform(0.10, 0.75, (PAD_BATCH_SIZE, 1)).astype(np.float32)
            directions = _normalize(padding_rng.standard_normal((PAD_BATCH_SIZE, span), dtype=np.float32))
            block = (real[sources] + strength * directions) @ projection
            end = min(start + PAD_BATCH_SIZE, size)
            matrix[start:end] = _normalize(block)[: end - start]
        matrix.flush()
        del matrix
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", type=int, default=250_000)
    parser.add_argument("--dimension", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--allow-source-sample", action="store_true", help="Accept and label local samples or projections"
    )
    args = parser.parse_args(argv)
    if args.output.suffix != ".npy":
        parser.error("--output must use the .npy extension")
    sidecar = args.output.with_suffix(".json")
    if sidecar.resolve() == args.output.resolve() or (
        sidecar.exists() and args.output.exists() and sidecar.samefile(args.output)
    ):
        parser.error("The matrix and its JSON sidecar must be different files")
    if not (PARQUET_ROWS <= args.size <= 250_000 and MODEL_DIMENSION <= args.dimension <= 4096):
        parser.error("size must fit the pinned corpus and be at most 250000; dimension must be in [384, 4096]")
    if not (1 <= args.threads <= 8 and 1 <= args.batch_size <= 256 and args.seed >= 0):
        parser.error("threads must be in [1, 8], batch size in [1, 256], and seed nonnegative")
    if _sha256(args.model_dir / MODEL_FILE) != MODEL_SHA256:
        raise ValueError("ONNX model does not match the pinned file")
    if _sha256(args.model_dir / "tokenizer.json") != TOKENIZER_SHA256:
        raise ValueError("Tokenizer does not match the pinned file")
    rows, counts, source_hashes = _load_rows(args.parquet_dir, allow_source_sample=args.allow_source_sample)
    if not rows:
        raise ValueError("Pinned corpus contains no usable descriptions")
    vectors = _embed(rows, args.model_dir, threads=args.threads, batch_size=args.batch_size)
    write_matrix(vectors, args.output, size=args.size, dimension=args.dimension, seed=args.seed)
    metadata = {
        "profile": "description-minilm-int8-proxy",
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "parquet_conversion_revision": PARQUET_REVISION,
        "source_selection": "all verified upstream rows"
        if source_hashes == PARQUET_SHA256
        else "local sample or projection",
        "source_shards": [
            {"name": name, "sha256": digest, "verified_upstream": digest == PARQUET_SHA256[name]}
            for name, digest in source_hashes.items()
        ],
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "model_sha256": MODEL_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "description_extraction": "bounded SKILL.md YAML frontmatter with stripped name and description",
        "max_tokens": MAX_TOKENS,
        "counts": {**counts, "synthetic": args.size - len(rows), "total": args.size},
        "source_dimension": MODEL_DIMENSION,
        "dimension": args.dimension,
        "dtype": "<f4",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "padding": "seeded base-space perturbations of real centers, then dense orthonormal lift",
        "matrix_sha256": _sha256(args.output),
        "human_labeled_overlap_queries": 0,
    }
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
