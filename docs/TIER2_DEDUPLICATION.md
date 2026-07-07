# Tier 2: Semantic Deduplication

Tier 2 covers two complementary workflows:

- **Intra-skill deduplication** finds repeated or overlapping guidance inside
  one skill before it bloats the agent's context window.
- **Inter-skill similarity** compares skills directly or checks one candidate
  against a versioned local catalog.

Both workflows run from local files. There is no Milvus, vector database,
catalog server, or NVIDIA-internal service dependency.

All three commands need an embeddings API: an LLM provider key
(see [CONFIGURATION.md](CONFIGURATION.md)) or any local OpenAI-compatible
endpoint (see the
[fully-local recipe](CONFIGURATION.md#fully-local-tier-2-no-external-calls)).

### Data sent to configured providers

- In the default inter-skill mode, skill names and descriptions are sent to
  the configured embeddings provider. With `--full-body`, the full manifest is sent instead.
- For intra-skill analysis, scannable content chunks are sent to the embeddings
  provider. Only overlapping candidate clusters are sent to the configured
  chat LLM for classification.

Use the fully local provider recipe above when this content must not leave the
machine. Catalog files remain local unless you explicitly share them.

## Intra-skill deduplication

```bash
skillevaluator context-optimization-check ./my-skill  # redundant content (threshold 0.8)
skillevaluator dedup-scan ./my-skill                  # alias for the command above
```

`context-optimization-check` clusters overlapping sections with embeddings and
uses a chat LLM to explain the overlap. `dedup-scan` is an alias with the same
options and behavior. Use `--threshold` to tune sensitivity.

## Inter-skill similarity

Compare every skill in a directory directly:

```bash
skillevaluator similarity-check ./skills  # embeddings only; default threshold 0.75
```

For repeated checks, save the collection as a local catalog and compare one
candidate skill against it:

```bash
skillevaluator similarity-check ./skills --save-catalog ./skill-catalog.json
skillevaluator similarity-check ./candidate-skill --catalog ./skill-catalog.json
```

`--save-catalog` builds the catalog from the supplied collection. `--catalog`
requires an existing catalog and compares only the supplied target against its
entries; it does not compare catalog entries with one another or silently
rebuild a missing catalog.

Catalogs use a versioned JSON schema and contain finite embedding vectors,
relative skill paths, display names, descriptions, content fingerprints, and
the provider/model metadata needed for compatibility checks. They do not
contain API keys. A malformed, incompatible, non-finite, or oversized catalog
is rejected with an actionable error; rebuild it with the provider and model
you intend to query. Because the file includes derived skill content, review it
before sharing it like any other generated project artifact.

Direct scans and catalog queries accept `--threshold` to tune sensitivity.

## Where the LLM comes in

- `similarity-check` is embeddings-only: no chat LLM is called.
- `context-optimization-check` and its `dedup-scan` alias use embeddings to
  find overlap candidates, then a chat LLM to analyze them. Both accept
  `--llm-model` to override the analysis model.

## Reports and exit behavior

All workflows support CLI, JSON, HTML, and Markdown output:

```bash
skillevaluator similarity-check ./skills \
  -r cli,json,html,markdown -o ./reports
```

Findings at or above the configured threshold are reported as Tier 2 results.
Blocking findings produce a non-zero exit code in every output mode, making the
commands usable as CI gates.

Tier 2 rejects linked or escaping input files before provider calls and applies
explicit input, chunk, catalog, and comparison limits. If a project exceeds a
limit, split the collection into intentional batches rather than scanning an
unbounded tree.

`validate` runs the Tier 2 dedup pass by default and skips it gracefully
when no embedding provider is configured; pass `--no-dedup` to turn it off
explicitly.
