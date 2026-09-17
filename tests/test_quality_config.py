from pathlib import Path

from skillevaluator.quality_config import load_quality_config


def test_packaged_default_quality_config_is_available() -> None:
    config = load_quality_config()

    assert config["limitations"] == {
        "required": True,
        "heading": "## Limitations",
    }


def test_quality_config_can_be_overridden(tmp_path: Path) -> None:
    config_path = tmp_path / "quality.yaml"
    config_path.write_text(
        'limitations:\n  required: true\n  heading: "## Failure Constellations"\n',
        encoding="utf-8",
    )

    config = load_quality_config(config_path)

    assert config["limitations"] == {
        "required": True,
        "heading": "## Failure Constellations",
    }
