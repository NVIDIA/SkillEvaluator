"""Measure SARIF rendering for repeated paths; run from an editable install."""

import hashlib
import json
import statistics
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.reporting import sarif_reporter


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        scan = root / "skills" / "example"
        scan.mkdir(parents=True)
        paths = [f"src/file {index}.py" for index in range(100)]
        (scan / "src").mkdir()
        for path in paths:
            (scan / path).write_text("print('example')\n" * 50)
        results = [ValidationResult(validator_name=f"CHECK-{index}") for index in range(5)]
        for path in paths:
            for index in range(50):
                results[index % len(results)].add_finding(
                    Finding(
                        category="STATIC",
                        severity=Severity.MEDIUM,
                        check_name=f"rule-{index % 10}",
                        message=f"Example diagnostic {index}",
                        file_path=path,
                        line_number=index + 1,
                        line_content="print('example')",
                    )
                )
        reporter = sarif_reporter.SARIFReporter(include_timestamp=False, workspace_root=root, scan_root=scan)
        output = reporter.render_all(results)
        times = []
        for _ in range(5):
            start = time.perf_counter()
            assert reporter.render_all(results) == output
            times.append(time.perf_counter() - start)
        with patch.object(sarif_reporter.os.path, "realpath", wraps=sarif_reporter.os.path.realpath) as realpath:
            assert reporter.render_all(results) == output
        print(
            json.dumps(
                {
                    "files": len(paths),
                    "findings": sum(len(result.findings) for result in results),
                    "validators": len(results),
                    "seconds": times,
                    "median_seconds": statistics.median(times),
                    "realpath_calls": realpath.call_count,
                    "report_bytes": len(output.encode()),
                    "sha256": hashlib.sha256(output.encode()).hexdigest(),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
