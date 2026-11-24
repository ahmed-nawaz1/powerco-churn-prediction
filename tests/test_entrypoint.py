"""Simple root-entrypoint and saved-result integrity checks."""

import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from powerco.config import DEFAULT_JOBS, RUNS_DIR
from powerco.io import sha, write_json
from powerco.validate import validate


class EntryPointTests(unittest.TestCase):
    def test_main_creates_a_new_run_path(self):
        with patch("powerco.workflow.run") as training:
            import main as entry

            importlib.reload(entry)
            self.assertEqual(entry.main(), 0)
            output = training.call_args.args[0]
            jobs = training.call_args.kwargs["jobs"]
            self.assertEqual(output.parent, RUNS_DIR)
            self.assertEqual(jobs, DEFAULT_JOBS)

    def test_importing_main_does_not_train(self):
        with patch("powerco.workflow.run") as training:
            import main as entry

            importlib.reload(entry)
            training.assert_not_called()

    def test_result_tampering_is_detected_before_model_loading(self):
        with tempfile.TemporaryDirectory(prefix="powerco_integrity_") as folder:
            saved = Path(folder)
            result = saved / "result.csv"
            result.write_text("original", encoding="utf-8")
            write_json(saved / "checksums.json", {"result.csv": sha(result)})
            write_json(saved / "manifest.json", {"completed_utc": "fixture"})
            result.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Changed result"):
                validate(saved)


if __name__ == "__main__":
    unittest.main()
