"""Stable project paths and predeclared experiment settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
ASSESSMENT_DIR = ROOT / "docs" / "assessment"
AUDIT_DIR = ROOT / "docs" / "audit"
RUNS_DIR = ROOT / "artifacts" / "runs"
LATEST_RUN = ROOT / "artifacts" / "latest_run.json"
SEED = 42
N_SPLITS = 5
DEFAULT_JOBS = 3

# Historical inputs moved without changing their contents or original manifest.
LEGACY_INPUT_PATHS = {
    "main.py": "legacy/main.py",
    "client_data.csv": "data/raw/client_data.csv",
    "price_data.csv": "data/raw/price_data.csv",
    "cps7008-assessment-1.pdf": "docs/assessment/cps7008-assessment-1.pdf",
    "audit/PROTOCOL.md": "docs/audit/PROTOCOL.md",
}


def project_path(value: str | Path) -> Path:
    """Resolve relative paths from the project, independently of shell cwd."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()
