"""Run the complete PowerCo churn experiment with one simple command."""

from datetime import datetime
from zoneinfo import ZoneInfo

from powerco import workflow
from powerco.config import DEFAULT_JOBS, RUNS_DIR


def main() -> int:
    """Create a new run, execute all    seven stages and save   the   evidence."""

    run_id = datetime.now(ZoneInfo("Asia/Karachi")).strftime("%Y%m%d_%H%M%S_%f")
    workflow.run(RUNS_DIR / run_id, jobs=DEFAULT_JOBS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
