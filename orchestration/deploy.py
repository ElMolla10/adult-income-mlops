"""
Register the training flow as a schedulable Prefect deployment.

This uses Prefect 3.x's `flow.serve()` API which:
  1. Creates a deployment named 'adult-income-weekly'
  2. Registers a weekly cron schedule (Sundays 02:00 UTC)
  3. Starts a long-running worker that polls for triggered runs
     and executes scheduled runs automatically.

Run this in a dedicated terminal AFTER `prefect server start`:

    python orchestration/deploy.py

Then, in the Prefect UI at http://localhost:4200:
  - Navigate to Deployments → adult_income_training_pipeline / adult-income-weekly
  - Click "Quick run" to trigger ad-hoc, or wait for the scheduled run

CLI alternative:
    prefect deployment run "adult_income_training_pipeline/adult-income-weekly"

To stop the worker: Ctrl+C in this terminal.
"""
import sys
from pathlib import Path

# Make the project importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from orchestration.flows.training_flow import training_pipeline


def main() -> None:
    print("=" * 70)
    print("Registering Prefect deployment: adult-income-weekly")
    print("Schedule: every Sunday at 02:00 UTC (cron: '0 2 * * 0')")
    print("UI:       http://localhost:4200")
    print("=" * 70)
    print()
    print("Starting worker — leave this terminal open.")
    print("Trigger ad-hoc runs from the Prefect UI or via CLI:")
    print('  prefect deployment run "adult_income_training_pipeline/adult-income-weekly"')
    print()

    # serve() creates the deployment AND keeps a worker running to execute it.
    training_pipeline.serve(
        name="adult-income-weekly",
        cron="0 2 * * 0",  # Sundays 02:00 UTC
        tags=["mlops", "training", "adult-income"],
        description=(
            "Weekly retraining of the Adult Income classifier. "
            "Runs the full 6-stage pipeline: prepare → validate → preprocess "
            "→ train → evaluate → register."
        ),
        version="1.0.0",
    )


if __name__ == "__main__":
    main()
