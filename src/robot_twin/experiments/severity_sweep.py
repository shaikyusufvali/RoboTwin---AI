"""
RoboTwin-AI Fault Severity Sweep

Runs the existing controlled experiment scenarios across multiple
fault-severity levels.

This module intentionally reuses the existing ScenarioExecutor.
No diagnostic, simulation, telemetry, vision, or fault-injection
logic is duplicated here.

Experiment design
-----------------
6 scenarios × 4 severity levels = 24 controlled trials.

Severity levels:
    0.30
    0.50
    0.70
    0.90

The generated results are saved separately for each severity level
and as one combined JSON dataset.

Important research note
-----------------------
Changing fault severity is a controlled experimental factor.

This is not yet a stochastic repeated-trial benchmark because the
current experiment framework does not inject randomized sensor noise
or randomized physical parameters. That will be added later in the
robustness phase.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robot_twin.experiments.experiment_runner import (
    build_initial_scenarios,
)
from robot_twin.experiments.scenario_executor import (
    ScenarioExecutor,
)


LOGGER = logging.getLogger(
    "robotwin.severity_sweep"
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "experiments"
    / "severity_sweep"
)

SEVERITY_LEVELS = (
    0.30,
    0.50,
    0.70,
    0.90,
)

JOINT_INDEX = 2

SETTLING_STEPS = 240


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def utc_timestamp() -> str:
    """Return the current UTC timestamp."""

    return (
        datetime.now(
            timezone.utc
        ).isoformat()
    )


def severity_label(
    severity: float,
) -> str:
    """
    Convert severity into a filesystem-safe label.

    Example:
        0.30 -> severity_0_30
    """

    return (
        f"severity_{severity:.2f}"
        .replace(".", "_")
    )


def validate_severity(
    severity: float,
) -> float:
    """Validate a fault severity."""

    severity = float(
        severity
    )

    if not 0.0 <= severity <= 1.0:
        raise ValueError(
            "Fault severity must be between "
            "0.0 and 1.0."
        )

    return severity


# ---------------------------------------------------------------------------
# Severity Sweep
# ---------------------------------------------------------------------------

class SeveritySweepRunner:
    """
    Execute the existing RoboTwin-AI scenarios across
    controlled fault-severity levels.
    """

    def __init__(
        self,
        *,
        output_directory: Path = OUTPUT_DIRECTORY,
        joint_index: int = JOINT_INDEX,
        settling_steps: int = SETTLING_STEPS,
    ) -> None:

        self.output_directory = Path(
            output_directory
        )

        self.joint_index = int(
            joint_index
        )

        self.settling_steps = int(
            settling_steps
        )

        if self.joint_index < 0:
            raise ValueError(
                "joint_index must be non-negative."
            )

        if self.settling_steps <= 0:
            raise ValueError(
                "settling_steps must be greater than zero."
            )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # -----------------------------------------------------------------------
    # Scenario execution
    # -----------------------------------------------------------------------

    def run_severity(
        self,
        severity: float,
    ) -> list[dict[str, Any]]:
        """
        Run all six existing scenarios at one severity level.
        """

        severity = validate_severity(
            severity
        )

        LOGGER.info(
            "Starting severity sweep level %.2f",
            severity,
        )

        scenarios = (
            build_initial_scenarios()
        )

        executor = ScenarioExecutor(
            joint_index=self.joint_index,
            fault_severity=severity,
            settling_steps=self.settling_steps,
            output_directory=self.output_directory,
        )

        results = executor.execute_all(
            scenarios
        )

        serialized_results = [
            asdict(result)
            for result in results
        ]

        successful = sum(
            1
            for result in results
            if result.execution_status
            == "SUCCESS"
        )

        LOGGER.info(
            "Severity %.2f completed: "
            "%d/%d successful",
            severity,
            successful,
            len(results),
        )

        return serialized_results

    # -----------------------------------------------------------------------
    # Save individual severity
    # -----------------------------------------------------------------------

    def save_severity_results(
        self,
        severity: float,
        results: list[dict[str, Any]],
    ) -> Path:
        """Save results for one severity level."""

        label = severity_label(
            severity
        )

        output_file = (
            self.output_directory
            / f"{label}.json"
        )

        payload = {
            "experiment": (
                "RoboTwin-AI fault severity sweep"
            ),
            "schema_version": "1.0",
            "created_at_utc": utc_timestamp(),
            "severity": severity,
            "joint_index": self.joint_index,
            "settling_steps": self.settling_steps,
            "scenario_count": len(results),
            "results": results,
        }

        output_file.write_text(
            json.dumps(
                payload,
                indent=4,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        LOGGER.info(
            "Saved severity %.2f results to %s",
            severity,
            output_file,
        )

        return output_file

    # -----------------------------------------------------------------------
    # Combined dataset
    # -----------------------------------------------------------------------

    def save_combined_results(
        self,
        all_results: list[dict[str, Any]],
    ) -> Path:
        """
        Save all severity levels into one experiment dataset.
        """

        output_file = (
            self.output_directory
            / "severity_sweep_results.json"
        )

        successful = sum(
            1
            for result in all_results
            if result.get(
                "execution_status"
            ) == "SUCCESS"
        )

        failed = (
            len(all_results)
            - successful
        )

        payload = {
            "experiment": (
                "RoboTwin-AI controlled "
                "fault severity sweep"
            ),
            "schema_version": "1.0",
            "created_at_utc": utc_timestamp(),

            "experimental_design": {
                "severity_levels": [
                    float(level)
                    for level in SEVERITY_LEVELS
                ],
                "scenarios_per_level": 6,
                "total_trials": len(
                    all_results
                ),
                "successful_trials": successful,
                "failed_trials": failed,
                "joint_index": self.joint_index,
                "settling_steps": self.settling_steps,
            },

            "research_notes": {
                "controlled_factor": (
                    "fault_severity"
                ),
                "stochastic_noise_injected": False,
                "randomized_physics_parameters": False,
                "final_benchmark": False,
                "purpose": (
                    "Measure diagnostic behavior "
                    "across controlled fault severity."
                ),
            },

            "results": all_results,
        }

        output_file.write_text(
            json.dumps(
                payload,
                indent=4,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        return output_file

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------

    @staticmethod
    def print_summary(
        all_results: list[dict[str, Any]],
    ) -> None:
        """Print the complete severity-sweep summary."""

        print()
        print("=" * 110)
        print(
            "ROBOTWIN-AI FAULT SEVERITY SWEEP"
        )
        print("=" * 110)

        print()
        print(
            f"Total trials : "
            f"{len(all_results)}"
        )

        successful = sum(
            1
            for result in all_results
            if result.get(
                "execution_status"
            ) == "SUCCESS"
        )

        failed = (
            len(all_results)
            - successful
        )

        print(
            f"Successful   : "
            f"{successful}"
        )

        print(
            f"Failed       : "
            f"{failed}"
        )

        print()
        print(
            f"{'SEVERITY':<12}"
            f"{'SCENARIO':<10}"
            f"{'TRUTH':<24}"
            f"{'PREDICTION':<24}"
            f"{'CONFIDENCE':>12}"
            f"{'STATUS':>12}"
        )

        print("-" * 110)

        for result in all_results:

            severity = result.get(
                "fault_severity",
                0.0,
            )

            scenario_id = result.get(
                "scenario_id",
                "UNKNOWN",
            )

            truth = result.get(
                "ground_truth_fault",
                "UNKNOWN",
            )

            prediction = result.get(
                "multimodal_fault",
                "UNKNOWN",
            )

            confidence = result.get(
                "multimodal_confidence",
                0.0,
            )

            status = result.get(
                "execution_status",
                "UNKNOWN",
            )

            print(
                f"{severity:<12.2f}"
                f"{scenario_id:<10}"
                f"{truth:<24}"
                f"{prediction:<24}"
                f"{confidence:>12.4f}"
                f"{status:>12}"
            )

        print("-" * 110)

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "This is a controlled severity sweep, "
            "not the final robustness benchmark."
        )

        print(
            "Sensor-noise, randomized operating "
            "conditions, and repeated stochastic "
            "trials will be evaluated later."
        )

        print("=" * 110)

    # -----------------------------------------------------------------------
    # Full run
    # -----------------------------------------------------------------------

    def run(
        self,
        severity_levels: tuple[
            float,
            ...,
        ] = SEVERITY_LEVELS,
    ) -> Path:
        """
        Execute all severity levels.
        """

        if not severity_levels:
            raise ValueError(
                "severity_levels cannot be empty."
            )

        all_results: list[
            dict[str, Any]
        ] = []

        for severity in severity_levels:

            severity_results = (
                self.run_severity(
                    severity
                )
            )

            self.save_severity_results(
                severity=severity,
                results=severity_results,
            )

            all_results.extend(
                severity_results
            )

        combined_file = (
            self.save_combined_results(
                all_results
            )
        )

        self.print_summary(
            all_results
        )

        print()
        print(
            "Combined dataset saved to:"
        )
        print(
            combined_file
        )
        print()

        return combined_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the RoboTwin-AI severity sweep."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    runner = SeveritySweepRunner(
        joint_index=JOINT_INDEX,
        settling_steps=SETTLING_STEPS,
    )

    runner.run()


if __name__ == "__main__":
    main()