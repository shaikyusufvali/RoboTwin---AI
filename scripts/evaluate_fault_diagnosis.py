from __future__ import annotations

import csv
from pathlib import Path

from robot_twin.simulation.robot_simulator import RobotSimulator
from robot_twin.telemetry.pybullet_adapter import PyBulletRobotAdapter
from robot_twin.diagnostics.fault_injector import (
    FaultInjector,
    FaultScenario,
    FaultType,
)
from robot_twin.diagnostics.diagnostic_brain import DiagnosticBrain


# ============================================================================
# Configuration
# ============================================================================

JOINT_INDEX = 2
TARGET_POSITION = 0.8

SEVERITIES = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]

FAULT_TYPES = [
    FaultType.NORMAL,
    FaultType.MOTOR_RESISTANCE,
    FaultType.LOW_MOTOR_POWER,
    FaultType.POSITION_DRIFT,
]

SETTLE_STEPS = 240

OUTPUT_DIR = Path(
    "data"
) / "experiments" / "fault_diagnosis_evaluation"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "diagnostic_results.csv"
)


# ============================================================================
# Helpers
# ============================================================================

def normalize_fault(value: object) -> str:
    """
    Convert enum/string fault representation into a stable string.
    """

    if hasattr(value, "value"):
        return str(value.value)

    return str(value)


def run_case(
    fault_type: FaultType,
    severity: float,
) -> dict:
    """
    Run one controlled fault scenario through the existing
    DiagnosticBrain.

    No replacement diagnostic logic is implemented here.
    """

    simulator = RobotSimulator()

    injector = FaultInjector(
        simulator
    )

    adapter = PyBulletRobotAdapter(
        simulator=simulator,
        fault_injector=injector,
    )

    diagnostic_brain = DiagnosticBrain()

    try:
        adapter.connect()

        # --------------------------------------------------------------
        # Configure controlled fault
        # --------------------------------------------------------------

        scenario = FaultScenario(
            fault_type=fault_type,
            joint_index=JOINT_INDEX,
            severity=severity,
        )

        injector.clear(
            joint_index=JOINT_INDEX
        )

        adapter.apply_fault(
            scenario
        )

        # --------------------------------------------------------------
        # Allow physics to settle
        # --------------------------------------------------------------

        for _ in range(SETTLE_STEPS):
            simulator.step()

        # --------------------------------------------------------------
        # Read live telemetry
        # --------------------------------------------------------------

        telemetry = adapter.get_joint_telemetry(
            joint_index=JOINT_INDEX
        )

        # --------------------------------------------------------------
        # Existing DiagnosticBrain
        # --------------------------------------------------------------

        result = diagnostic_brain.diagnose(
            telemetry=telemetry,
            target_position=TARGET_POSITION,
        )

        actual_fault = normalize_fault(
            fault_type
        )

        predicted_fault = normalize_fault(
            result.fault_type
        )

        confidence = float(
            result.confidence
        )

        return {
            "severity": severity,
            "actual_fault": actual_fault,
            "predicted_fault": predicted_fault,
            "confidence": confidence,
            "correct": (
                actual_fault
                == predicted_fault
            ),
            "position_error": float(
                telemetry.position
                - TARGET_POSITION
            ),
            "velocity": float(
                telemetry.velocity
            ),
            "force": float(
                telemetry.force
            ),
            "motor_current": float(
                telemetry.motor_current
            ),
            "temperature": float(
                telemetry.temperature
            ),
        }

    finally:
        adapter.disconnect()


# ============================================================================
# Evaluation
# ============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict] = []

    total = (
        len(SEVERITIES)
        * len(FAULT_TYPES)
    )

    completed = 0

    print()
    print(
        "FAULT DIAGNOSIS EVALUATION"
    )
    print(
        "=" * 100
    )

    for severity in SEVERITIES:

        for fault_type in FAULT_TYPES:

            result = run_case(
                fault_type=fault_type,
                severity=severity,
            )

            rows.append(result)

            completed += 1

            status = (
                "PASS"
                if result["correct"]
                else "FAIL"
            )

            print(
                f"[{completed:02d}/{total}] "
                f"severity={severity:.2f} | "
                f"actual={result['actual_fault']:<20} | "
                f"predicted={result['predicted_fault']:<20} | "
                f"confidence={result['confidence']:.4f} | "
                f"{status}"
            )

    # ------------------------------------------------------------------
    # Save raw evaluation results
    # ------------------------------------------------------------------

    fieldnames = [
        "severity",
        "actual_fault",
        "predicted_fault",
        "confidence",
        "correct",
        "position_error",
        "velocity",
        "force",
        "motor_current",
        "temperature",
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------------
    # Overall accuracy
    # ------------------------------------------------------------------

    correct_count = sum(
        1
        for row in rows
        if row["correct"]
    )

    accuracy = (
        correct_count / len(rows)
        if rows
        else 0.0
    )

    print()
    print(
        "=" * 100
    )
    print(
        "EVALUATION COMPLETE"
    )
    print(
        "=" * 100
    )

    print(
        f"Total cases : {len(rows)}"
    )

    print(
        f"Correct     : {correct_count}"
    )

    print(
        f"Incorrect   : "
        f"{len(rows) - correct_count}"
    )

    print(
        f"Accuracy    : {accuracy:.4f}"
    )

    print()
    print(
        f"Saved       : {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()