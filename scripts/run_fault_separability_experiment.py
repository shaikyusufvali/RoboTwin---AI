from __future__ import annotations

import csv
from pathlib import Path

from robot_twin.simulation.robot_simulator import RobotSimulator
from robot_twin.telemetry.pybullet_adapter import PyBulletRobotAdapter
from robot_twin.diagnostics.fault_injector import (
    FaultScenario,
    FaultType,
)
from robot_twin.diagnostics.fault_signature_engine import (
    FaultSignatureEngine,
)


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

OUTPUT_DIR = Path("data") / "experiments" / "fault_separability"
OUTPUT_FILE = OUTPUT_DIR / "fault_signature_separability.csv"


def run_fault(
    fault_type: FaultType,
    severity: float,
) -> dict:
    simulator = RobotSimulator()
    adapter = PyBulletRobotAdapter(simulator)
    signature_engine = FaultSignatureEngine()

    try:
        adapter.connect()

        scenario = FaultScenario(
            fault_type=fault_type,
            joint_index=JOINT_INDEX,
            severity=severity,
        )

        adapter.apply_fault(scenario)

        for _ in range(SETTLE_STEPS):
            simulator.step()

        telemetry = adapter.get_joint_telemetry(
            joint_index=JOINT_INDEX
        )

        signature = signature_engine.build_signature(
            telemetry=telemetry,
            target_position=TARGET_POSITION,
        )

        result = signature.as_dict()

        return {
            "severity": severity,
            "fault_type": fault_type.value,
            **result,
        }

    finally:
        adapter.disconnect()


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict] = []

    print("\n" + "=" * 110)
    print("ROBOTWIN-AI FAULT SEPARABILITY EXPERIMENT")
    print("=" * 110)
    print(f"Joint       : {JOINT_INDEX}")
    print(f"Target      : {TARGET_POSITION}")
    print(f"Settle steps: {SETTLE_STEPS}")
    print(f"Severities  : {SEVERITIES}")
    print("=" * 110)

    for severity in SEVERITIES:
        print(
            f"\n{'=' * 30} "
            f"SEVERITY {severity:.2f} "
            f"{'=' * 30}"
        )

        for fault_type in FAULT_TYPES:
            print(
                f"Running "
                f"{fault_type.value:<20} "
                f"severity={severity:.2f}"
            )

            result = run_fault(
                fault_type=fault_type,
                severity=severity,
            )

            rows.append(result)

            print(
                f"  current={result['motor_current']:.4f} | "
                f"temperature={result['temperature']:.4f} | "
                f"force={result['force_magnitude']:.4f} | "
                f"velocity={result['velocity_magnitude']:.4f}"
            )

    fieldnames = [
        "severity",
        "fault_type",
        "position_error",
        "velocity_magnitude",
        "force_magnitude",
        "motor_current",
        "temperature",
        "effort_level",
        "motion_level",
        "position_deviation",
        "thermal_state",
        "persistence_score",
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

    print("\n" + "=" * 110)
    print("FAULT SEPARABILITY EXPERIMENT COMPLETE")
    print("=" * 110)
    print(f"Samples : {len(rows)}")
    print(f"Saved   : {OUTPUT_FILE}")
    print("=" * 110)


if __name__ == "__main__":
    main()