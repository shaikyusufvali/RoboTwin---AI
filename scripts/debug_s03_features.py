"""Inspect the runtime telemetry feature vector for S03."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import joblib
import pybullet as p

from robot_twin.diagnostics.fault_classifier import FEATURES
from robot_twin.diagnostics.fault_injector import (
    FaultInjector,
    FaultScenario,
    FaultType,
)
from robot_twin.simulation.robot_simulator import RobotSimulator
from robot_twin.telemetry.pybullet_adapter import PyBulletRobotAdapter


DATASET_FILE = Path("data/robot_diagnostics_dataset.csv")
MODEL_FILE = Path("data/robot_fault_classifier_clean.joblib")

JOINT_INDEX = 2
SEVERITY = 0.7
SETTLING_STEPS = 240
TIME_STEP = 1.0 / 240.0


def main() -> None:
    print()
    print("=" * 80)
    print("ROBOTWIN-AI S03 RUNTIME FEATURE DIAGNOSTIC")
    print("=" * 80)

    simulator = RobotSimulator()

    injector = FaultInjector(simulator)

    adapter = PyBulletRobotAdapter(
        simulator=simulator,
        fault_injector=injector,
    )

    try:
        adapter.connect()

        scenario = FaultScenario(
            fault_type=FaultType.LOW_MOTOR_POWER,
            joint_index=JOINT_INDEX,
            severity=SEVERITY,
        )

        adapter.apply_fault(scenario)

        # Match the controlled experiment settling procedure.
        for _ in range(SETTLING_STEPS):
            injector.apply(scenario)
            simulator.step()

        telemetry = adapter.get_joint_telemetry(JOINT_INDEX)

        fault_telemetry = injector.get_fault_telemetry(JOINT_INDEX)

        commanded_position = fault_telemetry.target_position

        position_error = (
            commanded_position - telemetry.position
        )

        features = {
            "joint_index": float(JOINT_INDEX),
            "commanded_position": float(commanded_position),
            "position": float(telemetry.position),
            "position_error": float(position_error),
            "velocity": float(telemetry.velocity),
            "acceleration": float(
                getattr(telemetry, "acceleration", 0.0)
            ),
            "force": float(telemetry.force),
            "motor_current": float(telemetry.motor_current),
            "temperature": float(telemetry.temperature),
        }

        print()
        print("GROUND TRUTH")
        print("-" * 80)
        print("Fault type : LOW_MOTOR_POWER")
        print(f"Joint      : {JOINT_INDEX}")
        print(f"Severity   : {SEVERITY}")

        print()
        print("RUNTIME TELEMETRY")
        print("-" * 80)

        for feature in FEATURES:
            print(
                f"{feature:20s}: "
                f"{features[feature]: .8f}"
            )

        # ------------------------------------------------------------------
        # Load model and inspect prediction.
        # ------------------------------------------------------------------

        artifact = joblib.load(MODEL_FILE)

        model = artifact["model"]

        model_features = artifact["features"]

        vector = [
            features[name]
            for name in model_features
        ]

        prediction = model.predict([vector])[0]

        probabilities = model.predict_proba([vector])[0]

        classes = model.classes_

        print()
        print("MODEL PREDICTION")
        print("-" * 80)
        print(f"Predicted fault : {prediction}")

        print()
        print("CLASS PROBABILITIES")
        print("-" * 80)

        ranked = sorted(
            zip(classes, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )

        for name, probability in ranked:
            print(
                f"{str(name):20s}: "
                f"{probability:.6f}"
            )

        # ------------------------------------------------------------------
        # Dataset statistics for each fault.
        # ------------------------------------------------------------------

        stats = defaultdict(
            lambda: defaultdict(list)
        )

        with DATASET_FILE.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                fault = row["fault_type"]

                for feature in FEATURES:
                    try:
                        value = float(row[feature])
                    except (TypeError, ValueError):
                        continue

                    stats[fault][feature].append(value)

        print()
        print("DISTANCE FROM DATASET CLASS MEANS")
        print("-" * 80)

        for fault in (
            "normal",
            "low_motor_power",
            "motor_resistance",
            "position_drift",
        ):
            print()
            print(f"[{fault}]")

            for feature in FEATURES:
                values = stats[fault][feature]

                if not values:
                    continue

                mean = sum(values) / len(values)

                minimum = min(values)
                maximum = max(values)

                runtime_value = features[feature]

                print(
                    f"{feature:20s} "
                    f"runtime={runtime_value: .5f} "
                    f"mean={mean: .5f} "
                    f"range=[{minimum: .5f}, {maximum: .5f}]"
                )

        print()
        print("=" * 80)
        print("END S03 FEATURE DIAGNOSTIC")
        print("=" * 80)

    finally:
        try:
            adapter.disconnect()
        except Exception:
            simulator.close()


if __name__ == "__main__":
    main()