"""Analyze recorded robot telemetry using expected behavior."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Final

from robot_twin.diagnostics.anomaly_detector import (
    AnomalyDetector,
)
from robot_twin.diagnostics.behavior_analyzer import (
    BehaviorAnalyzer,
)
from robot_twin.telemetry.robot_adapter import (
    JointTelemetry,
)


LOGGER = logging.getLogger(__name__)

INPUT_FILE: Final[Path] = (
    Path("data") / "motion_robot_telemetry.csv"
)

OUTPUT_FILE: Final[Path] = (
    Path("data") / "analyzed_robot_telemetry.csv"
)


class TelemetryAnalyzer:
    """Analyze recorded telemetry for behavioral deviations."""

    def __init__(
        self,
        anomaly_detector: AnomalyDetector | None = None,
        behavior_analyzer: BehaviorAnalyzer | None = None,
    ) -> None:
        self._anomaly_detector = (
            anomaly_detector or AnomalyDetector()
        )
        self._behavior_analyzer = (
            behavior_analyzer or BehaviorAnalyzer()
        )

    def analyze(
        self,
        input_file: Path,
        output_file: Path,
    ) -> tuple[int, int]:
        """Analyze telemetry and save diagnostic features."""

        if not input_file.exists():
            raise FileNotFoundError(
                f"Telemetry file not found: {input_file}"
            )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        total_records = 0
        anomaly_records = 0

        with input_file.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as source:
            reader = csv.DictReader(source)

            if reader.fieldnames is None:
                raise ValueError(
                    "Telemetry file has no header."
                )

            required_columns = {
                "joint_index",
                "position",
                "velocity",
                "force",
                "motor_current",
                "temperature",
                "target_position",
            }

            missing_columns = (
                required_columns
                - set(reader.fieldnames)
            )

            if missing_columns:
                raise ValueError(
                    "Missing telemetry columns: "
                    + ", ".join(sorted(missing_columns))
                )

            fieldnames = [
                *reader.fieldnames,
                "position_error",
                "expected_position",
                "expected_velocity",
                "expected_acceleration",
                "expected_force",
                "expected_motor_current",
                "expected_temperature",
                "position_deviation",
                "velocity_deviation",
                "acceleration_deviation",
                "force_deviation",
                "motor_current_deviation",
                "temperature_deviation",
                "behavior_anomaly",
                "behavior_anomaly_score",
                "behavior_anomaly_reasons",
                "threshold_anomaly",
                "threshold_anomaly_score",
                "threshold_anomaly_reasons",
            ]

            with output_file.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as destination:
                writer = csv.DictWriter(
                    destination,
                    fieldnames=fieldnames,
                )
                writer.writeheader()

                for row in reader:
                    telemetry = JointTelemetry(
                        joint_index=int(
                            row["joint_index"]
                        ),
                        position=float(
                            row["position"]
                        ),
                        velocity=float(
                            row["velocity"]
                        ),
                        force=float(
                            row["force"]
                        ),
                        motor_current=float(
                            row["motor_current"]
                        ),
                        temperature=float(
                            row["temperature"]
                        ),
                    )

                    target_position = float(
                        row["target_position"]
                    )

                    position_error = (
                        target_position
                        - telemetry.position
                    )

                    behavior_result = (
                        self._behavior_analyzer.analyze(
                            telemetry=telemetry,
                            target_position=target_position,
                        )
                    )

                    threshold_result = (
                        self._anomaly_detector.detect(
                            telemetry=telemetry,
                            target_position=target_position,
                        )
                    )

                    if (
                        behavior_result.anomaly
                        or threshold_result.is_anomaly
                    ):
                        anomaly_records += 1

                    deviation = (
                        behavior_result.deviation
                    )
                    expected = (
                        behavior_result.expected
                    )

                    analyzed_row = {
                        **row,
                        "position_error": (
                            f"{position_error:.6f}"
                        ),
                        "expected_position": (
                            f"{expected.position:.6f}"
                        ),
                        "expected_velocity": (
                            f"{expected.velocity:.6f}"
                        ),
                        "expected_acceleration": (
                            f"{expected.acceleration:.6f}"
                        ),
                        "expected_force": (
                            f"{expected.force:.6f}"
                        ),
                        "expected_motor_current": (
                            f"{expected.motor_current:.6f}"
                        ),
                        "expected_temperature": (
                            f"{expected.temperature:.6f}"
                        ),
                        "position_deviation": (
                            f"{deviation.position:.6f}"
                        ),
                        "velocity_deviation": (
                            f"{deviation.velocity:.6f}"
                        ),
                        "acceleration_deviation": (
                            f"{deviation.acceleration:.6f}"
                        ),
                        "force_deviation": (
                            f"{deviation.force:.6f}"
                        ),
                        "motor_current_deviation": (
                            f"{deviation.motor_current:.6f}"
                        ),
                        "temperature_deviation": (
                            f"{deviation.temperature:.6f}"
                        ),
                        "behavior_anomaly": str(
                            behavior_result.anomaly
                        ),
                        "behavior_anomaly_score": (
                            f"{behavior_result.anomaly_score:.6f}"
                        ),
                        "behavior_anomaly_reasons": (
                            " | ".join(
                                behavior_result.reasons
                            )
                        ),
                        "threshold_anomaly": str(
                            threshold_result.is_anomaly
                        ),
                        "threshold_anomaly_score": (
                            f"{threshold_result.anomaly_score:.6f}"
                        ),
                        "threshold_anomaly_reasons": (
                            " | ".join(
                                threshold_result.reasons
                            )
                        ),
                    }

                    writer.writerow(analyzed_row)
                    total_records += 1

        LOGGER.info(
            "Telemetry analysis completed. "
            "Records=%d | anomalies=%d",
            total_records,
            anomaly_records,
        )

        return total_records, anomaly_records


def main() -> None:
    """Run telemetry analysis."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    analyzer = TelemetryAnalyzer()

    total_records, anomaly_records = (
        analyzer.analyze(
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE,
        )
    )

    normal_records = (
        total_records - anomaly_records
    )

    print()
    print("ROBOTWIN-AI TELEMETRY ANALYSIS")
    print("=" * 45)
    print(f"Total records : {total_records}")
    print(f"Normal        : {normal_records}")
    print(f"Anomalies     : {anomaly_records}")
    print(f"Output        : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()