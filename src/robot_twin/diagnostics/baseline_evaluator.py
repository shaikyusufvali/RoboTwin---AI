"""Evaluate the healthy-baseline anomaly detector."""

from __future__ import annotations

import csv
import logging
import tempfile
from pathlib import Path
from typing import Final

from robot_twin.diagnostics.baseline_anomaly_detector import (
    BaselineAnomalyDetector,
)
from robot_twin.diagnostics.baseline_builder import (
    BaselineBuilder,
)
from robot_twin.telemetry.robot_adapter import (
    JointTelemetry,
)


LOGGER = logging.getLogger(__name__)

INPUT_FILE: Final[Path] = (
    Path("data") / "motion_robot_telemetry.csv"
)

TRAIN_RATIO: Final[float] = 0.80

POSITION_DRIFT: Final[float] = 0.30
FORCE_MULTIPLIER: Final[float] = 2.50
CURRENT_MULTIPLIER: Final[float] = 2.50
TEMPERATURE_INCREASE: Final[float] = 25.0


class BaselineEvaluator:
    """Evaluate anomaly detection using unseen healthy data."""

    def evaluate(
        self,
        input_file: Path,
    ) -> None:
        rows = self._load_rows(input_file)

        if len(rows) < 10:
            raise ValueError(
                "At least 10 telemetry records are required "
                "for evaluation."
            )

        split_index = int(
            len(rows) * TRAIN_RATIO
        )

        training_rows = rows[:split_index]
        test_rows = rows[split_index:]

        if not training_rows or not test_rows:
            raise ValueError(
                "Unable to create training and test datasets."
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            training_file = (
                temp_path / "baseline_training.csv"
            )

            baseline_file = (
                temp_path / "healthy_baseline.json"
            )

            self._write_rows(
                training_file,
                training_rows,
            )

            BaselineBuilder().build(
                input_file=training_file,
                output_file=baseline_file,
            )

            detector = BaselineAnomalyDetector(
                baseline_file=baseline_file,
            )

            healthy_results = self._evaluate_healthy(
                detector,
                test_rows,
            )

            fault_results = self._evaluate_faults(
                detector,
                test_rows,
            )

        self._print_results(
            healthy_results=healthy_results,
            fault_results=fault_results,
        )

    @staticmethod
    def _load_rows(
        input_file: Path,
    ) -> list[dict[str, str]]:
        if not input_file.exists():
            raise FileNotFoundError(
                f"Telemetry file not found: {input_file}"
            )

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

            return list(reader)

    @staticmethod
    def _write_rows(
        output_file: Path,
        rows: list[dict[str, str]],
    ) -> None:
        if not rows:
            raise ValueError(
                "Cannot write an empty telemetry dataset."
            )

        fieldnames = list(rows[0].keys())

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
            writer.writerows(rows)

    @staticmethod
    def _create_telemetry(
        row: dict[str, str],
        *,
        position_offset: float = 0.0,
        force_multiplier: float = 1.0,
        current_multiplier: float = 1.0,
        temperature_increase: float = 0.0,
    ) -> JointTelemetry:
        return JointTelemetry(
            joint_index=int(row["joint_index"]),
            position=(
                float(row["position"])
                + position_offset
            ),
            velocity=float(row["velocity"]),
            force=(
                float(row["force"])
                * force_multiplier
            ),
            motor_current=(
                float(row["motor_current"])
                * current_multiplier
            ),
            temperature=(
                float(row["temperature"])
                + temperature_increase
            ),
        )

    @staticmethod
    def _evaluate_healthy(
        detector: BaselineAnomalyDetector,
        rows: list[dict[str, str]],
    ) -> dict[str, int]:
        false_positives = 0
        true_negatives = 0

        for row in rows:
            telemetry = BaselineEvaluator._create_telemetry(
                row
            )

            result = detector.detect(
                telemetry=telemetry,
                target_position=float(
                    row["target_position"]
                ),
            )

            if result.anomaly:
                false_positives += 1
            else:
                true_negatives += 1

        return {
            "total": len(rows),
            "false_positives": false_positives,
            "true_negatives": true_negatives,
        }

    @staticmethod
    def _evaluate_faults(
        detector: BaselineAnomalyDetector,
        rows: list[dict[str, str]],
    ) -> dict[str, int]:
        true_positives = 0
        false_negatives = 0

        for row in rows:
            fault_detected = False

            fault_variants = (
                BaselineEvaluator._create_telemetry(
                    row,
                    position_offset=POSITION_DRIFT,
                ),
                BaselineEvaluator._create_telemetry(
                    row,
                    force_multiplier=FORCE_MULTIPLIER,
                ),
                BaselineEvaluator._create_telemetry(
                    row,
                    current_multiplier=CURRENT_MULTIPLIER,
                ),
                BaselineEvaluator._create_telemetry(
                    row,
                    temperature_increase=TEMPERATURE_INCREASE,
                ),
            )

            target_position = float(
                row["target_position"]
            )

            for telemetry in fault_variants:
                result = detector.detect(
                    telemetry=telemetry,
                    target_position=target_position,
                )

                if result.anomaly:
                    fault_detected = True
                    break

            if fault_detected:
                true_positives += 1
            else:
                false_negatives += 1

        return {
            "total": len(rows),
            "true_positives": true_positives,
            "false_negatives": false_negatives,
        }

    @staticmethod
    def _print_results(
        healthy_results: dict[str, int],
        fault_results: dict[str, int],
    ) -> None:
        true_negatives = healthy_results[
            "true_negatives"
        ]

        false_positives = healthy_results[
            "false_positives"
        ]

        true_positives = fault_results[
            "true_positives"
        ]

        false_negatives = fault_results[
            "false_negatives"
        ]

        total_healthy = healthy_results["total"]
        total_faults = fault_results["total"]

        precision_denominator = (
            true_positives + false_positives
        )

        recall_denominator = (
            true_positives + false_negatives
        )

        precision = (
            true_positives
            / precision_denominator
            if precision_denominator
            else 0.0
        )

        recall = (
            true_positives
            / recall_denominator
            if recall_denominator
            else 0.0
        )

        f1_denominator = precision + recall

        f1_score = (
            2 * precision * recall
            / f1_denominator
            if f1_denominator
            else 0.0
        )

        false_positive_rate = (
            false_positives / total_healthy
            if total_healthy
            else 0.0
        )

        detection_rate = (
            true_positives / total_faults
            if total_faults
            else 0.0
        )

        print()
        print(
            "ROBOTWIN-AI BASELINE EVALUATION"
        )
        print("=" * 55)

        print()
        print("DATA SPLIT")
        print("-" * 55)
        print(
            f"Healthy test samples : {total_healthy}"
        )
        print(
            f"Fault test samples   : {total_faults}"
        )

        print()
        print("CONFUSION MATRIX")
        print("-" * 55)
        print(
            f"True Positives       : {true_positives}"
        )
        print(
            f"True Negatives       : {true_negatives}"
        )
        print(
            f"False Positives      : {false_positives}"
        )
        print(
            f"False Negatives      : {false_negatives}"
        )

        print()
        print("METRICS")
        print("-" * 55)
        print(
            f"Precision            : "
            f"{precision:.4f}"
        )
        print(
            f"Recall               : "
            f"{recall:.4f}"
        )
        print(
            f"F1 Score             : "
            f"{f1_score:.4f}"
        )
        print(
            f"False Positive Rate  : "
            f"{false_positive_rate:.4f}"
        )
        print(
            f"Fault Detection Rate : "
            f"{detection_rate:.4f}"
        )


def main() -> None:
    """Run baseline detector evaluation."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    evaluator = BaselineEvaluator()

    evaluator.evaluate(
        input_file=INPUT_FILE,
    )


if __name__ == "__main__":
    main()