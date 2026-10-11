"""
RoboTwin-AI Scientific Evaluation Engine

Purpose
-------
Evaluates controlled experiment results produced by scenario_executor.py.

Metrics
-------
- Accuracy
- Per-class precision
- Per-class recall
- Per-class F1
- Macro precision / recall / F1
- Confusion matrix
- False positives / false negatives
- Obstacle detection metrics
- Detection latency statistics when available

Important
---------
The current six-scenario experiment is treated as a pilot evaluation.
It is not considered a statistically sufficient final benchmark.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = PROJECT_ROOT / "data" / "experiments" / "scenario_results.json"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "experiments"
    / "scientific_evaluation.json"
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

LOGGER = logging.getLogger("robotwin.scientific_evaluation")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NORMAL_LABEL = "NORMAL"

FAULT_LABELS = (
    "NORMAL",
    "MOTOR_RESISTANCE",
    "LOW_MOTOR_POWER",
    "POSITION_DRIFT",
)

SUCCESS_STATUS = "SUCCESS"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def safe_float(value: Any, default: float | None = None) -> float | None:
    """Safely convert a value to float."""

    if value is None:
        return default

    try:
        number = float(value)

        if not math.isfinite(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def normalize_label(value: Any) -> str:
    """
    Normalize fault labels into the canonical evaluation format.
    """

    if value is None:
        return "UNKNOWN"

    text = str(value).strip().upper()

    if not text:
        return "UNKNOWN"

    return text


def safe_percentage(value: float) -> float:
    """Convert a ratio to percentage."""

    return round(value * 100.0, 4)


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------------

def calculate_binary_metrics(
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> dict[str, float]:
    """
    Calculate precision, recall and F1 for one class.
    """

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    if precision_denominator == 0:
        precision = 0.0
    else:
        precision = true_positive / precision_denominator

    if recall_denominator == 0:
        recall = 0.0
    else:
        recall = true_positive / recall_denominator

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2.0
            * precision
            * recall
            / (precision + recall)
        )

    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


# ---------------------------------------------------------------------------
# Scientific Evaluation Engine
# ---------------------------------------------------------------------------

class ScientificEvaluationEngine:
    """
    Evaluates RoboTwin-AI controlled experiment results.
    """

    def __init__(
        self,
        input_file: Path = INPUT_FILE,
        output_file: Path = OUTPUT_FILE,
    ) -> None:

        self.input_file = Path(input_file)
        self.output_file = Path(output_file)

    # -----------------------------------------------------------------------
    # Input
    # -----------------------------------------------------------------------

    def load_results(self) -> dict[str, Any]:
        """Load scenario results JSON."""

        if not self.input_file.exists():
            raise FileNotFoundError(
                f"Experiment result file not found: "
                f"{self.input_file}"
            )

        with self.input_file.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(
                "scenario_results.json must contain a JSON object."
            )

        return data

    # -----------------------------------------------------------------------
    # Scenario extraction
    # -----------------------------------------------------------------------

    def extract_scenarios(
        self,
        data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Extract scenario records from known result structures.
        """

        possible_keys = (
            "scenarios",
            "results",
            "experiment_results",
            "scenario_results",
        )

        scenarios: Any = None

        for key in possible_keys:
            candidate = data.get(key)

            if isinstance(candidate, list):
                scenarios = candidate
                break

        if scenarios is None:
            raise ValueError(
                "Could not find scenario records in "
                "scenario_results.json."
            )

        valid_records: list[dict[str, Any]] = []

        for item in scenarios:

            if isinstance(item, dict):
                valid_records.append(item)

        if not valid_records:
            raise ValueError(
                "No valid scenario records found."
            )

        return valid_records

    # -----------------------------------------------------------------------
    # Field extraction
    # -----------------------------------------------------------------------

    @staticmethod
    def get_ground_truth(
        record: dict[str, Any],
    ) -> str:
        """Extract ground-truth fault label."""

        ground_truth = record.get("ground_truth")

        if isinstance(ground_truth, dict):
            for key in (
                "fault_type",
                "fault",
                "label",
                "status",
            ):
                if key in ground_truth:
                    return normalize_label(
                        ground_truth[key]
                    )

        for key in (
            "ground_truth_fault",
            "truth",
            "fault_type",
            "fault",
        ):
            if key in record:
                return normalize_label(
                    record[key]
                )

        return "UNKNOWN"

    @staticmethod
    def get_prediction(
        record: dict[str, Any],
    ) -> str:
        """Extract multimodal diagnostic prediction."""

        multimodal = record.get("multimodal")

        if isinstance(multimodal, dict):
            for key in (
                "fault_type",
                "fault",
                "prediction",
                "status",
            ):
                if key in multimodal:
                    return normalize_label(
                        multimodal[key]
                    )

        for key in (
            "multimodal_fault",
            "multimodal_prediction",
            "prediction",
        ):
            if key in record:
                return normalize_label(
                    record[key]
                )

        return "UNKNOWN"

    @staticmethod
    def get_obstacle_truth(
        record: dict[str, Any],
    ) -> bool | None:
        """Extract obstacle ground truth."""

        ground_truth = record.get("ground_truth")

        if isinstance(ground_truth, dict):
            if "obstacle_present" in ground_truth:
                return bool(
                    ground_truth["obstacle_present"]
                )

            if "obstacle" in ground_truth:
                return bool(
                    ground_truth["obstacle"]
                )

        for key in (
            "obstacle_present",
            "ground_truth_obstacle",
        ):
            if key in record:
                return bool(record[key])

        return None

    @staticmethod
    def get_obstacle_prediction(
        record: dict[str, Any],
    ) -> bool | None:
        """Extract obstacle detection result."""

        vision = record.get("vision")

        if isinstance(vision, dict):
            for key in (
                "obstacle_detected",
                "obstacle",
                "detected",
            ):
                if key in vision:
                    return bool(vision[key])

        multimodal = record.get("multimodal")

        if isinstance(multimodal, dict):
            for key in (
                "obstacle_detected",
                "obstacle",
            ):
                if key in multimodal:
                    return bool(multimodal[key])

        for key in (
            "obstacle_detected",
            "vision_obstacle",
        ):
            if key in record:
                return bool(record[key])

        return None

    @staticmethod
    def get_latency(
        record: dict[str, Any],
    ) -> float | None:
        """Extract diagnostic latency in milliseconds."""

        latency_keys = (
            "diagnostic_latency_ms",
            "detection_latency_ms",
            "latency_ms",
            "multimodal_latency_ms",
        )

        for key in latency_keys:

            value = safe_float(
                record.get(key)
            )

            if value is not None:
                return value

        multimodal = record.get("multimodal")

        if isinstance(multimodal, dict):

            for key in latency_keys:

                value = safe_float(
                    multimodal.get(key)
                )

                if value is not None:
                    return value

        return None

    # -----------------------------------------------------------------------
    # Classification evaluation
    # -----------------------------------------------------------------------

    def evaluate_classification(
        self,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate fault classification."""

        evaluated = []

        for record in records:

            truth = self.get_ground_truth(record)
            prediction = self.get_prediction(record)

            evaluated.append(
                {
                    "truth": truth,
                    "prediction": prediction,
                }
            )

        labels = sorted(
            set(
                FAULT_LABELS
            ).union(
                {
                    item["truth"]
                    for item in evaluated
                }
            ).union(
                {
                    item["prediction"]
                    for item in evaluated
                }
            )
        )

        confusion_matrix = {
            actual: {
                predicted: 0
                for predicted in labels
            }
            for actual in labels
        }

        correct = 0

        for item in evaluated:

            truth = item["truth"]
            prediction = item["prediction"]

            confusion_matrix[truth][prediction] += 1

            if truth == prediction:
                correct += 1

        total = len(evaluated)

        accuracy = (
            correct / total
            if total > 0
            else 0.0
        )

        per_class: dict[str, Any] = {}

        for label in labels:

            true_positive = confusion_matrix[label][label]

            false_positive = sum(
                confusion_matrix[actual][label]
                for actual in labels
                if actual != label
            )

            false_negative = sum(
                confusion_matrix[label][predicted]
                for predicted in labels
                if predicted != label
            )

            metrics = calculate_binary_metrics(
                true_positive=true_positive,
                false_positive=false_positive,
                false_negative=false_negative,
            )

            support = sum(
                confusion_matrix[label].values()
            )

            per_class[label] = {
                "support": support,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                **metrics,
            }

        macro_precision = (
            sum(
                item["precision"]
                for item in per_class.values()
            )
            / len(per_class)
            if per_class
            else 0.0
        )

        macro_recall = (
            sum(
                item["recall"]
                for item in per_class.values()
            )
            / len(per_class)
            if per_class
            else 0.0
        )

        macro_f1 = (
            sum(
                item["f1"]
                for item in per_class.values()
            )
            / len(per_class)
            if per_class
            else 0.0
        )

        false_positive_records = []
        false_negative_records = []

        for record in records:

            truth = self.get_ground_truth(record)
            prediction = self.get_prediction(record)

            if truth != prediction:

                error = {
                    "scenario_id": record.get(
                        "scenario_id",
                        record.get("id", "UNKNOWN"),
                    ),
                    "truth": truth,
                    "prediction": prediction,
                }

                false_negative_records.append(error)

                false_positive_records.append(error)

        return {
            "records_evaluated": total,
            "correct_predictions": correct,
            "incorrect_predictions": total - correct,
            "accuracy": round(
                accuracy,
                6,
            ),
            "macro_precision": round(
                macro_precision,
                6,
            ),
            "macro_recall": round(
                macro_recall,
                6,
            ),
            "macro_f1": round(
                macro_f1,
                6,
            ),
            "labels": labels,
            "confusion_matrix": confusion_matrix,
            "per_class": per_class,
            "misclassifications": false_positive_records,
            "pilot_evaluation": total < 30,
        }

    # -----------------------------------------------------------------------
    # Obstacle evaluation
    # -----------------------------------------------------------------------

    def evaluate_obstacle_detection(
        self,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate obstacle detection."""

        evaluated = []

        for record in records:

            truth = self.get_obstacle_truth(record)
            prediction = self.get_obstacle_prediction(record)

            if truth is None or prediction is None:
                continue

            evaluated.append(
                {
                    "scenario_id": record.get(
                        "scenario_id",
                        record.get("id", "UNKNOWN"),
                    ),
                    "truth": truth,
                    "prediction": prediction,
                }
            )

        if not evaluated:

            return {
                "records_evaluated": 0,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "true_positive": 0,
                "true_negative": 0,
                "false_positive": 0,
                "false_negative": 0,
            }

        true_positive = sum(
            1
            for item in evaluated
            if item["truth"] is True
            and item["prediction"] is True
        )

        true_negative = sum(
            1
            for item in evaluated
            if item["truth"] is False
            and item["prediction"] is False
        )

        false_positive = sum(
            1
            for item in evaluated
            if item["truth"] is False
            and item["prediction"] is True
        )

        false_negative = sum(
            1
            for item in evaluated
            if item["truth"] is True
            and item["prediction"] is False
        )

        total = len(evaluated)

        accuracy = (
            (true_positive + true_negative) / total
            if total
            else 0.0
        )

        metrics = calculate_binary_metrics(
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
        )

        return {
            "records_evaluated": total,
            "accuracy": round(accuracy, 6),
            **metrics,
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }

    # -----------------------------------------------------------------------
    # Latency evaluation
    # -----------------------------------------------------------------------

    def evaluate_latency(
        self,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate available diagnostic latency measurements."""

        values: list[float] = []

        for record in records:

            latency = self.get_latency(record)

            if latency is not None and latency >= 0:
                values.append(latency)

        if not values:

            return {
                "records_with_latency": 0,
                "mean_ms": None,
                "min_ms": None,
                "max_ms": None,
            }

        mean_latency = (
            sum(values) / len(values)
        )

        return {
            "records_with_latency": len(values),
            "mean_ms": round(
                mean_latency,
                4,
            ),
            "min_ms": round(
                min(values),
                4,
            ),
            "max_ms": round(
                max(values),
                4,
            ),
        }

    # -----------------------------------------------------------------------
    # Pilot interpretation
    # -----------------------------------------------------------------------

    @staticmethod
    def build_interpretation(
        classification: dict[str, Any],
        obstacle: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build a factual interpretation of the current evaluation.

        This deliberately avoids presenting a small pilot as a final
        scientific benchmark.
        """

        records = classification["records_evaluated"]

        statements = []

        statements.append(
            f"{records} experiment records were evaluated."
        )

        if records < 30:

            statements.append(
                "The current result is a pilot evaluation "
                "because the experiment count is small."
            )

            statements.append(
                "Repeated runs across seeds and fault severities "
                "are required before making a robust performance claim."
            )

        if classification["incorrect_predictions"] == 0:

            statements.append(
                "No classification errors were observed in "
                "the current pilot records."
            )

        if obstacle["records_evaluated"] > 0:

            statements.append(
                "Obstacle detection was evaluated on scenarios "
                "containing obstacle ground truth."
            )

        return {
            "statements": statements,
            "final_benchmark_ready": False,
        }

    # -----------------------------------------------------------------------
    # Complete evaluation
    # -----------------------------------------------------------------------

    def evaluate(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the complete scientific evaluation."""

        records = self.extract_scenarios(data)

        classification = self.evaluate_classification(
            records
        )

        obstacle = self.evaluate_obstacle_detection(
            records
        )

        latency = self.evaluate_latency(
            records
        )

        interpretation = self.build_interpretation(
            classification,
            obstacle,
        )

        evaluation = {
            "schema_version": "1.0",
            "generated_at": utc_timestamp(),

            "source": {
                "input_file": str(
                    self.input_file
                ),
                "record_count": len(records),
            },

            "evaluation_type": (
                "single_run_pilot"
                if len(records) < 30
                else "controlled_experiment_evaluation"
            ),

            "classification": classification,

            "obstacle_detection": obstacle,

            "latency": latency,

            "interpretation": interpretation,
        }

        return evaluation

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    def save(
        self,
        evaluation: dict[str, Any],
    ) -> None:
        """Save evaluation results."""

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.output_file.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                evaluation,
                file,
                indent=4,
            )

    # -----------------------------------------------------------------------
    # Console report
    # -----------------------------------------------------------------------

    @staticmethod
    def print_report(
        evaluation: dict[str, Any],
    ) -> None:
        """Print a readable scientific evaluation report."""

        classification = evaluation[
            "classification"
        ]

        obstacle = evaluation[
            "obstacle_detection"
        ]

        latency = evaluation[
            "latency"
        ]

        interpretation = evaluation[
            "interpretation"
        ]

        print()
        print("=" * 100)
        print("ROBOTWIN-AI SCIENTIFIC EVALUATION")
        print("=" * 100)

        print()
        print(
            f"Evaluation type : "
            f"{evaluation['evaluation_type']}"
        )

        print(
            f"Records evaluated: "
            f"{classification['records_evaluated']}"
        )

        print()
        print("CLASSIFICATION METRICS")
        print("-" * 100)

        print(
            f"Accuracy         : "
            f"{classification['accuracy']:.4f}"
        )

        print(
            f"Macro Precision  : "
            f"{classification['macro_precision']:.4f}"
        )

        print(
            f"Macro Recall     : "
            f"{classification['macro_recall']:.4f}"
        )

        print(
            f"Macro F1         : "
            f"{classification['macro_f1']:.4f}"
        )

        print()
        print("PER-CLASS METRICS")
        print("-" * 100)

        header = (
            f"{'CLASS':<24}"
            f"{'SUPPORT':>10}"
            f"{'PRECISION':>12}"
            f"{'RECALL':>12}"
            f"{'F1':>12}"
        )

        print(header)
        print("-" * 100)

        for label, metrics in (
            classification["per_class"].items()
        ):

            print(
                f"{label:<24}"
                f"{metrics['support']:>10}"
                f"{metrics['precision']:>12.4f}"
                f"{metrics['recall']:>12.4f}"
                f"{metrics['f1']:>12.4f}"
            )

        print()
        print("CONFUSION MATRIX")
        print("-" * 100)

        labels = classification["labels"]
        matrix = classification["confusion_matrix"]

        print(
            f"{'ACTUAL / PREDICTED':<24}",
            end="",
        )

        for label in labels:
            print(
                f"{label:>20}",
                end="",
            )

        print()

        for actual in labels:

            print(
                f"{actual:<24}",
                end="",
            )

            for predicted in labels:

                value = matrix[actual][predicted]

                print(
                    f"{value:>20}",
                    end="",
                )

            print()

        print()
        print("CLASSIFICATION ERRORS")
        print("-" * 100)

        errors = classification[
            "misclassifications"
        ]

        if not errors:

            print(
                "No classification errors "
                "observed in the current pilot."
            )

        else:

            for error in errors:

                print(
                    f"{error['scenario_id']} | "
                    f"truth={error['truth']} | "
                    f"prediction={error['prediction']}"
                )

        print()
        print("OBSTACLE DETECTION")
        print("-" * 100)

        if obstacle["records_evaluated"] == 0:

            print(
                "No usable obstacle ground-truth "
                "records were found."
            )

        else:

            print(
                f"Records evaluated: "
                f"{obstacle['records_evaluated']}"
            )

            print(
                f"Accuracy         : "
                f"{obstacle['accuracy']:.4f}"
            )

            print(
                f"Precision        : "
                f"{obstacle['precision']:.4f}"
            )

            print(
                f"Recall           : "
                f"{obstacle['recall']:.4f}"
            )

            print(
                f"F1               : "
                f"{obstacle['f1']:.4f}"
            )

            print(
                f"True Positive    : "
                f"{obstacle['true_positive']}"
            )

            print(
                f"True Negative    : "
                f"{obstacle['true_negative']}"
            )

            print(
                f"False Positive   : "
                f"{obstacle['false_positive']}"
            )

            print(
                f"False Negative   : "
                f"{obstacle['false_negative']}"
            )

        print()
        print("LATENCY")
        print("-" * 100)

        if latency["records_with_latency"] == 0:

            print(
                "No diagnostic latency values "
                "were available in the current results."
            )

        else:

            print(
                f"Records          : "
                f"{latency['records_with_latency']}"
            )

            print(
                f"Mean latency     : "
                f"{latency['mean_ms']:.4f} ms"
            )

            print(
                f"Minimum latency  : "
                f"{latency['min_ms']:.4f} ms"
            )

            print(
                f"Maximum latency  : "
                f"{latency['max_ms']:.4f} ms"
            )

        print()
        print("SCIENTIFIC INTERPRETATION")
        print("-" * 100)

        for statement in interpretation[
            "statements"
        ]:

            print(f"- {statement}")

        print()
        print(
            "IMPORTANT: Current evaluation is a pilot. "
            "Repeated controlled experiments are required "
            "for robust scientific validation."
        )

        print()
        print("=" * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """CLI entry point."""

    print()
    print(
        "Starting RoboTwin-AI Scientific Evaluation..."
    )

    try:

        engine = ScientificEvaluationEngine()

        LOGGER.info(
            "Loading experiment results from %s",
            engine.input_file,
        )

        data = engine.load_results()

        evaluation = engine.evaluate(data)

        engine.save(evaluation)

        engine.print_report(evaluation)

        print()
        print(
            f"Evaluation saved to: "
            f"{engine.output_file}"
        )
        print()

        return 0

    except FileNotFoundError as exc:

        LOGGER.error("%s", exc)

        return 1

    except json.JSONDecodeError as exc:

        LOGGER.error(
            "Invalid JSON in experiment results: %s",
            exc,
        )

        return 1

    except Exception as exc:

        LOGGER.exception(
            "Scientific evaluation failed: %s",
            exc,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())