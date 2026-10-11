"""
RoboTwin-AI
Randomized Robustness Error Analysis

Reads the existing randomized robustness experiment results and
analyzes diagnostic failures without rerunning PyBullet simulations.

Purpose
-------
Identify:

1. Overall diagnostic accuracy
2. Per-fault accuracy
3. Confusion matrix
4. Severity-wise behavior
5. Genuine misclassifications
6. MOTOR_RESISTANCE -> POSITION_DRIFT failures
7. Vision-condition impact
8. Confidence of correct vs incorrect predictions

Input
-----
data/experiments/robustness/randomized_robustness_results.json

Output
------
data/experiments/robustness/robustness_error_analysis.json
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


INPUT_FILE = (
    Path("data")
    / "experiments"
    / "robustness"
    / "randomized_robustness_results.json"
)

OUTPUT_FILE = (
    Path("data")
    / "experiments"
    / "robustness"
    / "robustness_error_analysis.json"
)


FAULT_LABELS = (
    "NORMAL",
    "MOTOR_RESISTANCE",
    "LOW_MOTOR_POWER",
    "POSITION_DRIFT",
    "UNKNOWN",
)


def safe_float(value: Any) -> float:
    """Convert a value to finite float."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0

    if result != result:
        return 0.0

    if result in (
        float("inf"),
        float("-inf"),
    ):
        return 0.0

    return result


def normalize_fault_label(
    value: Any,
) -> str:
    """
    Normalize fault labels.

    Example:

        motor_resistance
        MOTOR_RESISTANCE

    become:

        MOTOR_RESISTANCE
    """

    if value is None:
        return "UNKNOWN"

    label = str(
        value
    ).strip().upper()

    aliases = {
        "NORMAL": "NORMAL",
        "MOTOR_RESISTANCE": "MOTOR_RESISTANCE",
        "LOW_MOTOR_POWER": "LOW_MOTOR_POWER",
        "POSITION_DRIFT": "POSITION_DRIFT",
        "UNKNOWN": "UNKNOWN",
    }

    return aliases.get(
        label,
        label,
    )


def load_results() -> dict[str, Any]:
    """Load existing robustness results."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Robustness result file was not found:\n"
            f"{INPUT_FILE}\n\n"
            "Run randomized_robustness first."
        )

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as source:

        payload = json.load(
            source
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "Robustness result file must contain a JSON object."
        )

    return payload


def extract_trials(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract trial records."""

    trials = payload.get(
        "trials",
        [],
    )

    if not isinstance(
        trials,
        list,
    ):
        raise ValueError(
            "'trials' must be a list."
        )

    return [
        trial
        for trial in trials
        if isinstance(
            trial,
            dict,
        )
    ]


def analyze_trials(
    trials: list[dict[str, Any]],
) -> dict[str, Any]:
    """Perform complete error analysis."""

    successful = [
        trial
        for trial in trials
        if str(
            trial.get(
                "execution_status",
                "",
            )
        ).upper()
        == "SUCCESS"
    ]

    # ---------------------------------------------------------------
    # Overall metrics
    # ---------------------------------------------------------------

    total = len(trials)

    successful_count = len(
        successful
    )

    telemetry_correct = sum(
        bool(
            trial.get(
                "telemetry_correct",
                False,
            )
        )
        for trial in successful
    )

    multimodal_correct = sum(
        bool(
            trial.get(
                "multimodal_correct",
                False,
            )
        )
        for trial in successful
    )

    telemetry_accuracy = (
        telemetry_correct
        / successful_count
        if successful_count
        else 0.0
    )

    multimodal_accuracy = (
        multimodal_correct
        / successful_count
        if successful_count
        else 0.0
    )

    # ---------------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------------

    confusion = {
        actual: {
            predicted: 0
            for predicted in FAULT_LABELS
        }
        for actual in FAULT_LABELS
    }

    for trial in successful:

        actual = normalize_fault_label(
            trial.get(
                "ground_truth_fault"
            )
        )

        predicted = normalize_fault_label(
            trial.get(
                "multimodal_fault"
            )
        )

        if actual not in confusion:
            confusion[actual] = {}

        confusion[actual][predicted] = (
            confusion[actual].get(
                predicted,
                0,
            )
            + 1
        )

    # ---------------------------------------------------------------
    # Genuine failures
    # ---------------------------------------------------------------

    failures = []

    for trial in successful:

        actual = normalize_fault_label(
            trial.get(
                "ground_truth_fault"
            )
        )

        predicted = normalize_fault_label(
            trial.get(
                "multimodal_fault"
            )
        )

        if actual != predicted:

            failures.append(
                {
                    "trial_id": trial.get(
                        "trial_id"
                    ),

                    "repetition": trial.get(
                        "repetition"
                    ),

                    "scenario_id": trial.get(
                        "scenario_id"
                    ),

                    "scenario_name": trial.get(
                        "scenario_name"
                    ),

                    "severity": safe_float(
                        trial.get(
                            "fault_severity"
                        )
                    ),

                    "seed": trial.get(
                        "seed"
                    ),

                    "actual_fault": actual,

                    "predicted_fault": predicted,

                    "confidence": safe_float(
                        trial.get(
                            "multimodal_confidence"
                        )
                    ),

                    "telemetry_fault": normalize_fault_label(
                        trial.get(
                            "telemetry_fault"
                        )
                    ),

                    "vision_quality": trial.get(
                        "requested_vision_quality"
                    ),

                    "obstacle_present": bool(
                        trial.get(
                            "ground_truth_obstacle",
                            False,
                        )
                    ),
                }
            )

    # ---------------------------------------------------------------
    # Per-fault statistics
    # ---------------------------------------------------------------

    per_fault = {}

    for fault in FAULT_LABELS:

        fault_trials = [
            trial
            for trial in successful
            if normalize_fault_label(
                trial.get(
                    "ground_truth_fault"
                )
            )
            == fault
        ]

        if not fault_trials:
            continue

        correct = sum(
            1
            for trial in fault_trials
            if normalize_fault_label(
                trial.get(
                    "multimodal_fault"
                )
            )
            == fault
        )

        incorrect = (
            len(fault_trials)
            - correct
        )

        per_fault[fault] = {
            "total": len(
                fault_trials
            ),

            "correct": correct,

            "incorrect": incorrect,

            "accuracy": round(
                correct
                / len(fault_trials),
                4,
            ),

            "mean_severity": round(
                mean(
                    safe_float(
                        trial.get(
                            "fault_severity"
                        )
                    )
                    for trial in fault_trials
                ),
                4,
            ),

            "mean_confidence": round(
                mean(
                    safe_float(
                        trial.get(
                            "multimodal_confidence"
                        )
                    )
                    for trial in fault_trials
                ),
                4,
            ),
        }

    # ---------------------------------------------------------------
    # Error pattern analysis
    # ---------------------------------------------------------------

    error_pairs = Counter()

    for failure in failures:

        pair = (
            failure["actual_fault"],
            failure["predicted_fault"],
        )

        error_pairs[pair] += 1

    error_patterns = []

    for (
        actual,
        predicted,
    ), count in error_pairs.items():

        related = [
            failure
            for failure in failures
            if (
                failure["actual_fault"]
                == actual
                and
                failure["predicted_fault"]
                == predicted
            )
        ]

        severities = [
            failure["severity"]
            for failure in related
        ]

        confidences = [
            failure["confidence"]
            for failure in related
        ]

        error_patterns.append(
            {
                "actual_fault": actual,

                "predicted_fault": predicted,

                "count": count,

                "mean_severity": round(
                    mean(
                        severities
                    ),
                    4,
                ),

                "minimum_severity": round(
                    min(
                        severities
                    ),
                    4,
                ),

                "maximum_severity": round(
                    max(
                        severities
                    ),
                    4,
                ),

                "mean_confidence": round(
                    mean(
                        confidences
                    ),
                    4,
                ),
            }
        )

    error_patterns.sort(
        key=lambda item: item["count"],
        reverse=True,
    )

    # ---------------------------------------------------------------
    # Severity analysis
    # ---------------------------------------------------------------

    severity_buckets = {
        "0.30-0.49": [],
        "0.50-0.69": [],
        "0.70-0.90": [],
    }

    for trial in successful:

        severity = safe_float(
            trial.get(
                "fault_severity"
            )
        )

        if severity < 0.50:
            bucket = "0.30-0.49"

        elif severity < 0.70:
            bucket = "0.50-0.69"

        else:
            bucket = "0.70-0.90"

        severity_buckets[
            bucket
        ].append(
            trial
        )

    severity_analysis = {}

    for bucket, bucket_trials in (
        severity_buckets.items()
    ):

        if not bucket_trials:
            continue

        correct = sum(
            1
            for trial in bucket_trials
            if normalize_fault_label(
                trial.get(
                    "ground_truth_fault"
                )
            )
            ==
            normalize_fault_label(
                trial.get(
                    "multimodal_fault"
                )
            )
        )

        severity_analysis[
            bucket
        ] = {
            "total": len(
                bucket_trials
            ),

            "correct": correct,

            "incorrect": (
                len(bucket_trials)
                - correct
            ),

            "accuracy": round(
                correct
                / len(bucket_trials),
                4,
            ),

            "mean_confidence": round(
                mean(
                    safe_float(
                        trial.get(
                            "multimodal_confidence"
                        )
                    )
                    for trial in bucket_trials
                ),
                4,
            ),
        }

    # ---------------------------------------------------------------
    # Vision-condition analysis
    # ---------------------------------------------------------------

    vision_conditions = defaultdict(
        list
    )

    for trial in successful:

        condition = str(
            trial.get(
                "requested_vision_quality",
                "unknown",
            )
        )

        vision_conditions[
            condition
        ].append(
            trial
        )

    vision_analysis = {}

    for condition, condition_trials in (
        vision_conditions.items()
    ):

        correct = sum(
            1
            for trial in condition_trials
            if normalize_fault_label(
                trial.get(
                    "ground_truth_fault"
                )
            )
            ==
            normalize_fault_label(
                trial.get(
                    "multimodal_fault"
                )
            )
        )

        vision_analysis[
            condition
        ] = {
            "total": len(
                condition_trials
            ),

            "correct": correct,

            "incorrect": (
                len(condition_trials)
                - correct
            ),

            "accuracy": round(
                correct
                / len(condition_trials),
                4,
            ),
        }

    # ---------------------------------------------------------------
    # Special motor-resistance analysis
    # ---------------------------------------------------------------

    motor_resistance_trials = [
        trial
        for trial in successful
        if normalize_fault_label(
            trial.get(
                "ground_truth_fault"
            )
        )
        == "MOTOR_RESISTANCE"
    ]

    motor_resistance_failures = [
        failure
        for failure in failures
        if failure["actual_fault"]
        == "MOTOR_RESISTANCE"
    ]

    motor_resistance_analysis = {
        "total_trials": len(
            motor_resistance_trials
        ),

        "correct": (
            len(
                motor_resistance_trials
            )
            - len(
                motor_resistance_failures
            )
        ),

        "incorrect": len(
            motor_resistance_failures
        ),

        "accuracy": round(
            (
                len(
                    motor_resistance_trials
                )
                - len(
                    motor_resistance_failures
                )
            )
            / len(
                motor_resistance_trials
            )
            if motor_resistance_trials
            else 0.0,
            4,
        ),

        "failure_severities": [
            failure["severity"]
            for failure
            in motor_resistance_failures
        ],

        "failure_predictions": dict(
            Counter(
                failure[
                    "predicted_fault"
                ]
                for failure
                in motor_resistance_failures
            )
        ),

        "obstacle_failure_count": sum(
            1
            for failure
            in motor_resistance_failures
            if failure[
                "obstacle_present"
            ]
        ),

        "poor_vision_failure_count": sum(
            1
            for failure
            in motor_resistance_failures
            if "poor"
            in str(
                failure[
                    "vision_quality"
                ]
            ).lower()
            or "reduced"
            in str(
                failure[
                    "vision_quality"
                ]
            ).lower()
        ),
    }

    # ---------------------------------------------------------------
    # Correct vs incorrect confidence
    # ---------------------------------------------------------------

    correct_confidences = [
        safe_float(
            trial.get(
                "multimodal_confidence"
            )
        )
        for trial in successful
        if normalize_fault_label(
            trial.get(
                "ground_truth_fault"
            )
        )
        ==
        normalize_fault_label(
            trial.get(
                "multimodal_fault"
            )
        )
    ]

    incorrect_confidences = [
        safe_float(
            trial.get(
                "multimodal_confidence"
            )
        )
        for trial in successful
        if normalize_fault_label(
            trial.get(
                "ground_truth_fault"
            )
        )
        !=
        normalize_fault_label(
            trial.get(
                "multimodal_fault"
            )
        )
    ]

    confidence_analysis = {
        "correct_mean": round(
            mean(
                correct_confidences
            )
            if correct_confidences
            else 0.0,
            4,
        ),

        "incorrect_mean": round(
            mean(
                incorrect_confidences
            )
            if incorrect_confidences
            else 0.0,
            4,
        ),

        "incorrect_confidences": (
            incorrect_confidences
        ),
    }

    # ---------------------------------------------------------------
    # Final interpretation
    # ---------------------------------------------------------------

    if failures:

        interpretation = (
            "The randomized robustness study "
            "completed successfully, but the "
            "diagnostic system shows genuine "
            "classification errors. The errors "
            "must be investigated before claiming "
            "robust fault diagnosis."
        )

    else:

        interpretation = (
            "No diagnostic misclassification was "
            "observed in this pilot study. This "
            "does not constitute a final robustness "
            "benchmark."
        )

    return {
        "analysis_metadata": {
            "analysis_type": (
                "randomized_robustness_error_analysis"
            ),

            "input_file": str(
                INPUT_FILE
            ),

            "output_file": str(
                OUTPUT_FILE
            ),

            "total_trials": total,

            "successful_trials": (
                successful_count
            ),
        },

        "overall": {
            "telemetry_accuracy": round(
                telemetry_accuracy,
                4,
            ),

            "multimodal_accuracy": round(
                multimodal_accuracy,
                4,
            ),

            "genuine_error_count": len(
                failures
            ),
        },

        "confusion_matrix": confusion,

        "per_fault": per_fault,

        "error_patterns": error_patterns,

        "genuine_failures": failures,

        "severity_analysis": (
            severity_analysis
        ),

        "vision_condition_analysis": (
            vision_analysis
        ),

        "motor_resistance_analysis": (
            motor_resistance_analysis
        ),

        "confidence_analysis": (
            confidence_analysis
        ),

        "interpretation": interpretation,
    }


def save_analysis(
    analysis: dict[str, Any],
) -> None:
    """Save analysis results."""

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as destination:

        json.dump(
            analysis,
            destination,
            indent=2,
            ensure_ascii=False,
        )


def print_report(
    analysis: dict[str, Any],
) -> None:
    """Print human-readable error analysis."""

    overall = analysis[
        "overall"
    ]

    print()

    print(
        "ROBOTWIN-AI ROBUSTNESS ERROR ANALYSIS"
    )

    print(
        "=" * 90
    )

    print(
        f"Total trials          : "
        f"{analysis['analysis_metadata']['total_trials']}"
    )

    print(
        f"Successful trials     : "
        f"{analysis['analysis_metadata']['successful_trials']}"
    )

    print(
        f"Telemetry accuracy    : "
        f"{overall['telemetry_accuracy']:.4f}"
    )

    print(
        f"Multimodal accuracy   : "
        f"{overall['multimodal_accuracy']:.4f}"
    )

    print(
        f"Genuine errors        : "
        f"{overall['genuine_error_count']}"
    )

    print(
        "=" * 90
    )

    print()

    print(
        "PER-FAULT ANALYSIS"
    )

    print(
        "-" * 90
    )

    for fault, values in (
        analysis[
            "per_fault"
        ].items()
    ):

        print(
            f"{fault:<22}"
            f" total={values['total']:<3} "
            f"correct={values['correct']:<3} "
            f"incorrect={values['incorrect']:<3} "
            f"accuracy={values['accuracy']:.4f}"
        )

    print()

    print(
        "ERROR PATTERNS"
    )

    print(
        "-" * 90
    )

    patterns = analysis[
        "error_patterns"
    ]

    if not patterns:

        print(
            "No genuine misclassification found."
        )

    else:

        for pattern in patterns:

            print(
                f"{pattern['actual_fault']}"
                f" -> "
                f"{pattern['predicted_fault']} | "
                f"count={pattern['count']} | "
                f"severity="
                f"{pattern['minimum_severity']:.4f}"
                f"-"
                f"{pattern['maximum_severity']:.4f}"
            )

    print()

    print(
        "MOTOR RESISTANCE ANALYSIS"
    )

    print(
        "-" * 90
    )

    motor = analysis[
        "motor_resistance_analysis"
    ]

    print(
        f"Total trials          : "
        f"{motor['total_trials']}"
    )

    print(
        f"Correct               : "
        f"{motor['correct']}"
    )

    print(
        f"Incorrect             : "
        f"{motor['incorrect']}"
    )

    print(
        f"Accuracy              : "
        f"{motor['accuracy']:.4f}"
    )

    print(
        f"Failure severities    : "
        f"{motor['failure_severities']}"
    )

    print(
        f"Failure predictions   : "
        f"{motor['failure_predictions']}"
    )

    print(
        f"Obstacle failures     : "
        f"{motor['obstacle_failure_count']}"
    )

    print(
        f"Poor-vision failures  : "
        f"{motor['poor_vision_failure_count']}"
    )

    print()

    print(
        "SEVERITY ANALYSIS"
    )

    print(
        "-" * 90
    )

    for bucket, values in (
        analysis[
            "severity_analysis"
        ].items()
    ):

        print(
            f"{bucket:<12} "
            f"total={values['total']:<3} "
            f"correct={values['correct']:<3} "
            f"incorrect={values['incorrect']:<3} "
            f"accuracy={values['accuracy']:.4f}"
        )

    print()

    print(
        "VISION CONDITION ANALYSIS"
    )

    print(
        "-" * 90
    )

    for condition, values in (
        analysis[
            "vision_condition_analysis"
        ].items()
    ):

        print(
            f"{condition:<20} "
            f"total={values['total']:<3} "
            f"correct={values['correct']:<3} "
            f"incorrect={values['incorrect']:<3} "
            f"accuracy={values['accuracy']:.4f}"
        )

    print()

    print(
        "CONFIDENCE ANALYSIS"
    )

    print(
        "-" * 90
    )

    confidence = analysis[
        "confidence_analysis"
    ]

    print(
        f"Correct mean confidence   : "
        f"{confidence['correct_mean']:.4f}"
    )

    print(
        f"Incorrect mean confidence : "
        f"{confidence['incorrect_mean']:.4f}"
    )

    print()

    print(
        "INTERPRETATION"
    )

    print(
        "-" * 90
    )

    print(
        analysis[
            "interpretation"
        ]
    )

    print()

    print(
        f"Analysis saved to: "
        f"{OUTPUT_FILE}"
    )

    print(
        "=" * 90
    )


def main() -> None:
    """Run robustness error analysis."""

    payload = load_results()

    trials = extract_trials(
        payload
    )

    if not trials:
        raise RuntimeError(
            "No trial records were found "
            "in the robustness results."
        )

    analysis = analyze_trials(
        trials
    )

    save_analysis(
        analysis
    )

    print_report(
        analysis
    )


if __name__ == "__main__":
    main()