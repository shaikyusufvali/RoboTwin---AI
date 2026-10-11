
"""
Analyze fault-class feature statistics for RoboTwin-AI.

This module compares telemetry features across fault classes
to identify feature overlap before retraining the classifier.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path


LOGGER = logging.getLogger(__name__)


DATASET_FILE = Path("data") / "robot_diagnostics_dataset.csv"

FEATURES = (
    "joint_index",
    "commanded_position",
    "position",
    "position_error",
    "velocity",
    "acceleration",
    "force",
    "motor_current",
    "temperature",
)

TARGET = "fault_type"


def load_dataset(
    dataset_file: Path,
) -> list[dict[str, float | str]]:
    """Load the diagnostic dataset."""

    if not dataset_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_file}"
        )

    records: list[dict[str, float | str]] = []

    with dataset_file.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "Dataset does not contain a header."
            )

        required_columns = (
            *FEATURES,
            TARGET,
        )

        missing_columns = [
            column
            for column in required_columns
            if column not in reader.fieldnames
        ]

        if missing_columns:
            raise ValueError(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        for row in reader:
            record: dict[str, float | str] = {
                TARGET: row[TARGET],
            }

            for feature in FEATURES:
                record[feature] = float(
                    row[feature]
                )

            records.append(record)

    if not records:
        raise ValueError(
            "Dataset contains no records."
        )

    return records


def group_by_fault(
    records: list[dict[str, float | str]],
) -> dict[str, list[dict[str, float | str]]]:
    """Group telemetry records by fault type."""

    groups: dict[
        str,
        list[dict[str, float | str]],
    ] = defaultdict(list)

    for record in records:
        fault_type = str(
            record[TARGET]
        )

        groups[fault_type].append(
            record
        )

    return dict(groups)


def calculate_mean(
    values: list[float],
) -> float:
    """Calculate the arithmetic mean."""

    if not values:
        return 0.0

    return sum(values) / len(values)


def calculate_minimum(
    values: list[float],
) -> float:
    """Return the minimum value."""

    return min(values)


def calculate_maximum(
    values: list[float],
) -> float:
    """Return the maximum value."""

    return max(values)


def calculate_range(
    values: list[float],
) -> float:
    """Calculate the value range."""

    return (
        calculate_maximum(values)
        - calculate_minimum(values)
    )


def analyze_feature(
    records: list[dict[str, float | str]],
    feature: str,
) -> tuple[float, float, float, float]:
    """
    Calculate mean, minimum, maximum, and range
    for one feature.
    """

    values = [
        float(record[feature])
        for record in records
    ]

    return (
        calculate_mean(values),
        calculate_minimum(values),
        calculate_maximum(values),
        calculate_range(values),
    )


def print_class_distribution(
    groups: dict[str, list[dict[str, float | str]]],
) -> None:
    """Print the number of samples in every fault class."""

    print()
    print("FAULT CLASS DISTRIBUTION")
    print("=" * 60)

    total = sum(
        len(records)
        for records in groups.values()
    )

    for fault_type in sorted(groups):
        count = len(
            groups[fault_type]
        )

        percentage = (
            count / total * 100
            if total
            else 0.0
        )

        print(
            f"{fault_type:<20}"
            f"{count:>8}"
            f"{percentage:>11.2f}%"
        )

    print("-" * 60)
    print(
        f"{'TOTAL':<20}"
        f"{total:>8}"
        f"{100.0:>11.2f}%"
    )


def print_feature_analysis(
    groups: dict[str, list[dict[str, float | str]]],
) -> None:
    """Print feature statistics for every fault class."""

    print()
    print("FEATURE ANALYSIS")
    print("=" * 100)

    for feature in FEATURES:
        print()
        print(f"Feature: {feature}")
        print("-" * 100)

        print(
            f"{'Fault Type':<22}"
            f"{'Mean':>12}"
            f"{'Minimum':>14}"
            f"{'Maximum':>14}"
            f"{'Range':>14}"
        )

        print("-" * 100)

        for fault_type in sorted(groups):
            mean, minimum, maximum, value_range = (
                analyze_feature(
                    groups[fault_type],
                    feature,
                )
            )

            print(
                f"{fault_type:<22}"
                f"{mean:>12.4f}"
                f"{minimum:>14.4f}"
                f"{maximum:>14.4f}"
                f"{value_range:>14.4f}"
            )


def calculate_class_separation(
    groups: dict[str, list[dict[str, float | str]]],
    feature: str,
) -> float:
    """
    Estimate feature separation using class mean spread.

    Higher values indicate that class means are farther apart.
    This is an exploratory statistic, not a model score.
    """

    class_means: list[float] = []

    for records in groups.values():
        values = [
            float(record[feature])
            for record in records
        ]

        class_means.append(
            calculate_mean(values)
        )

    if len(class_means) < 2:
        return 0.0

    return (
        max(class_means)
        - min(class_means)
    )


def print_feature_separation(
    groups: dict[str, list[dict[str, float | str]]],
) -> None:
    """Print exploratory feature-separation scores."""

    separation_scores: list[
        tuple[str, float]
    ] = []

    for feature in FEATURES:
        score = calculate_class_separation(
            groups,
            feature,
        )

        separation_scores.append(
            (feature, score)
        )

    separation_scores.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    print()
    print("FEATURE SEPARATION")
    print("=" * 60)
    print(
        "Higher score means greater spread between "
        "class means."
    )
    print()

    for feature, score in separation_scores:
        print(
            f"{feature:<25}"
            f"{score:>12.4f}"
        )


def main() -> None:
    """Run fault dataset analysis."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    LOGGER.info(
        "Loading dataset: %s",
        DATASET_FILE,
    )

    records = load_dataset(
        DATASET_FILE
    )

    LOGGER.info(
        "Loaded %s records.",
        len(records),
    )

    groups = group_by_fault(
        records
    )

    print()
    print("ROBOTWIN-AI FAULT ANALYSIS")
    print("=" * 60)

    print_class_distribution(
        groups
    )

    print_feature_analysis(
        groups
    )

    print_feature_separation(
        groups
    )


if __name__ == "__main__":
    main()
