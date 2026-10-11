"""NumPy-based baseline classifier for RoboTwin-AI fault diagnosis."""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path


LOGGER = logging.getLogger(__name__)

DATASET_FILE = Path("data") / "robot_diagnostics_dataset_v2.csv"

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

CLASS_NAMES = (
    "normal",
    "motor_resistance",
    "low_motor_power",
    "position_drift",
)


class BaselineClassifier:
    """Classify robot faults using standardized distance."""

    def __init__(
        self,
        class_means: dict[str, list[float]],
        feature_means: list[float],
        feature_stds: list[float],
    ) -> None:
        self.class_means = class_means
        self.feature_means = feature_means
        self.feature_stds = feature_stds

    def predict(
        self,
        values: list[float],
    ) -> tuple[str, float]:
        """Return predicted class and confidence."""
        standardized = self._standardize(values)

        distances = {
            class_name: self._distance(
                standardized,
                class_mean,
            )
            for class_name, class_mean in self.class_means.items()
        }

        ordered = sorted(
            distances.items(),
            key=lambda item: item[1],
        )

        predicted_class, best_distance = ordered[0]
        second_distance = ordered[1][1]

        confidence = self._calculate_confidence(
            best_distance,
            second_distance,
        )

        return predicted_class, confidence

    def _standardize(
        self,
        values: list[float],
    ) -> list[float]:
        """Standardize a feature vector."""
        return [
            (value - mean) / std
            for value, mean, std in zip(
                values,
                self.feature_means,
                self.feature_stds,
            )
        ]

    @staticmethod
    def _distance(
        first: list[float],
        second: list[float],
    ) -> float:
        """Calculate Euclidean distance."""
        return math.sqrt(
            sum(
                (a - b) ** 2
                for a, b in zip(first, second)
            )
        )

    @staticmethod
    def _calculate_confidence(
        best_distance: float,
        second_distance: float,
    ) -> float:
        """Estimate confidence from distance separation."""
        if second_distance <= 0:
            return 0.0

        confidence = 1.0 - (
            best_distance / second_distance
        )

        return max(0.0, min(1.0, confidence))


def load_dataset(
    dataset_file: Path,
) -> tuple[list[list[float]], list[str]]:
    """Load features and labels from the diagnostic dataset."""
    if not dataset_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_file}"
        )

    features: list[list[float]] = []
    labels: list[str] = []

    with dataset_file.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            features.append(
                [
                    float(row[column])
                    for column in FEATURES
                ]
            )
            labels.append(row[TARGET])

    return features, labels


def calculate_statistics(
    features: list[list[float]],
) -> tuple[list[float], list[float]]:
    """Calculate mean and standard deviation for each feature."""
    feature_count = len(FEATURES)

    means: list[float] = []
    stds: list[float] = []

    for index in range(feature_count):
        values = [
            row[index]
            for row in features
        ]

        mean = sum(values) / len(values)

        variance = sum(
            (value - mean) ** 2
            for value in values
        ) / len(values)

        std = math.sqrt(variance)

        means.append(mean)
        stds.append(std if std > 0 else 1.0)

    return means, stds


def calculate_class_means(
    features: list[list[float]],
    labels: list[str],
    feature_means: list[float],
    feature_stds: list[float],
) -> dict[str, list[float]]:
    """Calculate standardized mean vector for each fault class."""
    grouped: dict[str, list[list[float]]] = {
        class_name: []
        for class_name in CLASS_NAMES
    }

    for row, label in zip(features, labels):
        grouped[label].append(row)

    class_means: dict[str, list[float]] = {}

    for class_name, rows in grouped.items():
        standardized_rows = [
            [
                (value - mean) / std
                for value, mean, std in zip(
                    row,
                    feature_means,
                    feature_stds,
                )
            ]
            for row in rows
        ]

        class_means[class_name] = [
            sum(row[index] for row in standardized_rows)
            / len(standardized_rows)
            for index in range(len(FEATURES))
        ]

    return class_means


def evaluate(
    classifier: BaselineClassifier,
    features: list[list[float]],
    labels: list[str],
) -> None:
    """Evaluate the baseline classifier."""
    correct = 0
    confusion: dict[str, dict[str, int]] = {
        actual: {
            predicted: 0
            for predicted in CLASS_NAMES
        }
        for actual in CLASS_NAMES
    }

    for values, actual in zip(features, labels):
        predicted, _ = classifier.predict(values)

        confusion[actual][predicted] += 1

        if predicted == actual:
            correct += 1

    accuracy = correct / len(labels)

    print("\nBaseline Evaluation")
    print("===================")
    print(f"Records: {len(labels)}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nConfusion Matrix")
    print("================")
    print(
        "actual\\predicted | "
        + " | ".join(CLASS_NAMES)
    )

    for actual in CLASS_NAMES:
        values = [
            str(confusion[actual][predicted])
            for predicted in CLASS_NAMES
        ]

        print(
            f"{actual:17} | "
            + " | ".join(values)
        )


def main() -> None:
    """Run the RoboTwin-AI baseline classifier."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    LOGGER.info(
        "Loading diagnostic dataset: %s",
        DATASET_FILE,
    )

    features, labels = load_dataset(DATASET_FILE)

    feature_means, feature_stds = calculate_statistics(
        features
    )

    class_means = calculate_class_means(
        features,
        labels,
        feature_means,
        feature_stds,
    )

    classifier = BaselineClassifier(
        class_means=class_means,
        feature_means=feature_means,
        feature_stds=feature_stds,
    )

    evaluate(
        classifier,
        features,
        labels,
    )


if __name__ == "__main__":
    main()