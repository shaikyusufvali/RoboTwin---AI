"""Train and evaluate the RoboTwin-AI fault classifier."""

from __future__ import annotations

import csv
import logging
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger(__name__)

DATASET_PATH = Path("data") / "fault_features.csv"
MODEL_PATH = Path("data") / "fault_classifier_v2.pkl"

FEATURE_COLUMNS = [
    "position",
    "position_error",
    "velocity",
    "acceleration",
    "force",
    "motor_current",
    "temperature",
    "abs_position_error",
    "abs_velocity",
    "abs_acceleration",
    "force_per_current",
    "current_per_force",
    "temperature_current_ratio",
]

TARGET_COLUMN = "fault_type"
EPISODE_COLUMN = "episode_id"

TEST_EPISODES = 8
RANDOM_SEED = 42


def load_dataset(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load engineered features, labels, and episode identifiers."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    features: list[list[float]] = []
    labels: list[str] = []
    episode_ids: list[int] = []

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = set(
            FEATURE_COLUMNS
            + [
                TARGET_COLUMN,
                EPISODE_COLUMN,
            ]
        )

        available_columns = set(
            reader.fieldnames or []
        )

        missing_columns = (
            required_columns - available_columns
        )

        if missing_columns:
            raise ValueError(
                "Missing dataset columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:
            features.append(
                [
                    float(row[column])
                    for column in FEATURE_COLUMNS
                ]
            )

            labels.append(
                row[TARGET_COLUMN]
            )

            episode_ids.append(
                int(row[EPISODE_COLUMN])
            )

    return (
        np.asarray(
            features,
            dtype=np.float64,
        ),
        np.asarray(labels),
        np.asarray(
            episode_ids,
            dtype=np.int64,
        ),
    )


def split_by_episode(
    features: np.ndarray,
    labels: np.ndarray,
    episode_ids: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Split complete episodes into training and test sets."""

    unique_episodes = np.unique(
        episode_ids
    )

    if len(unique_episodes) <= TEST_EPISODES:
        raise ValueError(
            "Not enough episodes for the test split."
        )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    test_episodes = np.sort(
        rng.choice(
            unique_episodes,
            size=TEST_EPISODES,
            replace=False,
        )
    )

    train_episodes = np.setdiff1d(
        unique_episodes,
        test_episodes,
    )

    train_mask = np.isin(
        episode_ids,
        train_episodes,
    )

    test_mask = np.isin(
        episode_ids,
        test_episodes,
    )

    LOGGER.info(
        "Training episodes: %s",
        train_episodes.tolist(),
    )

    LOGGER.info(
        "Testing episodes: %s",
        test_episodes.tolist(),
    )

    return (
        features[train_mask],
        features[test_mask],
        labels[train_mask],
        labels[test_mask],
    )


def train_model(
    train_features: np.ndarray,
    train_labels: np.ndarray,
) -> tuple[
    StandardScaler,
    LogisticRegression,
]:
    """Scale features and train logistic regression."""

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        train_features
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
    )

    model.fit(
        scaled_features,
        train_labels,
    )

    return scaler, model


def evaluate_model(
    scaler: StandardScaler,
    model: LogisticRegression,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> None:
    """Evaluate the classifier on unseen episodes."""

    scaled_test_features = (
        scaler.transform(test_features)
    )

    predictions = model.predict(
        scaled_test_features
    )

    accuracy = accuracy_score(
        test_labels,
        predictions,
    )

    class_names = list(
        model.classes_
    )

    matrix = confusion_matrix(
        test_labels,
        predictions,
        labels=class_names,
    )

    print()
    print("=" * 72)
    print("ROBOTWIN-AI FAULT CLASSIFIER V2")
    print("=" * 72)

    print(
        f"Test records: {len(test_labels)}"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print()
    print("Test class distribution:")

    print(
        Counter(test_labels)
    )

    print()
    print("Classes:")

    print(class_names)

    print()
    print("Confusion Matrix")
    print(
        "Rows = Actual | Columns = Predicted"
    )

    print(matrix)

    print()
    print("Classification Report")

    print(
        classification_report(
            test_labels,
            predictions,
            labels=class_names,
            zero_division=0,
        )
    )


def save_model(
    path: Path,
    scaler: StandardScaler,
    model: LogisticRegression,
) -> None:
    """Save the trained model and preprocessing configuration."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_package = {
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "scaler": scaler,
        "model": model,
    }

    with path.open(
        "wb"
    ) as file:
        pickle.dump(
            model_package,
            file,
        )

    LOGGER.info(
        "Model saved: %s",
        path,
    )


def main() -> None:
    """Run the complete training pipeline."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    features, labels, episode_ids = (
        load_dataset(
            DATASET_PATH
        )
    )

    LOGGER.info(
        "Loaded %d engineered records.",
        len(labels),
    )

    LOGGER.info(
        "Using %d features.",
        len(FEATURE_COLUMNS),
    )

    (
        train_features,
        test_features,
        train_labels,
        test_labels,
    ) = split_by_episode(
        features,
        labels,
        episode_ids,
    )

    LOGGER.info(
        "Training records: %d",
        len(train_labels),
    )

    LOGGER.info(
        "Testing records: %d",
        len(test_labels),
    )

    scaler, model = train_model(
        train_features,
        train_labels,
    )

    evaluate_model(
        scaler,
        model,
        test_features,
        test_labels,
    )

    save_model(
        MODEL_PATH,
        scaler,
        model,
    )


if __name__ == "__main__":
    main()