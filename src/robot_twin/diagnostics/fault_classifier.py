"""Train and use the RoboTwin-AI fault classifier."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Final

from robot_twin.telemetry.robot_adapter import JointTelemetry


LOGGER = logging.getLogger(__name__)


DATASET_FILE: Final[Path] = (
    Path("data") / "robot_diagnostics_dataset.csv"
)

MODEL_FILE: Final[Path] = (
    Path("data") / "robot_fault_classifier_clean.joblib"
)

FEATURES: Final[tuple[str, ...]] = (
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

TARGET: Final[str] = "fault_type"

TEST_SIZE: Final[float] = 0.20
RANDOM_STATE: Final[int] = 42

UNKNOWN_FAULT: Final[str] = "UNKNOWN"


class FaultClassifier:
    """Train, save, load, and use the robot fault classifier.

    Scikit-learn is an optional component of the diagnostic system.

    When the native scikit-learn runtime is unavailable, the classifier
    enters a cached degraded mode and returns UNKNOWN without repeatedly
    attempting the same native import for every telemetry sample.

    The rest of RoboTwin-AI continues operating through:

        - telemetry
        - healthy baselines
        - anomaly detection
        - fault evidence
        - temporal confirmation
    """

    def __init__(
        self,
        dataset_file: Path = DATASET_FILE,
        model_file: Path = MODEL_FILE,
    ) -> None:
        self.dataset_file = Path(dataset_file)
        self.model_file = Path(model_file)

        self.model: Any | None = None

        # Runtime state.
        #
        # None  = runtime has not been checked.
        # True  = scikit-learn runtime is available.
        # False = scikit-learn runtime is unavailable.
        self._ml_available: bool | None = None

        self._ml_error: str | None = None

        # Prevent repeated model-loading attempts during a monitoring
        # session when the runtime or artifact is unavailable.
        self._model_load_attempted = False

        # Logging state.
        self._runtime_warning_logged = False
        self._degraded_mode_logged = False

    @property
    def ml_available(self) -> bool:
        """Return whether scikit-learn is currently usable."""

        return self._check_ml_runtime()

    @property
    def ml_error(self) -> str | None:
        """Return the most recent ML runtime error."""

        return self._ml_error

    def load_dataset(
        self,
    ) -> tuple[list[list[float]], list[str]]:
        """Load features and target labels from the CSV dataset."""

        if not self.dataset_file.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_file}"
            )

        with self.dataset_file.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            reader = csv.DictReader(csv_file)

            if reader.fieldnames is None:
                raise ValueError(
                    "Dataset does not contain a header."
                )

            required_columns = set(FEATURES) | {TARGET}

            missing_columns = (
                required_columns
                - set(reader.fieldnames)
            )

            if missing_columns:
                raise ValueError(
                    "Dataset is missing required columns: "
                    f"{sorted(missing_columns)}"
                )

            feature_rows: list[list[float]] = []
            labels: list[str] = []

            for row_number, row in enumerate(
                reader,
                start=2,
            ):
                try:
                    feature_values = [
                        float(row[feature])
                        for feature in FEATURES
                    ]
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "Invalid feature value at CSV row "
                        f"{row_number}."
                    ) from exc

                label = row[TARGET]

                if not label:
                    raise ValueError(
                        "Missing target label at CSV row "
                        f"{row_number}."
                    )

                feature_rows.append(feature_values)
                labels.append(label)

        if not feature_rows:
            raise ValueError(
                "Dataset does not contain any records."
            )

        LOGGER.info(
            "Loaded dataset: records=%d | features=%d",
            len(feature_rows),
            len(FEATURES),
        )

        return feature_rows, labels

    def train(self) -> dict[str, object]:
        """Train and evaluate the Random Forest model."""

        self._require_ml_runtime()

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
        )
        from sklearn.model_selection import train_test_split

        features, labels = self.load_dataset()

        x_train, x_test, y_train, y_test = (
            train_test_split(
                features,
                labels,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=labels,
            )
        )

        LOGGER.info(
            "Train records: %d | Test records: %d",
            len(x_train),
            len(x_test),
        )

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )

        model.fit(
            x_train,
            y_train,
        )

        predictions = model.predict(
            x_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        labels_order = list(
            model.classes_
        )

        report = classification_report(
            y_test,
            predictions,
            labels=labels_order,
            zero_division=0,
        )

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=labels_order,
        )

        self.model = model

        self._print_results(
            accuracy=accuracy,
            report=report,
            matrix=matrix,
            labels=labels_order,
        )

        return {
            "accuracy": accuracy,
            "report": report,
            "confusion_matrix": matrix,
            "labels": labels_order,
        }

    def save(self) -> None:
        """Save the trained classifier and metadata."""

        if self.model is None:
            raise RuntimeError(
                "Model has not been trained."
            )

        try:
            import joblib
        except Exception as exc:
            raise RuntimeError(
                "joblib is unavailable. "
                "The trained classifier cannot be saved."
            ) from exc

        self.model_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        artifact = {
            "model": self.model,
            "features": FEATURES,
            "target": TARGET,
            "dataset": str(self.dataset_file),
            "classifier": "RandomForestClassifier",
            "version": 1,
        }

        joblib.dump(
            artifact,
            self.model_file,
        )

        LOGGER.info(
            "Fault classifier saved to: %s",
            self.model_file,
        )

    def load_model(self) -> bool:
        """Load an existing trained classifier from disk.

        Returns:
            True when a valid model was loaded.
            False when the ML runtime or model artifact is unavailable.
        """

        if self.model is not None:
            return True

        if self._model_load_attempted:
            return False

        self._model_load_attempted = True

        if not self._check_ml_runtime():
            self.model = None

            self._log_degraded_mode_once()

            return False

        if not self.model_file.exists():
            LOGGER.warning(
                "ML model is not available: %s",
                self.model_file,
            )

            return False

        try:
            import joblib

            artifact = joblib.load(
                self.model_file
            )

        except Exception as exc:
            self.model = None

            self._ml_error = (
                f"Failed to load classifier artifact: {exc}"
            )

            LOGGER.warning(
                "Unable to load ML classifier: %s",
                exc,
            )

            return False

        if not isinstance(
            artifact,
            dict,
        ):
            raise ValueError(
                "Invalid classifier artifact."
            )

        model = artifact.get("model")
        saved_features = artifact.get("features")
        saved_target = artifact.get("target")

        if model is None:
            raise ValueError(
                "Classifier artifact does not contain "
                "a trained model."
            )

        if tuple(saved_features or ()) != FEATURES:
            raise ValueError(
                "Saved model feature definition does not "
                "match the current classifier."
            )

        if saved_target != TARGET:
            raise ValueError(
                "Saved model target does not match "
                "the current classifier."
            )

        self.model = model

        LOGGER.info(
            "Fault classifier loaded: %s",
            self.model_file,
        )

        return True

    def predict(
        self,
        telemetry: JointTelemetry,
        commanded_position: float,
        acceleration: float = 0.0,
    ) -> str:
        """Predict the fault type for one telemetry record.

        If the ML runtime is unavailable, UNKNOWN is returned.
        """

        if not self._ensure_model_loaded():
            return UNKNOWN_FAULT

        try:
            features = self._build_features(
                telemetry=telemetry,
                commanded_position=commanded_position,
                acceleration=acceleration,
            )

            prediction = self.model.predict(
                [features]
            )[0]

            return str(prediction)

        except Exception as exc:
            LOGGER.warning(
                "ML fault prediction unavailable: %s",
                exc,
            )

            return UNKNOWN_FAULT

    def predict_with_confidence(
        self,
        telemetry: JointTelemetry,
        commanded_position: float,
        acceleration: float = 0.0,
    ) -> tuple[str, float]:
        """Predict a fault type and return model confidence.

        The confidence is the highest class probability produced by
        the Random Forest model. It is model confidence, not a
        probability that the physical robot has failed.

        When the ML runtime is unavailable, the method returns:

            ("UNKNOWN", 0.0)
        """

        if not self._ensure_model_loaded():
            return (
                UNKNOWN_FAULT,
                0.0,
            )

        try:
            features = self._build_features(
                telemetry=telemetry,
                commanded_position=commanded_position,
                acceleration=acceleration,
            )

            probabilities = (
                self.model.predict_proba(
                    [features]
                )[0]
            )

            predicted_index = int(
                probabilities.argmax()
            )

            predicted_fault = str(
                self.model.classes_[
                    predicted_index
                ]
            )

            confidence = float(
                probabilities[
                    predicted_index
                ]
            )

            return (
                predicted_fault,
                round(
                    confidence,
                    4,
                ),
            )

        except Exception as exc:
            LOGGER.warning(
                "ML confidence prediction unavailable: %s",
                exc,
            )

            return (
                UNKNOWN_FAULT,
                0.0,
            )

    @staticmethod
    def _build_features(
        telemetry: JointTelemetry,
        commanded_position: float,
        acceleration: float,
    ) -> list[float]:
        """Convert telemetry into exact training feature order."""

        position_error = (
            commanded_position
            - telemetry.position
        )

        return [
            float(telemetry.joint_index),
            float(commanded_position),
            float(telemetry.position),
            float(position_error),
            float(telemetry.velocity),
            float(acceleration),
            float(telemetry.force),
            float(telemetry.motor_current),
            float(telemetry.temperature),
        ]

    def _ensure_model_loaded(self) -> bool:
        """Ensure a trained model is available."""

        if self.model is not None:
            return True

        return self.load_model()

    def _require_ml_runtime(self) -> None:
        """Require a working scikit-learn runtime."""

        if not self._check_ml_runtime():
            raise RuntimeError(
                "scikit-learn is unavailable in the current "
                "environment. The diagnostic monitor can still "
                "run in evidence-based mode, but ML training "
                "cannot be performed."
            )

    def _check_ml_runtime(self) -> bool:
        """Check and cache scikit-learn runtime availability."""

        if self._ml_available is not None:
            return self._ml_available

        try:
            from sklearn.ensemble import (
                RandomForestClassifier,
            )

            del RandomForestClassifier

        except Exception as exc:
            self._ml_available = False
            self._ml_error = str(exc)

            self._log_runtime_warning_once(
                exc
            )

            return False

        self._ml_available = True
        self._ml_error = None

        LOGGER.info(
            "scikit-learn runtime is available."
        )

        return True

    def _log_runtime_warning_once(
        self,
        exc: Exception,
    ) -> None:
        """Log the native runtime failure only once."""

        if self._runtime_warning_logged:
            return

        self._runtime_warning_logged = True

        LOGGER.warning(
            "scikit-learn runtime unavailable: %s",
            exc,
        )

    def _log_degraded_mode_once(self) -> None:
        """Log degraded ML mode only once."""

        if self._degraded_mode_logged:
            return

        self._degraded_mode_logged = True

        LOGGER.warning(
            "ML classifier unavailable. "
            "RoboTwin-AI will continue in "
            "evidence-based diagnostic mode."
        )

    @staticmethod
    def _print_results(
        accuracy: float,
        report: str,
        matrix: Any,
        labels: list[str],
    ) -> None:
        """Display model evaluation results."""

        print()
        print("RoboTwin-AI Fault Classifier")
        print("============================")
        print()

        print("Dataset")
        print("-------")
        print(f"Features : {len(FEATURES)}")
        print(f"Target   : {TARGET}")
        print()

        print("Model")
        print("-----")
        print("Algorithm       : Random Forest")
        print("Estimators      : 200")
        print("Max depth       : 12")
        print("Min samples leaf: 2")
        print("Class weight    : balanced")
        print()

        print("Accuracy")
        print("--------")
        print(f"{accuracy:.4f}")
        print(f"{accuracy:.2%}")
        print()

        print("Classification Report")
        print("---------------------")
        print(report)

        print("Confusion Matrix")
        print("----------------")
        print("Rows    = Actual")
        print("Columns = Predicted")
        print()

        header = "Actual \\ Predicted"

        print(
            f"{header:<22}",
            end="",
        )

        for label in labels:
            print(
                f"{label:<20}",
                end="",
            )

        print()

        for label, row in zip(
            labels,
            matrix,
        ):
            print(
                f"{label:<22}",
                end="",
            )

            for value in row:
                print(
                    f"{value:<20}",
                    end="",
                )

            print()


def main() -> None:
    """Train and save the clean-label fault classifier."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    classifier = FaultClassifier()

    try:
        classifier.train()
        classifier.save()

    except Exception:
        LOGGER.exception(
            "Fault classifier training failed."
        )
        raise


if __name__ == "__main__":
    main()