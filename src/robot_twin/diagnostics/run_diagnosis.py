"""Run end-to-end RoboTwin-AI fault diagnosis."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from robot_twin.diagnostics.diagnostic_engine import (
    DiagnosticEngine,
    DiagnosticEvidence,
    format_report,
)


LOGGER = logging.getLogger(__name__)

MODEL_PATH = Path("data") / "fault_classifier_v2.pkl"

SAMPLE_TELEMETRY = {
    "position": 0.82,
    "position_error": 0.24,
    "velocity": 0.03,
    "acceleration": 0.15,
    "force": 32.0,
    "motor_current": 1.8,
    "temperature": 62.0,
    "abs_position_error": 0.24,
    "abs_velocity": 0.03,
    "abs_acceleration": 0.15,
    "force_per_current": 17.78,
    "current_per_force": 0.05625,
    "temperature_current_ratio": 34.44,
}


def load_model(
    path: Path,
) -> dict[str, Any]:
    """Load the trained classifier package."""

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    with path.open("rb") as file:
        package = pickle.load(file)

    required_keys = {
        "feature_columns",
        "target_column",
        "scaler",
        "model",
    }

    missing_keys = (
        required_keys - package.keys()
    )

    if missing_keys:
        raise ValueError(
            "Invalid model package. Missing keys: "
            f"{sorted(missing_keys)}"
        )

    return package


def predict_fault(
    package: dict[str, Any],
    telemetry: dict[str, float],
) -> tuple[str, float]:
    """Predict the fault type and confidence."""

    feature_columns = package[
        "feature_columns"
    ]

    scaler = package["scaler"]
    model = package["model"]

    features = np.asarray(
        [
            telemetry[column]
            for column in feature_columns
        ],
        dtype=np.float64,
    ).reshape(1, -1)

    scaled_features = scaler.transform(
        features
    )

    prediction = model.predict(
        scaled_features
    )[0]

    probabilities = model.predict_proba(
        scaled_features
    )[0]

    confidence = float(
        np.max(probabilities)
    )

    return str(prediction), confidence


def build_evidence(
    telemetry: dict[str, float],
) -> DiagnosticEvidence:
    """Convert telemetry values into diagnostic evidence."""

    return DiagnosticEvidence(
        position_error=telemetry["position_error"],
        velocity=telemetry["velocity"],
        force=telemetry["force"],
        motor_current=telemetry["motor_current"],
        temperature=telemetry["temperature"],
    )


def run_diagnosis(
    joint_index: int,
    telemetry: dict[str, float],
) -> None:
    """Run prediction and generate the diagnostic report."""

    package = load_model(
        MODEL_PATH
    )

    fault_type, confidence = predict_fault(
        package,
        telemetry,
    )

    LOGGER.info(
        "Predicted fault: %s",
        fault_type,
    )

    LOGGER.info(
        "Prediction confidence: %.2f",
        confidence,
    )

    evidence = build_evidence(
        telemetry
    )

    engine = DiagnosticEngine()

    report = engine.create_report(
        joint_index=joint_index,
        fault_type=fault_type,
        confidence=confidence,
        evidence=evidence,
    )

    print(
        format_report(report)
    )


def main() -> None:
    """Run the RoboTwin-AI end-to-end diagnostic."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    run_diagnosis(
        joint_index=2,
        telemetry=SAMPLE_TELEMETRY,
    )


if __name__ == "__main__":
    main()