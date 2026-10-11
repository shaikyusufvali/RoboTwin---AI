"""Real-time robot telemetry monitoring and fault diagnosis."""

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
from robot_twin.simulation.robot_simulator import (
    RobotSimulator,
    SimulationConfig,
)


LOGGER = logging.getLogger(__name__)

MODEL_PATH = Path("data") / "fault_classifier_v2.pkl"

MONITORED_JOINT_INDEX = 2
MONITORING_DURATION_SECONDS = 15.0


def load_model(path: Path) -> dict[str, Any]:
    """Load the trained fault-classification package."""

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

    missing_keys = required_keys - package.keys()

    if missing_keys:
        raise ValueError(
            "Invalid model package. Missing keys: "
            f"{sorted(missing_keys)}"
        )

    return package


def calculate_features(
    state: dict[str, float | int],
) -> dict[str, float]:
    """Convert robot telemetry into model features."""

    position = float(state["position"])
    velocity = float(state["velocity"])
    force = float(state["force"])

    position_error = position

    motor_current = abs(force) / 20.0
    temperature = 25.0 + abs(motor_current) * 10.0

    return {
        "position": position,
        "position_error": position_error,
        "velocity": velocity,
        "acceleration": 0.0,
        "force": force,
        "motor_current": motor_current,
        "temperature": temperature,
        "abs_position_error": abs(position_error),
        "abs_velocity": abs(velocity),
        "abs_acceleration": 0.0,
        "force_per_current": (
            force / max(abs(motor_current), 1e-6)
        ),
        "current_per_force": (
            abs(motor_current)
            / max(abs(force), 1e-6)
        ),
        "temperature_current_ratio": (
            temperature
            / max(abs(motor_current), 1e-6)
        ),
    }


def predict_fault(
    package: dict[str, Any],
    features: dict[str, float],
) -> tuple[str, float]:
    """Predict the current robot fault."""

    feature_columns = package["feature_columns"]
    scaler = package["scaler"]
    model = package["model"]

    values = np.asarray(
        [
            features[column]
            for column in feature_columns
        ],
        dtype=np.float64,
    ).reshape(1, -1)

    scaled_values = scaler.transform(values)

    prediction = model.predict(scaled_values)[0]

    probabilities = model.predict_proba(
        scaled_values
    )[0]

    confidence = float(
        np.max(probabilities)
    )

    return str(prediction), confidence


def create_evidence(
    features: dict[str, float],
) -> DiagnosticEvidence:
    """Create diagnostic evidence from telemetry."""

    return DiagnosticEvidence(
        position_error=features["position_error"],
        velocity=features["velocity"],
        force=features["force"],
        motor_current=features["motor_current"],
        temperature=features["temperature"],
    )


def monitor_robot(
    simulator: RobotSimulator,
    package: dict[str, Any],
) -> None:
    """Monitor one robot joint and diagnose its behavior."""

    engine = DiagnosticEngine()

    LOGGER.info(
        "Starting real-time monitoring."
    )

    LOGGER.info(
        "Monitoring joint: %d",
        MONITORED_JOINT_INDEX,
    )

    start_time = simulator.config.time_step
    elapsed_time = 0.0

    last_reported_fault: str | None = None

    while elapsed_time < MONITORING_DURATION_SECONDS:
        simulator.step()

        elapsed_time += start_time

        joint_states = simulator.get_joint_states()

        monitored_state = next(
            (
                state
                for state in joint_states
                if state["joint_index"]
                == MONITORED_JOINT_INDEX
            ),
            None,
        )

        if monitored_state is None:
            raise RuntimeError(
                f"Joint {MONITORED_JOINT_INDEX} "
                "was not found."
            )

        features = calculate_features(
            monitored_state
        )

        fault_type, confidence = predict_fault(
            package,
            features,
        )

        if fault_type != last_reported_fault:
            LOGGER.info(
                "Joint %d prediction: %s "
                "(%.1f%% confidence)",
                MONITORED_JOINT_INDEX,
                fault_type,
                confidence * 100,
            )

            evidence = create_evidence(
                features
            )

            report = engine.create_report(
                joint_index=MONITORED_JOINT_INDEX,
                fault_type=fault_type,
                confidence=confidence,
                evidence=evidence,
            )

            print(
                format_report(report)
            )

            last_reported_fault = fault_type


def main() -> None:
    """Start the RoboTwin-AI real-time monitor."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    package = load_model(MODEL_PATH)

    config = SimulationConfig(
        gui=True,
        real_time=False,
    )

    try:
        with RobotSimulator(config) as simulator:
            monitor_robot(
                simulator,
                package,
            )
    except KeyboardInterrupt:
        LOGGER.info(
            "Monitoring stopped by user."
        )


if __name__ == "__main__":
    main()