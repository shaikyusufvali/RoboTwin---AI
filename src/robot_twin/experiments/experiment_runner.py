
"""
Controlled scientific experiment runner for RoboTwin-AI.

This module provides the first research-grade experiment layer for comparing:

    1. Telemetry-only diagnosis
    2. Vision-only diagnosis
    3. Multimodal diagnosis

The runner deliberately separates:

    - experimental condition
    - ground-truth fault
    - observed telemetry
    - observed vision
    - system prediction
    - confidence
    - execution latency

This separation is required for reproducible evaluation and prevents
the evaluation layer from deriving ground truth from model predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
import csv
import math
from pathlib import Path
import time
from typing import Iterable

from robot_twin.diagnostics.fault_injector import FaultType


class DiagnosticMode(str, Enum):
    """Diagnostic systems being compared."""

    TELEMETRY_ONLY = "telemetry_only"
    VISION_ONLY = "vision_only"
    MULTIMODAL = "multimodal"


@dataclass(frozen=True)
class ExperimentScenario:
    """
    Defines one controlled experimental condition.

    Ground truth belongs to the scenario definition, not to the
    diagnostic model.
    """

    scenario_id: str
    name: str
    fault_type: FaultType
    obstacle_present: bool = False
    vision_quality: str = "normal"
    seed: int = 0
    fault_severity: float = 0.70

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError("scenario_id cannot be empty.")

        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.seed < 0:
            raise ValueError("seed cannot be negative.")

        if self.vision_quality not in {
            "normal",
            "poor",
            "unavailable",
        }:
            raise ValueError(
                "vision_quality must be one of: "
                "normal, poor, unavailable."
            )


@dataclass(frozen=True)
class ExperimentObservation:
    """
    Raw evidence collected for one experimental run.

    This object intentionally contains observations rather than predictions.
    """

    joint_index: int

    telemetry_available: bool
    vision_available: bool

    telemetry_anomaly_score: float
    vision_anomaly_score: float
    vision_reliability: float

    obstacle_detected: bool

    def __post_init__(self) -> None:
        if self.joint_index < 0:
            raise ValueError("joint_index cannot be negative.")

        for name, value in (
            ("telemetry_anomaly_score", self.telemetry_anomaly_score),
            ("vision_anomaly_score", self.vision_anomaly_score),
            ("vision_reliability", self.vision_reliability),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")

            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")


@dataclass(frozen=True)
class ExperimentPrediction:
    """Prediction produced by one diagnostic mode."""

    scenario_id: str
    mode: DiagnosticMode

    ground_truth: FaultType
    predicted_fault: FaultType

    confidence: float
    anomaly: bool

    latency_seconds: float

    obstacle_detected: bool
    vision_reliability: float

    correct: bool

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")

        if self.latency_seconds < 0.0:
            raise ValueError("latency_seconds cannot be negative.")


class ExperimentRunner:
    """
    Executes and records controlled RoboTwin-AI experiments.

    This class intentionally does not contain robot-specific fault logic.

    The simulator, telemetry system, vision system, and diagnostic brains
    remain responsible for producing observations and predictions.

    The experiment runner is responsible for scientific bookkeeping.
    """

    def __init__(
        self,
        output_directory: str | Path = "data/experiments",
    ) -> None:
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._results: list[ExperimentPrediction] = []

    @property
    def results(self) -> tuple[ExperimentPrediction, ...]:
        """Return immutable experiment results."""

        return tuple(self._results)

    def record_prediction(
        self,
        *,
        scenario: ExperimentScenario,
        mode: DiagnosticMode,
        predicted_fault: FaultType,
        confidence: float,
        anomaly: bool,
        latency_seconds: float,
        observation: ExperimentObservation,
    ) -> ExperimentPrediction:
        """
        Record one model prediction.

        Ground truth is copied directly from the controlled scenario.
        """

        prediction = ExperimentPrediction(
            scenario_id=scenario.scenario_id,
            mode=mode,
            ground_truth=scenario.fault_type,
            predicted_fault=predicted_fault,
            confidence=self._validate_confidence(confidence),
            anomaly=bool(anomaly),
            latency_seconds=float(latency_seconds),
            obstacle_detected=observation.obstacle_detected,
            vision_reliability=observation.vision_reliability,
            correct=predicted_fault == scenario.fault_type,
        )

        self._results.append(prediction)

        return prediction

    def save_results(
        self,
        filename: str = "experiment_results.csv",
    ) -> Path:
        """Persist all recorded predictions as a CSV dataset."""

        if not filename.endswith(".csv"):
            raise ValueError("filename must end with .csv")

        output_path = self.output_directory / filename

        fieldnames = [
            "scenario_id",
            "mode",
            "ground_truth",
            "predicted_fault",
            "confidence",
            "anomaly",
            "latency_seconds",
            "obstacle_detected",
            "vision_reliability",
            "correct",
        ]

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for result in self._results:
                row = asdict(result)

                row["mode"] = result.mode.value
                row["ground_truth"] = result.ground_truth.value
                row["predicted_fault"] = result.predicted_fault.value

                writer.writerow(row)

        return output_path

    def clear(self) -> None:
        """Clear recorded experiment results."""

        self._results.clear()

    @staticmethod
    def _validate_confidence(value: float) -> float:
        value = float(value)

        if not math.isfinite(value):
            raise ValueError("confidence must be finite.")

        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")

        return value


def build_initial_scenarios() -> tuple[ExperimentScenario, ...]:
    """
    Build the first controlled experiment matrix.

    These scenarios are deliberately small and reproducible.
    Additional repetitions and operating conditions will be added
    after the baseline pipeline is validated.
    """

    return (
        ExperimentScenario(
            scenario_id="S01",
            name="Healthy baseline",
            fault_type=FaultType.NORMAL,
            seed=1001,
        ),
        ExperimentScenario(
            scenario_id="S02",
            name="Motor resistance",
            fault_type=FaultType.MOTOR_RESISTANCE,
            seed=1002,
        ),
        ExperimentScenario(
            scenario_id="S03",
            name="Low motor power",
            fault_type=FaultType.LOW_MOTOR_POWER,
            seed=1003,
        ),
        ExperimentScenario(
            scenario_id="S04",
            name="Position drift",
            fault_type=FaultType.POSITION_DRIFT,
            seed=1004,
        ),
        ExperimentScenario(
            scenario_id="S05",
            name="Motor resistance with obstacle",
            fault_type=FaultType.MOTOR_RESISTANCE,
            obstacle_present=True,
            seed=1005,
        ),
        ExperimentScenario(
            scenario_id="S06",
            name="Motor resistance with poor vision",
            fault_type=FaultType.MOTOR_RESISTANCE,
            vision_quality="poor",
            seed=1006,
        ),
    )


def main() -> None:
    """Run a bookkeeping-only validation of the experiment framework."""

    runner = ExperimentRunner()

    scenarios = build_initial_scenarios()

    print("ROBOTWIN-AI EXPERIMENT FRAMEWORK")
    print("=" * 60)

    for scenario in scenarios:
        print(
            f"{scenario.scenario_id} | "
            f"{scenario.name} | "
            f"fault={scenario.fault_type.value} | "
            f"vision={scenario.vision_quality} | "
            f"obstacle={scenario.obstacle_present} | "
            f"seed={scenario.seed}"
        )

    print()
    print(f"Scenarios defined: {len(scenarios)}")
    print(f"Output directory : {runner.output_directory}")


if __name__ == "__main__":
    main()

