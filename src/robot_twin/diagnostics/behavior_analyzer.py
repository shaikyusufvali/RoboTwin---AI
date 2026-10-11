"""Analyze deviations between expected and actual robot behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from robot_twin.diagnostics.expected_behavior import (
    BehaviorDeviation,
    ExpectedBehavior,
    ExpectedBehaviorModel,
)
from robot_twin.telemetry.robot_adapter import JointTelemetry


DEFAULT_POSITION_THRESHOLD: Final[float] = 0.05
DEFAULT_VELOCITY_THRESHOLD: Final[float] = 0.50
DEFAULT_ACCELERATION_THRESHOLD: Final[float] = 1.00
DEFAULT_FORCE_THRESHOLD: Final[float] = 10.00
DEFAULT_CURRENT_THRESHOLD: Final[float] = 1.00
DEFAULT_TEMPERATURE_THRESHOLD: Final[float] = 10.00


@dataclass(frozen=True)
class DeviationThresholds:
    """Maximum acceptable deviation for each telemetry signal."""

    position: float = DEFAULT_POSITION_THRESHOLD
    velocity: float = DEFAULT_VELOCITY_THRESHOLD
    acceleration: float = DEFAULT_ACCELERATION_THRESHOLD
    force: float = DEFAULT_FORCE_THRESHOLD
    motor_current: float = DEFAULT_CURRENT_THRESHOLD
    temperature: float = DEFAULT_TEMPERATURE_THRESHOLD


@dataclass(frozen=True)
class BehaviorAnalysis:
    """Result of expected-versus-actual behavior analysis."""

    expected: ExpectedBehavior
    deviation: BehaviorDeviation
    anomaly: bool
    anomaly_score: float
    reasons: tuple[str, ...]


class BehaviorAnalyzer:
    """Compare actual robot telemetry with expected behavior."""

    def __init__(
        self,
        model: ExpectedBehaviorModel | None = None,
        thresholds: DeviationThresholds | None = None,
    ) -> None:
        self._model = model or ExpectedBehaviorModel()
        self._thresholds = (
            thresholds or DeviationThresholds()
        )

        self._validate_thresholds()

    def analyze(
        self,
        telemetry: JointTelemetry,
        target_position: float,
        acceleration: float = 0.0,
    ) -> BehaviorAnalysis:
        """Analyze actual telemetry against expected behavior."""

        expected = self._model.predict(
            target_position=target_position,
            current_position=telemetry.position,
            current_velocity=telemetry.velocity,
        )

        deviation = self._model.calculate_deviation(
            expected=expected,
            actual_position=telemetry.position,
            actual_velocity=telemetry.velocity,
            actual_acceleration=acceleration,
            actual_force=telemetry.force,
            actual_motor_current=telemetry.motor_current,
            actual_temperature=telemetry.temperature,
        )

        anomaly, score, reasons = self._evaluate_deviation(
            deviation
        )

        return BehaviorAnalysis(
            expected=expected,
            deviation=deviation,
            anomaly=anomaly,
            anomaly_score=score,
            reasons=tuple(reasons),
        )

    def _evaluate_deviation(
        self,
        deviation: BehaviorDeviation,
    ) -> tuple[bool, float, list[str]]:
        """Evaluate deviation against configured thresholds."""

        checks = (
            (
                "position deviation",
                abs(deviation.position),
                self._thresholds.position,
            ),
            (
                "velocity deviation",
                abs(deviation.velocity),
                self._thresholds.velocity,
            ),
            (
                "acceleration deviation",
                abs(deviation.acceleration),
                self._thresholds.acceleration,
            ),
            (
                "force deviation",
                abs(deviation.force),
                self._thresholds.force,
            ),
            (
                "motor current deviation",
                abs(deviation.motor_current),
                self._thresholds.motor_current,
            ),
            (
                "temperature deviation",
                abs(deviation.temperature),
                self._thresholds.temperature,
            ),
        )

        reasons: list[str] = []
        ratios: list[float] = []

        for name, value, threshold in checks:
            ratio = value / threshold
            ratios.append(ratio)

            if value > threshold:
                reasons.append(
                    f"{name} exceeds expected range"
                )

        anomaly_score = self._calculate_score(ratios)

        return (
            bool(reasons),
            anomaly_score,
            reasons,
        )

    @staticmethod
    def _calculate_score(
        ratios: tuple[float, ...] | list[float],
    ) -> float:
        """Convert deviation ratios into a bounded anomaly score."""

        if not ratios:
            return 0.0

        maximum_ratio = max(ratios)

        return max(
            0.0,
            min(1.0, maximum_ratio),
        )

    def _validate_thresholds(self) -> None:
        """Validate configured deviation thresholds."""

        thresholds = (
            self._thresholds.position,
            self._thresholds.velocity,
            self._thresholds.acceleration,
            self._thresholds.force,
            self._thresholds.motor_current,
            self._thresholds.temperature,
        )

        if any(value <= 0 for value in thresholds):
            raise ValueError(
                "All deviation thresholds must be greater than zero."
            )