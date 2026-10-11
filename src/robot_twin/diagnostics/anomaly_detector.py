"""Telemetry anomaly detection for RoboTwin-AI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

from robot_twin.telemetry.robot_adapter import JointTelemetry


LOGGER = logging.getLogger(__name__)

DEFAULT_POSITION_ERROR_LIMIT: Final[float] = 0.15
DEFAULT_VELOCITY_LIMIT: Final[float] = 5.0
DEFAULT_FORCE_LIMIT: Final[float] = 40.0
DEFAULT_CURRENT_LIMIT: Final[float] = 3.5
DEFAULT_TEMPERATURE_LIMIT: Final[float] = 55.0


@dataclass(frozen=True)
class AnomalyThresholds:
    """Thresholds used to identify abnormal telemetry."""

    position_error_limit: float = (
        DEFAULT_POSITION_ERROR_LIMIT
    )
    velocity_limit: float = DEFAULT_VELOCITY_LIMIT
    force_limit: float = DEFAULT_FORCE_LIMIT
    current_limit: float = DEFAULT_CURRENT_LIMIT
    temperature_limit: float = DEFAULT_TEMPERATURE_LIMIT


@dataclass(frozen=True)
class AnomalyResult:
    """Result of telemetry anomaly analysis."""

    joint_index: int
    is_anomaly: bool
    anomaly_score: float
    reasons: tuple[str, ...]


class AnomalyDetector:
    """Detect abnormal robot behavior from telemetry."""

    def __init__(
        self,
        thresholds: AnomalyThresholds | None = None,
    ) -> None:
        self._thresholds = (
            thresholds or AnomalyThresholds()
        )

    def detect(
        self,
        telemetry: JointTelemetry,
        target_position: float,
    ) -> AnomalyResult:
        """Analyze one telemetry sample."""

        position_error = abs(
            target_position - telemetry.position
        )

        reasons: list[str] = []

        if (
            position_error
            > self._thresholds.position_error_limit
        ):
            reasons.append(
                "Position error is above the configured limit."
            )

        if (
            abs(telemetry.velocity)
            > self._thresholds.velocity_limit
        ):
            reasons.append(
                "Joint velocity is above the configured limit."
            )

        if (
            abs(telemetry.force)
            > self._thresholds.force_limit
        ):
            reasons.append(
                "Joint force is above the configured limit."
            )

        if (
            telemetry.motor_current
            > self._thresholds.current_limit
        ):
            reasons.append(
                "Motor current is above the configured limit."
            )

        if (
            telemetry.temperature
            > self._thresholds.temperature_limit
        ):
            reasons.append(
                "Motor temperature is above the configured limit."
            )

        anomaly_score = self._calculate_score(
            position_error=position_error,
            velocity=abs(telemetry.velocity),
            force=abs(telemetry.force),
            current=telemetry.motor_current,
            temperature=telemetry.temperature,
        )

        result = AnomalyResult(
            joint_index=telemetry.joint_index,
            is_anomaly=bool(reasons),
            anomaly_score=anomaly_score,
            reasons=tuple(reasons),
        )

        LOGGER.debug(
            "Joint %d | anomaly=%s | score=%.3f",
            result.joint_index,
            result.is_anomaly,
            result.anomaly_score,
        )

        return result

    def _calculate_score(
        self,
        position_error: float,
        velocity: float,
        force: float,
        current: float,
        temperature: float,
    ) -> float:
        """Calculate a normalized anomaly score."""

        ratios = (
            position_error
            / self._thresholds.position_error_limit,
            velocity
            / self._thresholds.velocity_limit,
            force
            / self._thresholds.force_limit,
            current
            / self._thresholds.current_limit,
            temperature
            / self._thresholds.temperature_limit,
        )

        exceeded = [
            max(0.0, ratio - 1.0)
            for ratio in ratios
        ]

        score = sum(exceeded) / len(exceeded)

        return min(1.0, score)