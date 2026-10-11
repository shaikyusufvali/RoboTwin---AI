"""Safety decision layer for RoboTwin-AI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from robot_twin.vision.obstacle_detector import (
    ObstacleDetection,
)


LOGGER = logging.getLogger(__name__)


class SafetyAction(str, Enum):
    """Action selected by the safety controller."""

    ALLOW = "allow"
    SLOW = "slow"
    STOP = "stop"


@dataclass(frozen=True)
class SafetyConfig:
    """Safety distance configuration."""

    stop_distance: float = 1.0
    warning_distance: float = 2.0


@dataclass(frozen=True)
class SafetyDecision:
    """Decision produced by the safety controller."""

    action: SafetyAction
    distance: float
    reason: str


class SafetyController:
    """Convert obstacle information into safe robot actions."""

    def __init__(
        self,
        config: SafetyConfig | None = None,
    ) -> None:
        self._config = config or SafetyConfig()

        if (
            self._config.stop_distance
            >= self._config.warning_distance
        ):
            raise ValueError(
                "stop_distance must be smaller than "
                "warning_distance."
            )

    def evaluate(
        self,
        detection: ObstacleDetection,
    ) -> SafetyDecision:
        """Evaluate the current obstacle state."""

        distance = detection.minimum_distance

        if distance <= self._config.stop_distance:
            decision = SafetyDecision(
                action=SafetyAction.STOP,
                distance=distance,
                reason=(
                    "Obstacle is inside the emergency "
                    "stop distance."
                ),
            )

        elif distance <= self._config.warning_distance:
            decision = SafetyDecision(
                action=SafetyAction.SLOW,
                distance=distance,
                reason=(
                    "Obstacle is inside the warning "
                    "distance."
                ),
            )

        else:
            decision = SafetyDecision(
                action=SafetyAction.ALLOW,
                distance=distance,
                reason=(
                    "Path is clear within the configured "
                    "safety limits."
                ),
            )

        LOGGER.info(
            "Safety decision: %s | distance=%.2f m",
            decision.action.value,
            decision.distance,
        )

        return decision