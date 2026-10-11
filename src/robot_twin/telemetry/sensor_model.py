"""
Sensor estimation layer for RoboTwin-AI.

This module converts simulation-level actuator behavior into
synthetic sensor measurements suitable for AI development,
diagnostic experiments, and controlled validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


DEFAULT_CURRENT_SCALE: Final[float] = 0.08
DEFAULT_VELOCITY_CURRENT_SCALE: Final[float] = 0.02
DEFAULT_POSITION_ERROR_CURRENT_SCALE: Final[float] = 7.5

DEFAULT_TEMPERATURE_AMBIENT: Final[float] = 25.0
DEFAULT_TEMPERATURE_GAIN: Final[float] = 8.0

DEFAULT_FAULT_CURRENT_OFFSET: Final[float] = 0.0
DEFAULT_FAULT_TEMPERATURE_OFFSET: Final[float] = 0.0


@dataclass(frozen=True)
class SensorModelConfig:
    """Configuration for estimated robot sensor signals."""

    current_scale: float = (
        DEFAULT_CURRENT_SCALE
    )

    velocity_current_scale: float = (
        DEFAULT_VELOCITY_CURRENT_SCALE
    )

    position_error_current_scale: float = (
        DEFAULT_POSITION_ERROR_CURRENT_SCALE
    )

    ambient_temperature: float = (
        DEFAULT_TEMPERATURE_AMBIENT
    )

    temperature_gain: float = (
        DEFAULT_TEMPERATURE_GAIN
    )

    fault_current_offset: float = (
        DEFAULT_FAULT_CURRENT_OFFSET
    )

    fault_temperature_offset: float = (
        DEFAULT_FAULT_TEMPERATURE_OFFSET
    )


@dataclass(frozen=True)
class SensorReadings:
    """Estimated actuator sensor readings."""

    motor_current: float
    temperature: float


class SensorModel:
    """Estimate actuator sensors from simulation telemetry."""

    def __init__(
        self,
        config: SensorModelConfig | None = None,
    ) -> None:
        self.config = (
            config
            or SensorModelConfig()
        )

    def estimate(
        self,
        force: float,
        velocity: float,
        position_error: float = 0.0,
        fault_load: float = 0.0,
        fault_current_bias: float = 0.0,
        fault_temperature_bias: float = 0.0,
    ) -> SensorReadings:
        """
        Estimate motor current and temperature.

        Args:
            force:
                Measured actuator force.

            velocity:
                Current joint velocity.

            position_error:
                Difference between intended target and
                actual joint position.

            fault_load:
                Synthetic actuator-load contribution.

            fault_current_bias:
                Fault-specific current contribution.

            fault_temperature_bias:
                Fault-specific temperature contribution.
        """

        if fault_load < 0.0:
            raise ValueError(
                "fault_load cannot be negative."
            )

        if fault_current_bias < 0.0:
            raise ValueError(
                "fault_current_bias cannot be negative."
            )

        if fault_temperature_bias < 0.0:
            raise ValueError(
                "fault_temperature_bias cannot be negative."
            )

        force_current = (
            abs(force)
            * self.config.current_scale
        )

        velocity_current = (
            abs(velocity)
            * self.config.velocity_current_scale
        )

        position_error_current = (
            abs(position_error)
            * self.config.position_error_current_scale
        )

        fault_current = (
            fault_load
            + self.config.fault_current_offset
            + fault_current_bias
        )

        current = (
            force_current
            + velocity_current
            + position_error_current
            + fault_current
        )

        temperature = (
            self.config.ambient_temperature
            + current
            * self.config.temperature_gain
            + self.config.fault_temperature_offset
            + fault_temperature_bias
        )

        return SensorReadings(
            motor_current=float(
                current
            ),
            temperature=float(
                temperature
            ),
        )