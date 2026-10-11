"""Expected robot behavior estimation for RoboTwin-AI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


DEFAULT_POSITION_RESPONSE: Final[float] = 0.8
DEFAULT_MAX_VELOCITY: Final[float] = 2.0
DEFAULT_MAX_ACCELERATION: Final[float] = 5.0
DEFAULT_NOMINAL_CURRENT: Final[float] = 1.0
DEFAULT_NOMINAL_TEMPERATURE: Final[float] = 40.0


@dataclass(frozen=True)
class ExpectedBehavior:
    """Expected state of a robot joint."""

    position: float
    velocity: float
    acceleration: float
    force: float
    motor_current: float
    temperature: float


@dataclass(frozen=True)
class BehaviorDeviation:
    """Difference between expected and actual robot behavior."""

    position: float
    velocity: float
    acceleration: float
    force: float
    motor_current: float
    temperature: float


class ExpectedBehaviorModel:
    """Estimate nominal robot behavior from the commanded state."""

    def __init__(
        self,
        position_response: float = DEFAULT_POSITION_RESPONSE,
        max_velocity: float = DEFAULT_MAX_VELOCITY,
        max_acceleration: float = DEFAULT_MAX_ACCELERATION,
        nominal_current: float = DEFAULT_NOMINAL_CURRENT,
        nominal_temperature: float = DEFAULT_NOMINAL_TEMPERATURE,
    ) -> None:
        if not 0.0 < position_response <= 1.0:
            raise ValueError(
                "position_response must be between 0 and 1."
            )

        if max_velocity <= 0:
            raise ValueError(
                "max_velocity must be greater than zero."
            )

        if max_acceleration <= 0:
            raise ValueError(
                "max_acceleration must be greater than zero."
            )

        if nominal_current < 0:
            raise ValueError(
                "nominal_current cannot be negative."
            )

        if nominal_temperature < 0:
            raise ValueError(
                "nominal_temperature cannot be negative."
            )

        self._position_response = position_response
        self._max_velocity = max_velocity
        self._max_acceleration = max_acceleration
        self._nominal_current = nominal_current
        self._nominal_temperature = nominal_temperature

    def predict(
        self,
        target_position: float,
        current_position: float,
        current_velocity: float = 0.0,
    ) -> ExpectedBehavior:
        """Estimate expected joint behavior."""

        position_error = (
            target_position - current_position
        )

        expected_position = (
            current_position
            + position_error * self._position_response
        )

        expected_velocity = self._calculate_expected_velocity(
            position_error,
            current_velocity,
        )

        expected_acceleration = (
            self._calculate_expected_acceleration(
                expected_velocity,
                current_velocity,
            )
        )

        expected_force = abs(
            position_error
        ) * self._max_acceleration

        expected_current = (
            self._nominal_current
            + abs(position_error) * 0.5
        )

        expected_temperature = (
            self._nominal_temperature
            + expected_current * 2.0
        )

        return ExpectedBehavior(
            position=expected_position,
            velocity=expected_velocity,
            acceleration=expected_acceleration,
            force=expected_force,
            motor_current=expected_current,
            temperature=expected_temperature,
        )

    def calculate_deviation(
        self,
        expected: ExpectedBehavior,
        actual_position: float,
        actual_velocity: float,
        actual_acceleration: float,
        actual_force: float,
        actual_motor_current: float,
        actual_temperature: float,
    ) -> BehaviorDeviation:
        """Calculate actual minus expected behavior."""

        return BehaviorDeviation(
            position=(
                actual_position
                - expected.position
            ),
            velocity=(
                actual_velocity
                - expected.velocity
            ),
            acceleration=(
                actual_acceleration
                - expected.acceleration
            ),
            force=(
                actual_force
                - expected.force
            ),
            motor_current=(
                actual_motor_current
                - expected.motor_current
            ),
            temperature=(
                actual_temperature
                - expected.temperature
            ),
        )

    def _calculate_expected_velocity(
        self,
        position_error: float,
        current_velocity: float,
    ) -> float:
        """Estimate velocity required to reach the target."""

        desired_velocity = (
            position_error
            * self._max_velocity
        )

        if current_velocity != 0.0:
            desired_velocity = (
                desired_velocity
                + current_velocity
            ) / 2.0

        return self._clamp(
            desired_velocity,
            -self._max_velocity,
            self._max_velocity,
        )

    def _calculate_expected_acceleration(
        self,
        expected_velocity: float,
        current_velocity: float,
    ) -> float:
        """Estimate acceleration from velocity change."""

        acceleration = (
            expected_velocity
            - current_velocity
        ) * self._max_acceleration

        return self._clamp(
            acceleration,
            -self._max_acceleration,
            self._max_acceleration,
        )

    @staticmethod
    def _clamp(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        """Clamp a value to a configured range."""

        return max(
            minimum,
            min(value, maximum),
        )