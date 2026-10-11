from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from robot_twin.telemetry.robot_adapter import JointTelemetry


@dataclass(frozen=True)
class FaultSignature:
    """Structured telemetry signature used for advanced diagnosis."""

    position_error: float
    velocity_magnitude: float
    force_magnitude: float
    motor_current: float
    temperature: float

    effort_level: str
    motion_level: str
    position_deviation: str
    thermal_state: str

    persistence_score: float = 0.0

    def as_dict(self) -> dict[str, float | str]:
        return {
            "position_error": self.position_error,
            "velocity_magnitude": self.velocity_magnitude,
            "force_magnitude": self.force_magnitude,
            "motor_current": self.motor_current,
            "temperature": self.temperature,
            "effort_level": self.effort_level,
            "motion_level": self.motion_level,
            "position_deviation": self.position_deviation,
            "thermal_state": self.thermal_state,
            "persistence_score": self.persistence_score,
        }


class FaultSignatureEngine:
    """
    Converts raw joint telemetry into a structured diagnostic signature.

    This component does not decide the final fault type.
    It only extracts measurable evidence.

    Keeping evidence extraction separate from diagnosis makes the
    diagnostic pipeline easier to test, extend, and validate.
    """

    POSITION_ERROR_LOW = 0.05
    POSITION_ERROR_HIGH = 0.20

    VELOCITY_LOW = 0.50
    VELOCITY_HIGH = 5.00

    FORCE_LOW = 5.00
    FORCE_HIGH = 20.00

    CURRENT_LOW = 0.50
    CURRENT_HIGH = 2.50

    TEMPERATURE_HIGH = 45.00
    TEMPERATURE_CRITICAL = 55.00

    def build_signature(
        self,
        telemetry: JointTelemetry,
        target_position: float,
        persistence_score: float = 0.0,
    ) -> FaultSignature:
        """Build a normalized diagnostic signature from telemetry."""

        position_error = (
            telemetry.position - target_position
        )

        velocity_magnitude = abs(
            telemetry.velocity
        )

        force_magnitude = abs(
            telemetry.force
        )

        motor_current = max(
            0.0,
            telemetry.motor_current,
        )

        temperature = telemetry.temperature

        return FaultSignature(
            position_error=position_error,
            velocity_magnitude=velocity_magnitude,
            force_magnitude=force_magnitude,
            motor_current=motor_current,
            temperature=temperature,
            effort_level=self._classify_effort(
                force_magnitude,
                motor_current,
            ),
            motion_level=self._classify_motion(
                velocity_magnitude,
            ),
            position_deviation=self._classify_position(
                abs(position_error),
            ),
            thermal_state=self._classify_temperature(
                temperature,
            ),
            persistence_score=self._clamp(
                persistence_score,
                0.0,
                1.0,
            ),
        )

    def _classify_effort(
        self,
        force: float,
        current: float,
    ) -> str:
        """Classify actuator effort."""

        if (
            force >= self.FORCE_HIGH
            or current >= self.CURRENT_HIGH
        ):
            return "HIGH"

        if (
            force <= self.FORCE_LOW
            and current <= self.CURRENT_LOW
        ):
            return "LOW"

        return "MODERATE"

    def _classify_motion(
        self,
        velocity: float,
    ) -> str:
        """Classify joint motion."""

        if velocity >= self.VELOCITY_HIGH:
            return "HIGH"

        if velocity <= self.VELOCITY_LOW:
            return "LOW"

        return "MODERATE"

    def _classify_position(
        self,
        position_error: float,
    ) -> str:
        """Classify position deviation."""

        if position_error >= self.POSITION_ERROR_HIGH:
            return "HIGH"

        if position_error <= self.POSITION_ERROR_LOW:
            return "LOW"

        return "MODERATE"

    def _classify_temperature(
        self,
        temperature: float,
    ) -> str:
        """Classify motor thermal state."""

        if temperature >= self.TEMPERATURE_CRITICAL:
            return "CRITICAL"

        if temperature >= self.TEMPERATURE_HIGH:
            return "HIGH"

        return "NORMAL"

    @staticmethod
    def _clamp(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        return max(
            minimum,
            min(value, maximum),
        )