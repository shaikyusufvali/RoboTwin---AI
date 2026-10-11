
"""
Fault evidence engine for RoboTwin-AI.

The evidence engine evaluates relationships between telemetry signals
to identify the most plausible simulated fault condition.

These rules are designed for synthetic simulation data and are not
intended to represent production robot safety logic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaultEvidence:
    """Evidence collected for one fault hypothesis."""

    fault_type: str
    score: float
    reasons: list[str]


class FaultEvidenceEngine:
    """Evaluate telemetry evidence for simulated robot faults."""

    LOW_MOTOR_POWER = "low_motor_power"
    MOTOR_RESISTANCE = "motor_resistance"
    POSITION_DRIFT = "position_drift"
    NORMAL = "normal"

    def __init__(
        self,
        position_error_threshold: float = 0.20,
        severe_position_error_threshold: float = 1.00,
        low_velocity_threshold: float = 0.50,
        high_current_threshold: float = 2.50,
        low_current_threshold: float = 0.50,
        high_temperature_threshold: float = 45.0,
        high_force_threshold: float = 20.0,
        low_force_threshold: float = 5.0,
        directional_position_error_threshold: float = 0.20,
    ) -> None:
        """
        Initialize telemetry thresholds.

        Important design rule:

        The simulated LOW_MOTOR_POWER and POSITION_DRIFT faults
        produce position errors in opposite directions.

        LOW_MOTOR_POWER:
            actual position < commanded position

        POSITION_DRIFT:
            actual position > commanded position

        Therefore the signed position error is intentionally preserved.

        The current digital-twin implementation also shows that
        LOW_MOTOR_POWER can produce moderately elevated motor current
        and temperature compared with the healthy state, while remaining
        well below the much stronger MOTOR_RESISTANCE signature.
        """

        self.position_error_threshold = float(
            position_error_threshold
        )

        self.severe_position_error_threshold = float(
            severe_position_error_threshold
        )

        self.low_velocity_threshold = float(
            low_velocity_threshold
        )

        self.high_current_threshold = float(
            high_current_threshold
        )

        self.low_current_threshold = float(
            low_current_threshold
        )

        self.high_temperature_threshold = float(
            high_temperature_threshold
        )

        self.high_force_threshold = float(
            high_force_threshold
        )

        self.low_force_threshold = float(
            low_force_threshold
        )

        self.directional_position_error_threshold = float(
            directional_position_error_threshold
        )

        self._validate_thresholds()

    def _validate_thresholds(self) -> None:
        """Validate all configured thresholds."""

        thresholds = {
            "position_error_threshold": (
                self.position_error_threshold
            ),
            "severe_position_error_threshold": (
                self.severe_position_error_threshold
            ),
            "low_velocity_threshold": (
                self.low_velocity_threshold
            ),
            "high_current_threshold": (
                self.high_current_threshold
            ),
            "low_current_threshold": (
                self.low_current_threshold
            ),
            "high_temperature_threshold": (
                self.high_temperature_threshold
            ),
            "high_force_threshold": (
                self.high_force_threshold
            ),
            "low_force_threshold": (
                self.low_force_threshold
            ),
            "directional_position_error_threshold": (
                self.directional_position_error_threshold
            ),
        }

        for name, value in thresholds.items():
            if value < 0.0:
                raise ValueError(
                    f"{name} must be non-negative."
                )

        if (
            self.low_current_threshold
            >= self.high_current_threshold
        ):
            raise ValueError(
                "low_current_threshold must be lower than "
                "high_current_threshold."
            )

        if (
            self.low_force_threshold
            >= self.high_force_threshold
        ):
            raise ValueError(
                "low_force_threshold must be lower than "
                "high_force_threshold."
            )

        if (
            self.position_error_threshold
            > self.severe_position_error_threshold
        ):
            raise ValueError(
                "position_error_threshold cannot be greater "
                "than severe_position_error_threshold."
            )

    def analyze(
        self,
        position_error: float,
        velocity: float,
        force: float,
        motor_current: float,
        temperature: float,
    ) -> dict[str, FaultEvidence]:
        """
        Analyze telemetry and return evidence for every hypothesis.

        Position error is intentionally NOT converted to absolute value
        for directional diagnosis.

        Signed error interpretation:

            position_error < 0
                Actual position is below commanded position.

            position_error > 0
                Actual position is above commanded position.
        """

        position_error = float(position_error)
        velocity = float(velocity)
        force = float(force)
        motor_current = float(motor_current)
        temperature = float(temperature)

        abs_position_error = abs(position_error)
        abs_velocity = abs(velocity)
        abs_force = abs(force)

        low_power_score = 0.0
        low_power_reasons: list[str] = []

        resistance_score = 0.0
        resistance_reasons: list[str] = []

        drift_score = 0.0
        drift_reasons: list[str] = []

        normal_score = 0.0
        normal_reasons: list[str] = []

        # ---------------------------------------------------------
        # ACTUATOR EFFORT CLASSIFICATION
        # ---------------------------------------------------------

        resistance_effort_detected = (
            abs_force >= self.high_force_threshold
            or motor_current >= self.high_current_threshold
        )

        # ---------------------------------------------------------
        # LOW MOTOR POWER
        # ---------------------------------------------------------

        # Low actuator output remains useful evidence.
        if abs_force <= self.low_force_threshold:
            low_power_score += 0.25
            low_power_reasons.append(
                "Actuator force output is low."
            )

        # ---------------------------------------------------------
        # MODERATE CURRENT
        # ---------------------------------------------------------
        #
        # Healthy state:
        #     current ≈ 0.46 A
        #
        # Low-power state:
        #     current ≈ 1.16 A
        #
        # Motor-resistance state:
        #     current ≈ 8.45 A
        #
        # Therefore moderate current supports LOW_MOTOR_POWER.
        # Very high current supports MOTOR_RESISTANCE.
        # ---------------------------------------------------------

        if (
            self.low_current_threshold
            < motor_current
            < self.high_current_threshold
        ):
            low_power_score += 0.25
            low_power_reasons.append(
                "Motor current is elevated above the healthy "
                "range but remains below the motor-resistance range."
            )

        # Moderate thermal elevation supports sustained actuator
        # effort without indicating severe resistance.
        if (
            self.high_temperature_threshold
            > temperature
            >= 32.0
        ):
            low_power_score += 0.15
            low_power_reasons.append(
                "Motor temperature is moderately elevated."
            )

        # ---------------------------------------------------------
        # DIRECTIONAL LOW-POWER EVIDENCE
        # ---------------------------------------------------------

        if (
            position_error
            <= -self.directional_position_error_threshold
        ):
            low_power_score += 0.25
            low_power_reasons.append(
                "Actual joint position is below the commanded "
                "position, matching the simulated low-power signature."
            )

        # Active movement while the actuator is experiencing
        # reduced output can support the low-power hypothesis.
        if abs_velocity > self.low_velocity_threshold:
            low_power_score += 0.10
            low_power_reasons.append(
                "Joint remains active while tracking the target."
            )

        # A large absolute position error is additional evidence.
        if abs_position_error >= self.position_error_threshold:
            low_power_score += 0.10
            low_power_reasons.append(
                "Position error is elevated."
            )

        # Normal thermal state is weak supporting evidence against
        # severe mechanical resistance.
        if temperature < self.high_temperature_threshold:
            low_power_score += 0.05

        # ---------------------------------------------------------
        # MOTOR RESISTANCE
        # ---------------------------------------------------------

        if resistance_effort_detected:
            resistance_score += 0.25
            resistance_reasons.append(
                "Actuator effort is elevated while the joint "
                "is experiencing abnormal behavior."
            )

        if (
            abs_position_error >= self.position_error_threshold
            and resistance_effort_detected
        ):
            resistance_score += 0.30
            resistance_reasons.append(
                "Position error is elevated despite strong "
                "actuator effort."
            )

        if (
            abs_position_error
            >= self.severe_position_error_threshold
            and resistance_effort_detected
        ):
            resistance_score += 0.15
            resistance_reasons.append(
                "Position error is severe while actuator "
                "effort remains high."
            )

        if (
            abs_velocity <= self.low_velocity_threshold
            and resistance_effort_detected
        ):
            resistance_score += 0.15
            resistance_reasons.append(
                "Joint velocity is low despite actuator effort."
            )

        if motor_current >= self.high_current_threshold:
            resistance_score += 0.10
            resistance_reasons.append(
                "Motor current is elevated."
            )

        if temperature >= self.high_temperature_threshold:
            resistance_score += 0.05
            resistance_reasons.append(
                "Motor temperature is elevated."
            )

        # ---------------------------------------------------------
        # POSITION DRIFT
        # ---------------------------------------------------------

        if abs_position_error >= self.position_error_threshold:
            drift_score += 0.25
            drift_reasons.append(
                "Position error is significantly elevated."
            )

        if (
            abs_position_error
            >= self.severe_position_error_threshold
        ):
            drift_score += 0.10
            drift_reasons.append(
                "Position error is severe."
            )

        # ---------------------------------------------------------
        # DIRECTIONAL POSITION DRIFT
        # ---------------------------------------------------------

        if (
            position_error
            >= self.directional_position_error_threshold
        ):
            drift_score += 0.45
            drift_reasons.append(
                "Actual joint position is above the commanded "
                "position, matching the simulated position-drift signature."
            )

        # Position drift should not be confused with mechanical
        # resistance. Strong actuator effort indicates resistance.
        if not resistance_effort_detected:
            drift_score += 0.15
            drift_reasons.append(
                "No strong actuator-effort abnormality was detected."
            )

        if (
            position_error
            >= self.directional_position_error_threshold
            and motor_current <= self.low_current_threshold
        ):
            drift_score += 0.10
            drift_reasons.append(
                "Position drift persists without elevated motor current."
            )

        if (
            position_error
            >= self.directional_position_error_threshold
            and abs_force < self.high_force_threshold
        ):
            drift_score += 0.05
            drift_reasons.append(
                "Position drift occurs without high actuator force."
            )

        if (
            abs_velocity > self.low_velocity_threshold
            and position_error
            >= self.directional_position_error_threshold
        ):
            drift_score += 0.05
            drift_reasons.append(
                "Joint remains active while positive position "
                "error persists."
            )

        # ---------------------------------------------------------
        # NORMAL
        # ---------------------------------------------------------
        #
        # IMPORTANT:
        #
        # NORMAL must represent genuinely healthy telemetry.
        #
        # Moderate current and elevated temperature are NOT treated
        # as healthy because they are supporting evidence for
        # LOW_MOTOR_POWER in the current digital twin.
        #
        # Velocity alone also does not prove that the actuator is
        # healthy.
        # ---------------------------------------------------------

        # Small position error supports NORMAL.
        if abs_position_error < self.position_error_threshold:
            normal_score += 0.30
            normal_reasons.append(
                "Position error is within the expected range."
            )

        # Only genuinely healthy current supports NORMAL.
        if motor_current <= self.low_current_threshold:
            normal_score += 0.15
            normal_reasons.append(
                "Motor current is within the healthy range."
            )

        # Active velocity alone does NOT prove healthy operation.
        # Therefore velocity is intentionally not rewarded here.

        # Only genuinely healthy temperature supports NORMAL.
        if temperature < 32.0:
            normal_score += 0.20
            normal_reasons.append(
                "Motor temperature is within the healthy range."
            )

        # Normal actuator force supports NORMAL.
        if abs_force < self.high_force_threshold:
            normal_score += 0.20
            normal_reasons.append(
                "Actuator force is within the expected range."
            )

        return {
            self.LOW_MOTOR_POWER: FaultEvidence(
                fault_type=self.LOW_MOTOR_POWER,
                score=min(low_power_score, 1.0),
                reasons=low_power_reasons,
            ),
            self.MOTOR_RESISTANCE: FaultEvidence(
                fault_type=self.MOTOR_RESISTANCE,
                score=min(resistance_score, 1.0),
                reasons=resistance_reasons,
            ),
            self.POSITION_DRIFT: FaultEvidence(
                fault_type=self.POSITION_DRIFT,
                score=min(drift_score, 1.0),
                reasons=drift_reasons,
            ),
            self.NORMAL: FaultEvidence(
                fault_type=self.NORMAL,
                score=min(normal_score, 1.0),
                reasons=normal_reasons,
            ),
        }

    def get_best_evidence(
        self,
        position_error: float,
        velocity: float,
        force: float,
        motor_current: float,
        temperature: float,
    ) -> FaultEvidence:
        """Return the strongest fault hypothesis."""

        evidence = self.analyze(
            position_error=position_error,
            velocity=velocity,
            force=force,
            motor_current=motor_current,
            temperature=temperature,
        )

        fault_evidence = [
            item
            for item in evidence.values()
            if item.fault_type != self.NORMAL
        ]

        if not fault_evidence:
            return evidence[self.NORMAL]

        best_fault = max(
            fault_evidence,
            key=lambda item: item.score,
        )

        if best_fault.score <= evidence[self.NORMAL].score:
            return evidence[self.NORMAL]

        return best_fault


def main() -> None:
    """Run deterministic evidence-engine demonstrations."""

    engine = FaultEvidenceEngine()

    print("ROBOT FAULT EVIDENCE ENGINE")
    print("=" * 70)

    # -------------------------------------------------------------
    # MOTOR RESISTANCE
    # -------------------------------------------------------------

    resistance = engine.get_best_evidence(
        position_error=-0.20,
        velocity=0.04,
        force=38.0,
        motor_current=3.04,
        temperature=49.3,
    )

    print("\nMOTOR RESISTANCE TEST")
    print("-" * 70)
    print(f"Best evidence : {resistance.fault_type}")
    print(f"Evidence score: {resistance.score:.4f}")

    for reason in resistance.reasons:
        print(f"  - {reason}")

    # -------------------------------------------------------------
    # LOW MOTOR POWER
    # -------------------------------------------------------------

    low_power = engine.get_best_evidence(
        position_error=-0.252,
        velocity=0.05,
        force=0.20,
        motor_current=0.20,
        temperature=30.0,
    )

    print("\nLOW MOTOR POWER TEST")
    print("-" * 70)
    print(f"Best evidence : {low_power.fault_type}")
    print(f"Evidence score: {low_power.score:.4f}")

    for reason in low_power.reasons:
        print(f"  - {reason}")

    # -------------------------------------------------------------
    # POSITION DRIFT
    # -------------------------------------------------------------

    drift = engine.get_best_evidence(
        position_error=0.245,
        velocity=0.05,
        force=0.20,
        motor_current=0.20,
        temperature=30.0,
    )

    print("\nPOSITION DRIFT TEST")
    print("-" * 70)
    print(f"Best evidence : {drift.fault_type}")
    print(f"Evidence score: {drift.score:.4f}")

    for reason in drift.reasons:
        print(f"  - {reason}")

    # -------------------------------------------------------------
    # NORMAL
    # -------------------------------------------------------------

    normal = engine.get_best_evidence(
        position_error=0.01,
        velocity=1.2,
        force=8.0,
        motor_current=0.20,
        temperature=30.0,
    )

    print("\nNORMAL TEST")
    print("-" * 70)
    print(f"Best evidence : {normal.fault_type}")
    print(f"Evidence score: {normal.score:.4f}")

    for reason in normal.reasons:
        print(f"  - {reason}")


if __name__ == "__main__":
    main()

