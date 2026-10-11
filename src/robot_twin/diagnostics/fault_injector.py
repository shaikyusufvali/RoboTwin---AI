"""
Fault injection models for RoboTwin-AI.

This module provides controlled actuator-fault simulation for the
RoboTwin-AI digital twin.

The injector preserves the existing RoboTwin-AI telemetry contract so
that PyBulletAdapter, SensorModel, DiagnosticBrain and the temporal
diagnostic pipeline continue to work without modification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Final

import pybullet as p

from robot_twin.simulation.robot_simulator import RobotSimulator


# ======================================================================
# Fault Types
# ======================================================================


class FaultType(str, Enum):
    """Supported simulated robot fault types."""

    NORMAL = "normal"
    MOTOR_RESISTANCE = "motor_resistance"
    LOW_MOTOR_POWER = "low_motor_power"
    POSITION_DRIFT = "position_drift"


# ======================================================================
# Fault Scenario
# ======================================================================


@dataclass(frozen=True)
class FaultScenario:
    """Description of one fault injection scenario."""

    fault_type: FaultType
    joint_index: int
    severity: float = 0.0


# ======================================================================
# Fault Telemetry
# ======================================================================


@dataclass(frozen=True)
class FaultTelemetry:
    """
    Synthetic fault-side telemetry.

    These fields are intentionally preserved because the existing
    PyBulletAdapter and SensorModel depend on this contract.
    """

    joint_index: int
    fault_type: FaultType
    severity: float

    target_position: float
    effective_target_position: float

    position: float
    velocity: float
    force: float

    fault_load: float
    current_bias: float
    temperature_bias: float


# ======================================================================
# Fault Injector
# ======================================================================


class FaultInjector:
    """
    Apply controlled actuator fault behavior to the simulated robot.

    The injector does not perform diagnosis.

    It creates bounded actuator behavior and synthetic fault signatures.

    Diagnostic interpretation remains the responsibility of:

        SensorModel
            ↓
        BaselineAnomalyDetector
            ↓
        FaultClassifier
            ↓
        FaultEvidenceEngine
            ↓
        DiagnosticBrain
            ↓
        TemporalDiagnosticEngine
    """

    # ==================================================================
    # Nominal target
    # ==================================================================

    DEFAULT_TARGET_POSITION: Final[float] = 0.8

    # ==================================================================
    # Actuator limits
    # ==================================================================

    MAX_MOTOR_FORCE: Final[float] = 40.0

    NORMAL_MOTOR_FORCE: Final[float] = 30.0

    # ==================================================================
    # Bounded velocity controller
    # ==================================================================

    # IMPORTANT:
    #
    # We intentionally use VELOCITY_CONTROL instead of POSITION_CONTROL.
    #
    # PyBullet POSITION_CONTROL can generate large transient velocities
    # for this KUKA configuration. A bounded velocity command gives the
    # digital twin a deterministic actuator command.
    #

    MAX_COMMAND_VELOCITY: Final[float] = 0.75

    POSITION_TO_VELOCITY_GAIN: Final[float] = 1.50

    # ==================================================================
    # Target safety
    # ==================================================================

    MAX_TARGET_MARGIN: Final[float] = 0.02

    # ==================================================================
    # Motor resistance
    # ==================================================================

    MOTOR_RESISTANCE_FORCE_REDUCTION: Final[float] = 0.18

    MOTOR_RESISTANCE_MIN_FORCE: Final[float] = 22.0

    MOTOR_RESISTANCE_TARGET_OFFSET: Final[float] = 0.06

    MOTOR_RESISTANCE_FAULT_LOAD: Final[float] = 4.0

    MOTOR_RESISTANCE_CURRENT_BIAS: Final[float] = 6.2

    MOTOR_RESISTANCE_TEMPERATURE_BIAS: Final[float] = 6.0

    # ==================================================================
    # Low motor power
    # ==================================================================

    LOW_POWER_FORCE_REDUCTION: Final[float] = 0.90

    LOW_POWER_MIN_FORCE: Final[float] = 4.0

    LOW_POWER_TARGET_REDUCTION: Final[float] = 0.45

    LOW_POWER_FAULT_LOAD: Final[float] = 1.0

    # ==================================================================
    # Position drift
    # ==================================================================

    POSITION_DRIFT_TARGET_OFFSET: Final[float] = 0.35

    POSITION_DRIFT_FAULT_LOAD: Final[float] = 0.1

    # ==================================================================
    # Constructor
    # ==================================================================

    def __init__(
        self,
        simulator: RobotSimulator,
    ) -> None:

        self._simulator = simulator

        self._fault_telemetry: dict[
            int,
            FaultTelemetry,
        ] = {}

    # ==================================================================
    # Main API
    # ==================================================================

    def apply(
        self,
        scenario: FaultScenario,
    ) -> FaultTelemetry:
        """
        Apply a fault scenario and return its telemetry contract.

        The return value is required by PyBulletAdapter.apply_fault().
        """

        self._validate_scenario(scenario)

        self._validate_joint(
            scenario.joint_index
        )

        if scenario.fault_type == FaultType.NORMAL:

            telemetry = self._apply_normal(
                scenario
            )

        elif scenario.fault_type == FaultType.MOTOR_RESISTANCE:

            telemetry = self._apply_motor_resistance(
                scenario
            )

        elif scenario.fault_type == FaultType.LOW_MOTOR_POWER:

            telemetry = self._apply_low_motor_power(
                scenario
            )

        elif scenario.fault_type == FaultType.POSITION_DRIFT:

            telemetry = self._apply_position_drift(
                scenario
            )

        else:

            raise ValueError(
                "Unsupported fault type: "
                f"{scenario.fault_type}"
            )

        self._fault_telemetry[
            scenario.joint_index
        ] = telemetry

        return telemetry

    # ==================================================================
    # Backward-compatible API
    # ==================================================================

    def apply_fault(
        self,
        scenario: FaultScenario,
    ) -> FaultTelemetry:
        """Backward-compatible fault injection API."""

        return self.apply(
            scenario
        )

    # ==================================================================
    # NORMAL
    # ==================================================================

    def _apply_normal(
        self,
        scenario: FaultScenario,
    ) -> FaultTelemetry:
        """Apply healthy actuator behavior."""

        target_position = (
            self.DEFAULT_TARGET_POSITION
        )

        effective_target_position = (
            target_position
        )

        self._set_bounded_velocity_control(
            joint_index=scenario.joint_index,
            target_position=(
                effective_target_position
            ),
            force=self.NORMAL_MOTOR_FORCE,
        )

        position, velocity, force = (
            self._read_joint_state(
                scenario.joint_index
            )
        )

        return FaultTelemetry(
            joint_index=scenario.joint_index,
            fault_type=FaultType.NORMAL,
            severity=0.0,
            target_position=target_position,
            effective_target_position=(
                effective_target_position
            ),
            position=position,
            velocity=velocity,
            force=force,
            fault_load=0.0,
            current_bias=0.0,
            temperature_bias=0.0,
        )

    # ==================================================================
    # MOTOR RESISTANCE
    # ==================================================================

    def _apply_motor_resistance(
        self,
        scenario: FaultScenario,
    ) -> FaultTelemetry:
        """Apply simulated motor/mechanical resistance."""

        severity = scenario.severity

        target_position = (
            self.DEFAULT_TARGET_POSITION
        )

        # --------------------------------------------------------------
        # Reduced actuator capability
        # --------------------------------------------------------------

        available_force = (
            self.MAX_MOTOR_FORCE
            * (
                1.0
                - (
                    self.MOTOR_RESISTANCE_FORCE_REDUCTION
                    * severity
                )
            )
        )

        available_force = max(
            self.MOTOR_RESISTANCE_MIN_FORCE,
            available_force,
        )

        # --------------------------------------------------------------
        # Small effective target reduction
        # --------------------------------------------------------------

        target_offset = (
            self.MOTOR_RESISTANCE_TARGET_OFFSET
            * severity
        )

        effective_target_position = (
            target_position
            - target_offset
        )

        # --------------------------------------------------------------
        # Synthetic mechanical load
        # --------------------------------------------------------------

        fault_load = (
            self.MOTOR_RESISTANCE_FAULT_LOAD
            * severity
        )

        # --------------------------------------------------------------
        # Synthetic electrical signature
        # --------------------------------------------------------------

        severity_scale = math.sqrt(
            severity
        )

        current_bias = (
            self.MOTOR_RESISTANCE_CURRENT_BIAS
            * severity_scale
        )

        temperature_bias = (
            self.MOTOR_RESISTANCE_TEMPERATURE_BIAS
            * severity_scale
        )

        self._set_bounded_velocity_control(
            joint_index=scenario.joint_index,
            target_position=(
                effective_target_position
            ),
            force=available_force,
        )

        position, velocity, force = (
            self._read_joint_state(
                scenario.joint_index
            )
        )

        return FaultTelemetry(
            joint_index=scenario.joint_index,
            fault_type=FaultType.MOTOR_RESISTANCE,
            severity=severity,
            target_position=target_position,
            effective_target_position=(
                effective_target_position
            ),
            position=position,
            velocity=velocity,
            force=force,
            fault_load=fault_load,
            current_bias=current_bias,
            temperature_bias=temperature_bias,
        )

    # ==================================================================
    # LOW MOTOR POWER
    # ==================================================================

    def _apply_low_motor_power(
        self,
        scenario: FaultScenario,
    ) -> FaultTelemetry:
        """Apply simulated low actuator power."""

        severity = scenario.severity

        target_position = (
            self.DEFAULT_TARGET_POSITION
        )

        # --------------------------------------------------------------
        # Strong actuator-force reduction
        # --------------------------------------------------------------

        available_force = (
            self.MAX_MOTOR_FORCE
            * (
                1.0
                - (
                    self.LOW_POWER_FORCE_REDUCTION
                    * severity
                )
            )
        )

        available_force = max(
            self.LOW_POWER_MIN_FORCE,
            available_force,
        )

        # --------------------------------------------------------------
        # Reduced effective target
        # --------------------------------------------------------------

        effective_target_position = (
            target_position
            * (
                1.0
                - (
                    self.LOW_POWER_TARGET_REDUCTION
                    * severity
                )
            )
        )

        # --------------------------------------------------------------
        # Synthetic fault load
        # --------------------------------------------------------------

        fault_load = (
            self.LOW_POWER_FAULT_LOAD
            * severity
        )

        self._set_bounded_velocity_control(
            joint_index=scenario.joint_index,
            target_position=(
                effective_target_position
            ),
            force=available_force,
        )

        position, velocity, force = (
            self._read_joint_state(
                scenario.joint_index
            )
        )

        return FaultTelemetry(
            joint_index=scenario.joint_index,
            fault_type=FaultType.LOW_MOTOR_POWER,
            severity=severity,
            target_position=target_position,
            effective_target_position=(
                effective_target_position
            ),
            position=position,
            velocity=velocity,
            force=force,
            fault_load=fault_load,
            current_bias=0.0,
            temperature_bias=0.0,
        )

    # ==================================================================
    # POSITION DRIFT
    # ==================================================================

    def _apply_position_drift(
        self,
        scenario: FaultScenario,
    ) -> FaultTelemetry:
        """Apply simulated position-feedback drift."""

        severity = scenario.severity

        target_position = (
            self.DEFAULT_TARGET_POSITION
        )

        effective_target_position = (
            target_position
            + (
                self.POSITION_DRIFT_TARGET_OFFSET
                * severity
            )
        )

        fault_load = (
            self.POSITION_DRIFT_FAULT_LOAD
            * severity
        )

        self._set_bounded_velocity_control(
            joint_index=scenario.joint_index,
            target_position=(
                effective_target_position
            ),
            force=self.MAX_MOTOR_FORCE,
        )

        position, velocity, force = (
            self._read_joint_state(
                scenario.joint_index
            )
        )

        return FaultTelemetry(
            joint_index=scenario.joint_index,
            fault_type=FaultType.POSITION_DRIFT,
            severity=severity,
            target_position=target_position,
            effective_target_position=(
                effective_target_position
            ),
            position=position,
            velocity=velocity,
            force=force,
            fault_load=fault_load,
            current_bias=0.0,
            temperature_bias=0.0,
        )

    # ==================================================================
    # BOUNDED VELOCITY CONTROLLER
    # ==================================================================

    def _set_bounded_velocity_control(
        self,
        joint_index: int,
        target_position: float,
        force: float,
    ) -> None:
        """
        Apply a deterministic bounded velocity command.

        Instead of asking PyBullet to directly solve a position target,
        we convert position error into a velocity command.

        position error
              ↓
        proportional gain
              ↓
        velocity command
              ↓
        clamp to ±MAX_COMMAND_VELOCITY
              ↓
        PyBullet VELOCITY_CONTROL
        """

        lower, upper = (
            self._get_joint_limits(
                joint_index
            )
        )

        safe_lower = (
            lower
            + self.MAX_TARGET_MARGIN
        )

        safe_upper = (
            upper
            - self.MAX_TARGET_MARGIN
        )

        safe_target = self._clamp(
            target_position,
            safe_lower,
            safe_upper,
        )

        # --------------------------------------------------------------
        # Read current position
        # --------------------------------------------------------------

        state = p.getJointState(
            self._simulator.robot_id,
            joint_index,
            physicsClientId=(
                self._simulator.client_id
            ),
        )

        current_position = float(
            state[0]
        )

        # --------------------------------------------------------------
        # Position error
        # --------------------------------------------------------------

        position_error = (
            safe_target
            - current_position
        )

        # --------------------------------------------------------------
        # Convert position error into velocity command
        # --------------------------------------------------------------

        target_velocity = (
            position_error
            * self.POSITION_TO_VELOCITY_GAIN
        )

        # --------------------------------------------------------------
        # HARD command limit
        # --------------------------------------------------------------

        target_velocity = self._clamp(
            target_velocity,
            -self.MAX_COMMAND_VELOCITY,
            self.MAX_COMMAND_VELOCITY,
        )

        # --------------------------------------------------------------
        # Force limit
        # --------------------------------------------------------------

        safe_force = self._clamp(
            force,
            0.0,
            self.MAX_MOTOR_FORCE,
        )

        # --------------------------------------------------------------
        # Velocity control
        # --------------------------------------------------------------

        p.setJointMotorControl2(
            bodyUniqueId=(
                self._simulator.robot_id
            ),
            jointIndex=joint_index,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=target_velocity,
            force=safe_force,
            physicsClientId=(
                self._simulator.client_id
            ),
        )

    # ==================================================================
    # Joint state
    # ==================================================================

    def _read_joint_state(
        self,
        joint_index: int,
    ) -> tuple[
        float,
        float,
        float,
    ]:
        """Read current PyBullet joint state."""

        state = p.getJointState(
            self._simulator.robot_id,
            joint_index,
            physicsClientId=(
                self._simulator.client_id
            ),
        )

        position = float(
            state[0]
        )

        velocity = float(
            state[1]
        )

        # Existing RoboTwin-AI contract:
        # state[3] = applied motor torque.
        #
        # Keep the field name "force" because the rest of the
        # RoboTwin-AI architecture already depends on it.
        force = float(
            state[3]
        )

        return (
            position,
            velocity,
            force,
        )

    # ==================================================================
    # Joint limits
    # ==================================================================

    def _get_joint_limits(
        self,
        joint_index: int,
    ) -> tuple[float, float]:
        """Read actual URDF joint limits."""

        if hasattr(
            self._simulator,
            "get_joint_limits",
        ):

            limits = (
                self._simulator.get_joint_limits(
                    joint_index
                )
            )

            lower = float(
                limits[0]
            )

            upper = float(
                limits[1]
            )

            if (
                math.isfinite(lower)
                and math.isfinite(upper)
                and lower < upper
            ):
                return (
                    lower,
                    upper,
                )

        return (
            -2.5,
            2.5,
        )

    # ==================================================================
    # Public telemetry
    # ==================================================================

    def get_fault_telemetry(
        self,
        joint_index: int,
    ) -> FaultTelemetry:
        """Return the last generated fault telemetry."""

        if joint_index in self._fault_telemetry:
            return self._fault_telemetry[
                joint_index
            ]

        return FaultTelemetry(
            joint_index=joint_index,
            fault_type=FaultType.NORMAL,
            severity=0.0,
            target_position=(
                self.DEFAULT_TARGET_POSITION
            ),
            effective_target_position=(
                self.DEFAULT_TARGET_POSITION
            ),
            position=0.0,
            velocity=0.0,
            force=0.0,
            fault_load=0.0,
            current_bias=0.0,
            temperature_bias=0.0,
        )

    def get_target_position(
        self,
        joint_index: int,
    ) -> float:
        """Return the diagnostic target position."""

        telemetry = self.get_fault_telemetry(
            joint_index
        )

        return telemetry.target_position

    # ==================================================================
    # Cleanup
    # ==================================================================

    def clear(
        self,
        joint_index: int,
    ) -> None:
        """Clear one joint's fault state."""

        self._validate_joint_index(
            joint_index
        )

        self._fault_telemetry.pop(
            joint_index,
            None,
        )

        p.setJointMotorControl2(
            bodyUniqueId=(
                self._simulator.robot_id
            ),
            jointIndex=joint_index,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=0.0,
            force=0.0,
            physicsClientId=(
                self._simulator.client_id
            ),
        )

    def clear_all(self) -> None:
        """Clear all fault states."""

        joint_indices = list(
            self._fault_telemetry.keys()
        )

        for joint_index in joint_indices:
            self.clear(
                joint_index
            )

        self._fault_telemetry.clear()

    # ==================================================================
    # Validation
    # ==================================================================

    def _validate_joint(
        self,
        joint_index: int,
    ) -> None:
        """Validate joint index and joint type."""

        self._validate_joint_index(
            joint_index
        )

        joint_count = p.getNumJoints(
            self._simulator.robot_id,
            self._simulator.client_id,
        )

        if joint_index >= joint_count:
            raise ValueError(
                f"Joint index {joint_index} "
                f"does not exist. "
                f"Robot contains "
                f"{joint_count} joints."
            )

        joint_info = p.getJointInfo(
            self._simulator.robot_id,
            joint_index,
            physicsClientId=(
                self._simulator.client_id
            ),
        )

        if (
            joint_info[2]
            != p.JOINT_REVOLUTE
        ):
            raise ValueError(
                f"Joint {joint_index} "
                "is not a revolute joint."
            )

    @staticmethod
    def _validate_joint_index(
        joint_index: int,
    ) -> None:
        """Validate joint index."""

        if not isinstance(
            joint_index,
            int,
        ):
            raise TypeError(
                "joint_index must be an integer."
            )

        if joint_index < 0:
            raise ValueError(
                "joint_index must be non-negative."
            )

    @staticmethod
    def _validate_scenario(
        scenario: FaultScenario,
    ) -> None:
        """Validate a fault scenario."""

        if not isinstance(
            scenario,
            FaultScenario,
        ):
            raise TypeError(
                "scenario must be a FaultScenario."
            )

        if not isinstance(
            scenario.fault_type,
            FaultType,
        ):
            raise TypeError(
                "fault_type must be a FaultType."
            )

        if not isinstance(
            scenario.joint_index,
            int,
        ):
            raise TypeError(
                "joint_index must be an integer."
            )

        if scenario.joint_index < 0:
            raise ValueError(
                "joint_index must be non-negative."
            )

        if not (
            0.0
            <= scenario.severity
            <= 1.0
        ):
            raise ValueError(
                "severity must be between 0.0 and 1.0."
            )

    @staticmethod
    def _clamp(
        value: float,
        lower: float,
        upper: float,
    ) -> float:
        """Clamp a value to a numeric interval."""

        return max(
            lower,
            min(
                upper,
                float(value),
            ),
        )