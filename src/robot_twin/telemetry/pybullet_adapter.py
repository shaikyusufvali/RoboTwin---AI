"""PyBullet implementation of the RoboTwin-AI robot adapter."""

from __future__ import annotations

from robot_twin.diagnostics.fault_injector import (
    FaultInjector,
    FaultScenario,
)

from robot_twin.simulation.robot_simulator import (
    RobotSimulator,
)

from robot_twin.telemetry.robot_adapter import (
    JointTelemetry,
    RobotAdapter,
)

from robot_twin.telemetry.sensor_model import (
    SensorModel,
)


class PyBulletRobotAdapter(RobotAdapter):
    """Expose PyBullet robot telemetry through a common interface."""

    def __init__(
        self,
        simulator: RobotSimulator,
        sensor_model: SensorModel | None = None,
        fault_injector: FaultInjector | None = None,
    ) -> None:

        self._simulator = simulator

        self._sensor_model = (
            sensor_model
            or SensorModel()
        )

        self._fault_injector = (
            fault_injector
            or FaultInjector(
                simulator
            )
        )

        self._connected = False

        self._commanded_positions: dict[
            int,
            float,
        ] = {}

        self._fault_loads: dict[
            int,
            float,
        ] = {}

    # ==================================================================
    # Connection
    # ==================================================================

    def connect(self) -> None:
        """Connect the adapter to the simulator."""

        if self._connected:
            return

        self._simulator.connect()

        self._simulator.load_world()

        self._simulator.load_robot()

        self._commanded_positions.clear()

        self._fault_loads.clear()

        self._fault_injector.clear_all()

        self._connected = True

    def disconnect(self) -> None:
        """Disconnect from the simulator."""

        if not self._connected:
            return

        self._fault_injector.clear_all()

        self._simulator.close()

        self._commanded_positions.clear()

        self._fault_loads.clear()

        self._connected = False

    # ==================================================================
    # Command state
    # ==================================================================

    def set_commanded_position(
        self,
        joint_index: int,
        target_position: float,
    ) -> None:

        self._require_connection()

        self._validate_joint_index(
            joint_index
        )

        self._commanded_positions[
            joint_index
        ] = float(
            target_position
        )

    def get_commanded_position(
        self,
        joint_index: int,
    ) -> float:

        self._require_connection()

        self._validate_joint_index(
            joint_index
        )

        if joint_index in self._commanded_positions:
            return self._commanded_positions[
                joint_index
            ]

        return self._fault_injector.get_target_position(
            joint_index
        )

    # ==================================================================
    # Fault load
    # ==================================================================

    def set_fault_load(
        self,
        joint_index: int,
        fault_load: float,
    ) -> None:

        self._require_connection()

        self._validate_joint_index(
            joint_index
        )

        if fault_load < 0.0:
            raise ValueError(
                "fault_load cannot be negative."
            )

        self._fault_loads[
            joint_index
        ] = float(
            fault_load
        )

    def clear_fault_load(
        self,
        joint_index: int,
    ) -> None:

        self._fault_loads.pop(
            joint_index,
            None,
        )

    # ==================================================================
    # Fault control
    # ==================================================================

    def apply_fault(
        self,
        scenario: FaultScenario,
    ) -> None:
        """Apply a fault and synchronize adapter state."""

        self._require_connection()

        fault_telemetry = (
            self._fault_injector.apply(
                scenario
            )
        )

        # Diagnostic target remains the healthy
        # commanded target.
        self.set_commanded_position(
            joint_index=scenario.joint_index,
            target_position=(
                fault_telemetry.target_position
            ),
        )

        self.set_fault_load(
            joint_index=scenario.joint_index,
            fault_load=(
                fault_telemetry.fault_load
            ),
        )

    # ==================================================================
    # Clear one fault
    # ==================================================================

    def clear_fault(
        self,
        joint_index: int,
    ) -> None:

        self._require_connection()

        self._fault_injector.clear(
            joint_index
        )

        self.clear_fault_load(
            joint_index
        )

        self._commanded_positions.pop(
            joint_index,
            None,
        )

    # ==================================================================
    # Clear all faults
    # ==================================================================

    def clear_all_faults(self) -> None:

        self._require_connection()

        self._fault_injector.clear_all()

        self._fault_loads.clear()

        self._commanded_positions.clear()

    # ==================================================================
    # Telemetry
    # ==================================================================

    def get_joint_telemetry(
        self,
        joint_index: int,
    ) -> JointTelemetry:
        """Read telemetry for one robot joint."""

        self._require_connection()

        self._validate_joint_index(
            joint_index
        )

        joint_states = (
            self._simulator.get_joint_states()
        )

        state = next(
            (
                item
                for item in joint_states
                if item["joint_index"]
                == joint_index
            ),
            None,
        )

        if state is None:
            raise ValueError(
                f"Joint {joint_index} "
                f"was not found."
            )

        position = float(
            state["position"]
        )

        velocity = float(
            state["velocity"]
        )

        force = float(
            state["force"]
        )

        commanded_position = (
            self.get_commanded_position(
                joint_index
            )
        )

        position_error = (
            commanded_position
            - position
        )

        fault_telemetry = (
            self._fault_injector.get_fault_telemetry(
                joint_index
            )
        )

        if fault_telemetry is None:
            fault_load = 0.0
            current_bias = 0.0
            temperature_bias = 0.0

        else:
            fault_load = (
                self._fault_loads.get(
                    joint_index,
                    fault_telemetry.fault_load,
                )
            )

            current_bias = (
                fault_telemetry.current_bias
            )

            temperature_bias = (
                fault_telemetry.temperature_bias
            )

        sensor_readings = (
            self._sensor_model.estimate(
                force=force,
                velocity=velocity,
                position_error=position_error,
                fault_load=fault_load,
                fault_current_bias=current_bias,
                fault_temperature_bias=temperature_bias,
            )
        )

        return JointTelemetry(
            joint_index=joint_index,
            position=position,
            velocity=velocity,
            force=force,
            motor_current=(
                sensor_readings.motor_current
            ),
            temperature=(
                sensor_readings.temperature
            ),
        )

    # ==================================================================
    # Status
    # ==================================================================

    def is_connected(self) -> bool:
        return self._connected

    # ==================================================================
    # Validation
    # ==================================================================

    @staticmethod
    def _validate_joint_index(
        joint_index: int,
    ) -> None:

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

    def _require_connection(self) -> None:

        if not self._connected:
            raise RuntimeError(
                "Robot adapter is not connected."
            )