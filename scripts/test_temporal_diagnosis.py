
from __future__ import annotations

from robot_twin.simulation.robot_simulator import RobotSimulator

from robot_twin.telemetry.pybullet_adapter import PyBulletRobotAdapter

from robot_twin.diagnostics.fault_injector import (
    FaultInjector,
    FaultScenario,
    FaultType,
)

from robot_twin.diagnostics.diagnostic_brain import DiagnosticBrain

from robot_twin.diagnostics.temporal_diagnostic_engine import (
    TemporalDiagnosticEngine,
)


JOINT_INDEX = 2

SEVERITY = 0.70

SETTLING_STEPS = 240


def run_observation(
    simulator: RobotSimulator,
    adapter: PyBulletRobotAdapter,
    temporal_engine: TemporalDiagnosticEngine,
    fault_type: FaultType,
    observation_number: int,
) -> None:

    # --------------------------------------------------------------
    # Clear previous fault
    # --------------------------------------------------------------

    adapter.clear_fault(
        JOINT_INDEX
    )

    # --------------------------------------------------------------
    # Create new scenario
    # --------------------------------------------------------------

    scenario = FaultScenario(
        fault_type=fault_type,
        joint_index=JOINT_INDEX,
        severity=SEVERITY,
    )

    # --------------------------------------------------------------
    # Apply the fault/controller ONCE
    # --------------------------------------------------------------

    adapter.apply_fault(
        scenario
    )

    # --------------------------------------------------------------
    # Allow the robot to physically settle
    # --------------------------------------------------------------

    for _ in range(SETTLING_STEPS):
        simulator.step()

    # --------------------------------------------------------------
    # Read final telemetry
    # --------------------------------------------------------------

    telemetry = adapter.get_joint_telemetry(
        JOINT_INDEX
    )

    target_position = adapter.get_commanded_position(
        JOINT_INDEX
    )

    # --------------------------------------------------------------
    # Temporal diagnosis
    # --------------------------------------------------------------

    result = temporal_engine.update(
        telemetry=telemetry,
        target_position=target_position,
    )

    # --------------------------------------------------------------
    # Output
    # --------------------------------------------------------------

    print()

    print(
        f"Observation {observation_number}"
    )

    print("-" * 70)

    print(
        f"Injected fault     : "
        f"{fault_type.value}"
    )

    print(
        f"Position           : "
        f"{telemetry.position:.6f}"
    )

    print(
        f"Target             : "
        f"{target_position:.6f}"
    )

    print(
        f"Position error     : "
        f"{target_position - telemetry.position:.6f}"
    )

    print(
        f"Velocity           : "
        f"{telemetry.velocity:.6f}"
    )

    print(
        f"Force/Torque       : "
        f"{telemetry.force:.6f}"
    )

    print(
        f"Motor current      : "
        f"{telemetry.motor_current:.6f}"
    )

    print(
        f"Temperature        : "
        f"{telemetry.temperature:.6f}"
    )

    print(
        f"Diagnosed fault    : "
        f"{result.fault_type}"
    )

    print(
        f"Confidence         : "
        f"{result.confidence:.4f}"
    )

    print(
        f"History size       : "
        f"{result.history_size}"
    )

    print(
        f"Persistence score  : "
        f"{result.persistence_score:.4f}"
    )

    print(
        f"State              : "
        f"{result.state}"
    )

    print(
        f"Trend              : "
        f"{result.trend}"
    )

    print(
        f"Developing         : "
        f"{result.developing}"
    )

    print(
        f"Persistent         : "
        f"{result.persistent}"
    )

    print(
        f"Stable             : "
        f"{result.stable}"
    )

    print(
        f"Message            : "
        f"{result.message}"
    )


# ==============================================================
# MAIN TEST
# ==============================================================

print("=" * 70)

print(
    "TEMPORAL DIAGNOSIS TEST"
)

print("=" * 70)


with RobotSimulator() as simulator:

    injector = FaultInjector(
        simulator
    )

    adapter = PyBulletRobotAdapter(
        simulator=simulator,
        fault_injector=injector,
    )

    adapter.connect()

    diagnostic_brain = DiagnosticBrain()

    temporal_engine = TemporalDiagnosticEngine(
        diagnostic_brain=diagnostic_brain,
        history_size=10,
    )

    # ----------------------------------------------------------
    # Test sequence
    # ----------------------------------------------------------

    sequence = [
        FaultType.NORMAL,
        FaultType.NORMAL,
        FaultType.MOTOR_RESISTANCE,
        FaultType.MOTOR_RESISTANCE,
        FaultType.MOTOR_RESISTANCE,
    ]

    # ----------------------------------------------------------
    # Run observations
    # ----------------------------------------------------------

    for observation_number, fault_type in enumerate(
        sequence,
        start=1,
    ):

        run_observation(
            simulator=simulator,
            adapter=adapter,
            temporal_engine=temporal_engine,
            fault_type=fault_type,
            observation_number=observation_number,
        )

    adapter.disconnect()


print()

print("=" * 70)

print(
    "TEMPORAL TEST COMPLETE"
)

print("=" * 70)

