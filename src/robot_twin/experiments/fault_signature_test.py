"""Controlled fault-signature validation for RoboTwin-AI."""

from __future__ import annotations

from dataclasses import dataclass

from robot_twin.diagnostics.fault_injector import (
    FaultInjector,
    FaultScenario,
    FaultType,
)
from robot_twin.simulation.robot_simulator import RobotSimulator
from robot_twin.telemetry.pybullet_adapter import PyBulletRobotAdapter


@dataclass(frozen=True)
class SignatureResult:
    """Measured telemetry signature for one fault condition."""

    fault_type: FaultType
    severity: float
    force: float
    position: float
    velocity: float
    motor_current: float
    temperature: float
    fault_load: float
    position_error: float


JOINT_INDEX = 2
FAULT_SEVERITY = 0.8

# Number of physics steps used to reach a stable operating state.
SETTLE_STEPS = 180


def settle_simulation(
    simulator: RobotSimulator,
    steps: int = SETTLE_STEPS,
) -> None:
    """Advance the simulator for a controlled number of steps."""

    if steps <= 0:
        raise ValueError(
            "steps must be greater than zero."
        )

    for _ in range(steps):
        simulator.step()


def collect_signature(
    simulator: RobotSimulator,
    adapter: PyBulletRobotAdapter,
    fault_type: FaultType,
    severity: float,
) -> SignatureResult:
    """
    Apply one fault condition and collect its telemetry signature.

    Every experiment starts from the nominal control condition so
    that measurements are not contaminated by the previous fault.
    """

    # ---------------------------------------------------------------
    # Step 1: Return the joint to nominal operating condition.
    # ---------------------------------------------------------------

    adapter.apply_fault(
        FaultScenario(
            fault_type=FaultType.NORMAL,
            joint_index=JOINT_INDEX,
            severity=0.0,
        )
    )

    settle_simulation(simulator)

    # ---------------------------------------------------------------
    # Step 2: Apply the requested experimental condition.
    # ---------------------------------------------------------------

    scenario = FaultScenario(
        fault_type=fault_type,
        joint_index=JOINT_INDEX,
        severity=severity,
    )

    adapter.apply_fault(scenario)

    settle_simulation(simulator)

    # ---------------------------------------------------------------
    # Step 3: Read synchronized telemetry.
    # ---------------------------------------------------------------

    telemetry = adapter.get_joint_telemetry(
        JOINT_INDEX
    )

    fault_telemetry = (
        adapter._fault_injector.get_fault_telemetry(
            JOINT_INDEX
        )
    )

    commanded_position = (
        fault_telemetry.target_position
    )

    position_error = (
        commanded_position
        - telemetry.position
    )

    result = SignatureResult(
        fault_type=fault_type,
        severity=severity,
        force=telemetry.force,
        position=telemetry.position,
        velocity=telemetry.velocity,
        motor_current=telemetry.motor_current,
        temperature=telemetry.temperature,
        fault_load=fault_telemetry.fault_load,
        position_error=position_error,
    )

    # ---------------------------------------------------------------
    # Step 4: Clear the fault before the next experiment.
    # ---------------------------------------------------------------

    adapter.clear_fault(
        JOINT_INDEX
    )

    return result


def print_signature(
    result: SignatureResult,
) -> None:
    """Print one measured fault signature."""

    print()
    print(
        f"Fault: {result.fault_type.value}"
    )

    print(
        f"  Severity       : "
        f"{result.severity:.2f}"
    )

    print(
        f"  Force          : "
        f"{result.force:.6f}"
    )

    print(
        f"  Position       : "
        f"{result.position:.6f}"
    )

    print(
        f"  Velocity       : "
        f"{result.velocity:.6f}"
    )

    print(
        f"  Motor Current  : "
        f"{result.motor_current:.6f}"
    )

    print(
        f"  Temperature    : "
        f"{result.temperature:.6f}"
    )

    print(
        f"  Fault Load     : "
        f"{result.fault_load:.6f}"
    )

    print(
        f"  Position Error : "
        f"{result.position_error:.6f}"
    )


def print_comparison(
    results: list[SignatureResult],
) -> None:
    """Print all experimental signatures in tabular form."""

    print()
    print("=" * 118)
    print(
        "ROBOTWIN-AI FAULT SIGNATURE COMPARISON"
    )
    print("=" * 118)

    header = (
        f"{'Fault':<22}"
        f"{'Severity':>10}"
        f"{'Force':>12}"
        f"{'Position':>12}"
        f"{'Velocity':>12}"
        f"{'Current':>12}"
        f"{'Temp':>12}"
        f"{'Load':>12}"
        f"{'Pos Error':>12}"
    )

    print(header)
    print("-" * len(header))

    for result in results:
        print(
            f"{result.fault_type.value:<22}"
            f"{result.severity:>10.2f}"
            f"{result.force:>12.4f}"
            f"{result.position:>12.4f}"
            f"{result.velocity:>12.4f}"
            f"{result.motor_current:>12.4f}"
            f"{result.temperature:>12.4f}"
            f"{result.fault_load:>12.4f}"
            f"{result.position_error:>12.4f}"
        )

    print("=" * 118)


def main() -> None:
    """Run the controlled fault-signature experiment."""

    simulator = RobotSimulator()

    injector = FaultInjector(
        simulator
    )

    adapter = PyBulletRobotAdapter(
        simulator=simulator,
        fault_injector=injector,
    )

    results: list[SignatureResult] = []

    fault_types = (
        FaultType.NORMAL,
        FaultType.MOTOR_RESISTANCE,
        FaultType.LOW_MOTOR_POWER,
        FaultType.POSITION_DRIFT,
    )

    try:
        # -----------------------------------------------------------
        # Use the adapter's existing lifecycle.
        #
        # adapter.connect() internally performs:
        #
        # simulator.connect()
        # simulator.load_world()
        # simulator.load_robot()
        # -----------------------------------------------------------

        adapter.connect()

        print()
        print("=" * 118)
        print(
            "ROBOTWIN-AI CONTROLLED "
            "FAULT SIGNATURE TEST"
        )
        print("=" * 118)

        print(
            f"Joint index     : {JOINT_INDEX}"
        )

        print(
            f"Fault severity  : "
            f"{FAULT_SEVERITY:.2f}"
        )

        print(
            f"Settle steps    : "
            f"{SETTLE_STEPS}"
        )

        print(
            f"Client ID       : "
            f"{simulator.client_id}"
        )

        print(
            f"Robot ID        : "
            f"{simulator.robot_id}"
        )

        print("=" * 118)

        # -----------------------------------------------------------
        # Execute each controlled condition.
        # -----------------------------------------------------------

        for fault_type in fault_types:

            result = collect_signature(
                simulator=simulator,
                adapter=adapter,
                fault_type=fault_type,
                severity=(
                    0.0
                    if fault_type == FaultType.NORMAL
                    else FAULT_SEVERITY
                ),
            )

            results.append(result)

            print_signature(
                result
            )

        # -----------------------------------------------------------
        # Final comparison.
        # -----------------------------------------------------------

        print_comparison(
            results
        )

    finally:
        adapter.disconnect()


if __name__ == "__main__":
    main()