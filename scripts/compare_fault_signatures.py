from robot_twin.simulation.robot_simulator import RobotSimulator
from robot_twin.telemetry.pybullet_adapter import PyBulletRobotAdapter
from robot_twin.diagnostics.fault_injector import (
    FaultScenario,
    FaultType,
)
from robot_twin.diagnostics.fault_signature_engine import (
    FaultSignatureEngine,
)


SEVERITY = 0.43
JOINT_INDEX = 2
TARGET_POSITION = 0.8


def run_fault(
    fault_type: FaultType,
) -> dict:
    simulator = RobotSimulator()
    adapter = PyBulletRobotAdapter(simulator)
    signature_engine = FaultSignatureEngine()

    try:
        adapter.connect()

        scenario = FaultScenario(
            fault_type=fault_type,
            joint_index=JOINT_INDEX,
            severity=SEVERITY,
        )

        adapter.apply_fault(scenario)

        # Allow the actuator to respond to the injected condition.
        for _ in range(240):
            simulator.step()

        telemetry = adapter.get_joint_telemetry(
            joint_index=JOINT_INDEX
        )

        signature = signature_engine.build_signature(
            telemetry=telemetry,
            target_position=TARGET_POSITION,
        )

        return signature.as_dict()

    finally:
        adapter.disconnect()


def main() -> None:
    fault_types = [
        FaultType.NORMAL,
        FaultType.MOTOR_RESISTANCE,
        FaultType.LOW_MOTOR_POWER,
        FaultType.POSITION_DRIFT,
    ]

    print("\n" + "=" * 100)
    print("ROBOTWIN-AI FAULT SIGNATURE COMPARISON")
    print("=" * 100)
    print(f"Severity   : {SEVERITY}")
    print(f"Joint      : {JOINT_INDEX}")
    print(f"Target     : {TARGET_POSITION}")
    print("=" * 100)

    results = {}

    for fault_type in fault_types:
        print(f"\nRunning: {fault_type.value}")

        results[fault_type.value] = run_fault(
            fault_type
        )

    numeric_fields = [
        "position_error",
        "velocity_magnitude",
        "force_magnitude",
        "motor_current",
        "temperature",
        "persistence_score",
    ]

    categorical_fields = [
        "effort_level",
        "motion_level",
        "position_deviation",
        "thermal_state",
    ]

    print("\n" + "-" * 100)
    print(
        f"{'SIGNAL':<25}"
        f"{'NORMAL':>18}"
        f"{'RESISTANCE':>18}"
        f"{'LOW_POWER':>18}"
        f"{'DRIFT':>18}"
    )
    print("-" * 100)

    names = {
        "NORMAL": "NORMAL",
        "MOTOR_RESISTANCE": "RESISTANCE",
        "LOW_MOTOR_POWER": "LOW_POWER",
        "POSITION_DRIFT": "DRIFT",
    }

    for field in numeric_fields:
        values = []

        for fault_type in fault_types:
            value = results[fault_type.value][field]
            values.append(f"{value:>18.4f}")

        print(
            f"{field:<25}"
            + "".join(values)
        )

    for field in categorical_fields:
        values = []

        for fault_type in fault_types:
            value = results[fault_type.value][field]
            values.append(f"{value:>18}")

        print(
            f"{field:<25}"
            + "".join(values)
        )

    print("-" * 100)
    print("\nComparison complete.")


if __name__ == "__main__":
    main()