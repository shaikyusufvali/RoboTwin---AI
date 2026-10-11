from robot_twin.simulation.robot_simulator import RobotSimulator
from robot_twin.telemetry.pybullet_adapter import PyBulletRobotAdapter
from robot_twin.diagnostics.fault_signature_engine import (
    FaultSignatureEngine,
)


def main() -> None:
    simulator = RobotSimulator()
    adapter = PyBulletRobotAdapter(simulator)
    signature_engine = FaultSignatureEngine()

    try:
        adapter.connect()

        telemetry = adapter.get_joint_telemetry(
            joint_index=2
        )

        signature = signature_engine.build_signature(
            telemetry=telemetry,
            target_position=0.8,
        )

        print("\n" + "=" * 70)
        print("ROBOTWIN-AI FAULT SIGNATURE")
        print("=" * 70)

        for key, value in signature.as_dict().items():
            print(f"{key:25}: {value}")

        print("=" * 70)

    finally:
        adapter.disconnect()


if __name__ == "__main__":
    main()