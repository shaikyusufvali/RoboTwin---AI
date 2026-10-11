"""Live multimodal diagnostic runtime for the RoboTwin-AI simulator."""

from __future__ import annotations

from robot_twin.diagnostics.diagnostic_brain import DiagnosticBrain
from robot_twin.diagnostics.multimodal_diagnostic_brain import (
    MultimodalDiagnosticBrain,
)
from robot_twin.simulation.robot_simulator import RobotSimulator
from robot_twin.telemetry.pybullet_adapter import PyBulletRobotAdapter
from robot_twin.vision.camera import RobotCamera
from robot_twin.vision.obstacle_detector import ObstacleDetector
from robot_twin.vision.vision_observer import VisionObserver


JOINT_INDEX = 2


def print_report(result) -> None:
    """Print the final multimodal diagnostic result."""

    print()
    print("ROBOT MULTIMODAL LIVE DIAGNOSTIC")
    print("=" * 60)
    print(f"Joint                : {result.joint_index}")
    print(f"Status               : {result.status}")
    print(f"Fault                : {result.fault_type}")
    print(f"Severity             : {result.severity}")
    print(
        f"Telemetry confidence : "
        f"{result.telemetry_confidence:.4f}"
    )
    print(
        f"Vision score         : "
        f"{result.vision_score:.4f}"
    )
    print(
        f"Combined confidence : "
        f"{result.combined_score:.4f}"
    )
    print(
        f"Obstacle detected    : "
        f"{result.obstacle_detected}"
    )

    print()
    print("Evidence:")

    for evidence in result.evidence:
        print(f"  - {evidence}")

    print()
    print("Probable Cause:")
    print(f"  {result.probable_cause}")

    print()
    print("Recommended Action:")
    print(f"  {result.recommended_action}")

    print("=" * 60)


def main() -> None:
    """Run live telemetry and multimodal diagnosis."""

    simulator = RobotSimulator()
    adapter = PyBulletRobotAdapter(simulator)

    try:
        adapter.connect()

        camera = RobotCamera(simulator)
        obstacle_detector = ObstacleDetector()

        vision_observer = VisionObserver(
            camera=camera,
            obstacle_detector=obstacle_detector,
        )

        diagnostic_brain = DiagnosticBrain()
        multimodal_brain = MultimodalDiagnosticBrain()

        telemetry = adapter.get_joint_telemetry(
            joint_index=JOINT_INDEX
        )

        target_position = telemetry.position

        diagnostic_result = diagnostic_brain.diagnose(
            telemetry=telemetry,
            target_position=target_position,
        )

        vision_observation = vision_observer.observe()

        multimodal_result = multimodal_brain.diagnose(
            diagnostic_result=diagnostic_result,
            vision_observation=vision_observation,
        )

        print_report(multimodal_result)

    finally:
        adapter.disconnect()


if __name__ == "__main__":
    main()