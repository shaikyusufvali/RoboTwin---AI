"""Test the RoboTwin-AI safety controller."""

from __future__ import annotations

from robot_twin.safety.safety_controller import (
    SafetyAction,
    SafetyController,
)
from robot_twin.vision.obstacle_detector import (
    ObstacleDetection,
)


def main() -> None:
    """Run safety decision tests."""

    controller = SafetyController()

    test_distances = (
        0.60,
        1.50,
        3.00,
    )

    print()
    print("ROBOTWIN-AI SAFETY CONTROLLER")
    print("=" * 45)

    for distance in test_distances:
        detection = ObstacleDetection(
            detected=distance <= 1.0,
            minimum_distance=distance,
            danger_distance=1.0,
            reason="Test obstacle distance.",
        )

        decision = controller.evaluate(detection)

        print()
        print(f"Distance : {distance:.2f} m")
        print(f"Action   : {decision.action.value}")
        print(f"Reason   : {decision.reason}")

        if distance == 0.60:
            assert decision.action == SafetyAction.STOP

        elif distance == 1.50:
            assert decision.action == SafetyAction.SLOW

        else:
            assert decision.action == SafetyAction.ALLOW


if __name__ == "__main__":
    main()