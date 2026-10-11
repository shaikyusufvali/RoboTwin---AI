"""Test safety-controlled KUKA joint motion."""

from __future__ import annotations

import logging

from robot_twin.safety.safe_motion_controller import SafeMotionController
from robot_twin.safety.safety_controller import SafetyAction, SafetyController
from robot_twin.simulation.motion_controller import JointMotionController
from robot_twin.simulation.robot_simulator import RobotSimulator, SimulationConfig
from robot_twin.vision.obstacle_detector import ObstacleDetection


def main() -> None:
    """Verify that safety decisions control robot motion."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    simulator = RobotSimulator(
        SimulationConfig(gui=False)
    )

    try:
        simulator.connect()
        simulator.load_world()
        simulator.load_robot()

        motion_controller = JointMotionController(simulator)
        safe_motion = SafeMotionController(motion_controller)
        safety_controller = SafetyController()

        joint_index = 2
        target_position = 0.8

        clear_detection = ObstacleDetection(
            detected=False,
            minimum_distance=3.0,
            danger_distance=1.0,
            reason="Path is clear.",
        )

        clear_decision = safety_controller.evaluate(
            clear_detection
        )

        print()
        print("ROBOTWIN-AI SAFETY MOTION TEST")
        print("=" * 50)
        print()
        print("Phase 1: Clear path")
        print(f"Safety action: {clear_decision.action.value}")

        allowed = safe_motion.execute_joint_motion(
            joint_index=joint_index,
            target_position=target_position,
            safety_decision=clear_decision,
        )

        print(f"Motion allowed: {allowed}")

        for _ in range(120):
            simulator.step()

        moving_position = simulator.get_joint_states()[2]["position"]

        print(f"Joint position : {moving_position:.3f} rad")

        obstacle_detection = ObstacleDetection(
            detected=True,
            minimum_distance=0.60,
            danger_distance=1.0,
            reason="Obstacle detected.",
        )

        stop_decision = safety_controller.evaluate(
            obstacle_detection
        )

        print()
        print("Phase 2: Obstacle detected")
        print(f"Safety action: {stop_decision.action.value}")

        stopped = safe_motion.execute_joint_motion(
            joint_index=joint_index,
            target_position=target_position,
            safety_decision=stop_decision,
        )

        print(f"Motion allowed: {stopped}")

        assert clear_decision.action == SafetyAction.ALLOW
        assert allowed is True
        assert stop_decision.action == SafetyAction.STOP
        assert stopped is False

        print()
        print("SAFETY MOTION TEST PASSED")

    finally:
        simulator.close()


if __name__ == "__main__":
    main()
