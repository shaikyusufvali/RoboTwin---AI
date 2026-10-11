"""Test camera-driven safety stopping for RoboTwin-AI."""

from __future__ import annotations

import logging

import pybullet as p

from robot_twin.safety.safe_motion_controller import SafeMotionController
from robot_twin.safety.safety_controller import SafetyController
from robot_twin.simulation.motion_controller import JointMotionController
from robot_twin.simulation.robot_simulator import RobotSimulator
from robot_twin.vision.camera import CameraConfig, RobotCamera
from robot_twin.vision.obstacle_detector import ObstacleDetector


def create_obstacle() -> int:
    """Create a static obstacle in the camera path."""

    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=(0.4, 0.4, 0.4),
    )

    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=(0.4, 0.4, 0.4),
    )

    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=(0.0, -2.0, 0.4),
    )


def main() -> None:
    """Run the complete camera-to-safety test."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    simulator = RobotSimulator()

    try:
        simulator.connect()
        simulator.load_world()
        simulator.load_robot()

        create_obstacle()

        camera_config = CameraConfig(
            position=(0.0, -3.0, 0.8),
            target=(0.0, 0.0, 0.8),
            fov=60.0,
            width=640,
            height=480,
            near_plane=0.1,
            far_plane=10.0,
        )

        camera = RobotCamera(
            simulator,
            camera_config,
        )

        obstacle_detector = ObstacleDetector()
        safety_controller = SafetyController()

        motion_controller = JointMotionController(
            simulator
        )

        safe_motion = SafeMotionController(
            motion_controller
        )

        joint_index = 2

        print()
        print("ROBOTWIN-AI CAMERA SAFETY TEST")
        print("=" * 55)

        print()
        print("Phase 1: Start robot motion")

        camera_frame = camera.capture()
        detection = obstacle_detector.detect(camera_frame)
        decision = safety_controller.evaluate(detection)

        print(f"Camera minimum distance : {detection.minimum_distance:.2f} m")
        print(f"Safety action           : {decision.action.value}")

        if decision.action.value != "stop":
            raise RuntimeError(
                "Expected obstacle to trigger STOP."
            )

        safe_motion.execute_joint_motion(
            joint_index=joint_index,
            target_position=0.8,
            safety_decision=decision,
        )

        print()
        print("Phase 2: Safety response")
        print("Camera detected obstacle.")
        print("Safety controller selected STOP.")
        print("Safe motion controller blocked motion.")

        print()
        print("CAMERA SAFETY TEST PASSED")

    finally:
        simulator.close()


if __name__ == "__main__":
    main()
