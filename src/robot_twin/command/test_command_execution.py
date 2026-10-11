"""Integration test for human command execution."""

from __future__ import annotations

import logging

from robot_twin.command.command_executor import CommandExecutor
from robot_twin.command.command_interpreter import CommandInterpreter
from robot_twin.safety.safe_motion_controller import SafeMotionController
from robot_twin.safety.safety_controller import (
    SafetyAction,
    SafetyController,
)
from robot_twin.simulation.motion_controller import JointMotionController
from robot_twin.simulation.robot_simulator import (
    RobotSimulator,
    SimulationConfig,
)
from robot_twin.vision.camera import (
    CameraConfig,
    RobotCamera,
)
from robot_twin.vision.obstacle_detector import (
    ObstacleDetector,
)


LOGGER = logging.getLogger(__name__)


def create_obstacle() -> int:
    """Create a controlled obstacle in front of the robot camera."""

    import pybullet as p

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


def build_safety_system(
    simulator: RobotSimulator,
) -> tuple[
    RobotCamera,
    ObstacleDetector,
    SafetyController,
]:
    """Create the camera and safety components."""

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

    return (
        camera,
        obstacle_detector,
        safety_controller,
    )


def run_clear_path_test(
    executor: CommandExecutor,
    interpreter: CommandInterpreter,
    camera: RobotCamera,
    obstacle_detector: ObstacleDetector,
    safety_controller: SafetyController,
    simulator: RobotSimulator,
) -> None:
    """Verify that a joint command executes when the path is clear."""

    command_text = "Move joint 2 to 30 degrees"
    command = interpreter.interpret(command_text)

    frame = camera.capture()
    detection = obstacle_detector.detect(frame)

    safety_decision = safety_controller.evaluate(detection)

    print()
    print("PHASE 1: CLEAR PATH")
    print("=" * 55)
    print(f"Human command       : {command_text}")
    print(
        f"Obstacle distance   : "
        f"{detection.minimum_distance:.2f} m"
    )
    print(
        f"Safety action       : "
        f"{safety_decision.action.value}"
    )

    assert safety_decision.action != SafetyAction.STOP

    executed = executor.execute(
        command=command,
        safety_decision=safety_decision,
    )

    assert executed is True

    for _ in range(240):
        simulator.step()

    joint_states = simulator.get_joint_states()

    joint_state = next(
        state
        for state in joint_states
        if state["joint_index"] == 2
    )

    actual_position = float(joint_state["position"])

    print(f"Command executed    : {executed}")
    print(
        f"Joint 2 position    : "
        f"{actual_position:.3f} rad"
    )

    assert abs(actual_position - 0.5236) < 0.05

    print("Clear-path command test PASSED")


def run_obstacle_block_test(
    executor: CommandExecutor,
    interpreter: CommandInterpreter,
    camera: RobotCamera,
    obstacle_detector: ObstacleDetector,
    safety_controller: SafetyController,
) -> None:
    """Verify that an obstacle blocks a movement command."""

    import pybullet as p

    create_obstacle()

    command_text = "Move joint 2 to 60 degrees"
    command = interpreter.interpret(command_text)

    frame = camera.capture()
    detection = obstacle_detector.detect(frame)

    safety_decision = safety_controller.evaluate(detection)

    print()
    print("PHASE 2: OBSTACLE DETECTED")
    print("=" * 55)
    print(f"Human command       : {command_text}")
    print(
        f"Obstacle distance   : "
        f"{detection.minimum_distance:.2f} m"
    )
    print(
        f"Safety action       : "
        f"{safety_decision.action.value}"
    )

    assert detection.detected is True
    assert safety_decision.action == SafetyAction.STOP

    executed = executor.execute(
        command=command,
        safety_decision=safety_decision,
    )

    assert executed is False

    print(f"Command executed    : {executed}")
    print("Movement blocked by safety controller.")
    print("Obstacle-block test PASSED")

    del p


def main() -> None:
    """Run the complete human-command safety integration test."""

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

        motion_controller = JointMotionController(
            simulator
        )

        safe_motion = SafeMotionController(
            motion_controller
        )

        executor = CommandExecutor(
            safe_motion
        )

        interpreter = CommandInterpreter()

        (
            camera,
            obstacle_detector,
            safety_controller,
        ) = build_safety_system(simulator)

        run_clear_path_test(
            executor=executor,
            interpreter=interpreter,
            camera=camera,
            obstacle_detector=obstacle_detector,
            safety_controller=safety_controller,
            simulator=simulator,
        )

        run_obstacle_block_test(
            executor=executor,
            interpreter=interpreter,
            camera=camera,
            obstacle_detector=obstacle_detector,
            safety_controller=safety_controller,
        )

        print()
        print("=" * 60)
        print("ROBOTWIN-AI COMMAND EXECUTION TEST PASSED")
        print("=" * 60)
        print()
        print("Human")
        print("  ↓")
        print("Command Interpreter")
        print("  ↓")
        print("Command Executor")
        print("  ↓")
        print("Camera + Safety")
        print("  ↓")
        print("Safe Motion Controller")
        print("  ↓")
        print("KUKA")

    finally:
        simulator.close()


if __name__ == "__main__":
    main()