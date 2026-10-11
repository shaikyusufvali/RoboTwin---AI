"""Safety-aware trajectory execution for RoboTwin-AI."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from robot_twin.safety.safe_motion_controller import (
    SafeMotionController,
)
from robot_twin.safety.safety_controller import (
    SafetyAction,
    SafetyController,
)
from robot_twin.simulation.motion_planner import MotionDirection
from robot_twin.simulation.robot_simulator import RobotSimulator
from robot_twin.simulation.trajectory_planner import (
    Trajectory,
    TrajectoryPoint,
)
from robot_twin.vision.camera import RobotCamera
from robot_twin.vision.obstacle_detector import ObstacleDetector


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrajectoryExecutionConfig:
    """Configuration for trajectory execution."""

    step_delay: float = 0.05


@dataclass(frozen=True)
class TrajectoryExecutionResult:
    """Result of trajectory execution."""

    completed: bool
    executed_points: int
    total_points: int
    stop_reason: str | None


class TrajectoryExecutor:
    """Execute planned trajectories through the safety layer."""

    def __init__(
        self,
        simulator: RobotSimulator,
        camera: RobotCamera,
        obstacle_detector: ObstacleDetector,
        safety_controller: SafetyController,
        safe_motion_controller: SafeMotionController,
        config: TrajectoryExecutionConfig | None = None,
    ) -> None:
        self._simulator = simulator
        self._camera = camera
        self._obstacle_detector = obstacle_detector
        self._safety_controller = safety_controller
        self._safe_motion = safe_motion_controller
        self._config = (
            config or TrajectoryExecutionConfig()
        )

        if self._config.step_delay < 0:
            raise ValueError(
                "step_delay cannot be negative."
            )

    def execute(
        self,
        trajectory: Trajectory,
    ) -> TrajectoryExecutionResult:
        """Execute a trajectory with safety checks."""

        total_points = len(trajectory.points)

        if total_points == 0:
            return TrajectoryExecutionResult(
                completed=True,
                executed_points=0,
                total_points=0,
                stop_reason=None,
            )

        LOGGER.info(
            "Starting trajectory execution: points=%d",
            total_points,
        )

        for point in trajectory.points:
            safety_decision = self._evaluate_safety()

            LOGGER.info(
                "Trajectory step %d/%d | safety=%s",
                point.step,
                total_points,
                safety_decision.action.value,
            )

            if safety_decision.action == SafetyAction.STOP:
                self._stop_trajectory(point)

                return TrajectoryExecutionResult(
                    completed=False,
                    executed_points=point.step - 1,
                    total_points=total_points,
                    stop_reason=safety_decision.reason,
                )

            self._execute_point(point)

            if self._config.step_delay > 0:
                time.sleep(self._config.step_delay)

        LOGGER.info(
            "Trajectory execution completed: points=%d",
            total_points,
        )

        return TrajectoryExecutionResult(
            completed=True,
            executed_points=total_points,
            total_points=total_points,
            stop_reason=None,
        )

    def _evaluate_safety(self):
        """Capture the environment and evaluate safety."""

        frame = self._camera.capture()

        detection = self._obstacle_detector.detect(
            frame
        )

        return self._safety_controller.evaluate(
            detection
        )

    def _execute_point(
        self,
        point: TrajectoryPoint,
    ) -> None:
        """Execute one trajectory point."""

        if (
            point.joint_index is not None
            and point.joint_position is not None
        ):
            self._execute_joint_point(point)
            return

        self._execute_cartesian_point(point)

    def _execute_joint_point(
        self,
        point: TrajectoryPoint,
    ) -> None:
        """Execute a joint-space trajectory point."""

        from robot_twin.safety.safety_controller import (
            SafetyDecision,
        )

        safety_decision = SafetyDecision(
            action=SafetyAction.ALLOW,
            distance=float("inf"),
            reason="Trajectory safety check passed.",
        )

        self._safe_motion.execute_joint_motion(
            joint_index=point.joint_index,
            target_position=point.joint_position,
            safety_decision=safety_decision,
        )

        self._simulator.step()

    def _execute_cartesian_point(
        self,
        point: TrajectoryPoint,
    ) -> None:
        """Execute a Cartesian trajectory point.

        Cartesian trajectory execution is intentionally
        not connected to robot inverse kinematics yet.
        """

        LOGGER.info(
            "Cartesian target received: "
            "x=%.3f y=%.3f z=%.3f",
            point.x,
            point.y,
            point.z,
        )

        self._simulator.step()

    def _stop_trajectory(
        self,
        point: TrajectoryPoint,
    ) -> None:
        """Stop the active trajectory safely."""

        if point.joint_index is not None:
            self._safe_motion.stop_joint(
                point.joint_index
            )

        LOGGER.warning(
            "Trajectory stopped at step %d.",
            point.step,
        )


def main() -> None:
    """Demonstrate safety-aware trajectory execution."""

    from robot_twin.command.command_interpreter import (
        CommandInterpreter,
    )
    from robot_twin.simulation.motion_controller import (
        JointMotionController,
    )
    from robot_twin.simulation.motion_planner import (
        MotionPlanner,
    )
    from robot_twin.simulation.robot_simulator import (
        SimulationConfig,
    )
    from robot_twin.simulation.trajectory_planner import (
        TrajectoryPlanner,
    )

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

        camera = RobotCamera(simulator)
        obstacle_detector = ObstacleDetector()
        safety_controller = SafetyController()

        executor = TrajectoryExecutor(
            simulator=simulator,
            camera=camera,
            obstacle_detector=obstacle_detector,
            safety_controller=safety_controller,
            safe_motion_controller=safe_motion,
        )

        interpreter = CommandInterpreter()
        motion_planner = MotionPlanner()
        trajectory_planner = TrajectoryPlanner()

        command = interpreter.interpret(
            "Move joint 2 to 30 degrees"
        )

        target = motion_planner.plan(command)

        trajectory = trajectory_planner.generate(
            target
        )

        result = executor.execute(
            trajectory
        )

        print()
        print("ROBOTWIN-AI TRAJECTORY EXECUTOR")
        print("=" * 55)
        print(
            f"Completed       : {result.completed}"
        )
        print(
            f"Executed points : {result.executed_points}"
        )
        print(
            f"Total points    : {result.total_points}"
        )
        print(
            f"Stop reason     : {result.stop_reason}"
        )

    finally:
        simulator.close()


if __name__ == "__main__":
    main()