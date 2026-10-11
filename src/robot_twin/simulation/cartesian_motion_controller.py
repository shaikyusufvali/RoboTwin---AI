"""Cartesian motion control for RoboTwin-AI."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from robot_twin.safety.safe_motion_controller import SafeMotionController
from robot_twin.safety.safety_controller import (
    SafetyAction,
    SafetyController,
)
from robot_twin.simulation.inverse_kinematics import (
    InverseKinematicsSolver,
)
from robot_twin.simulation.motion_controller import (
    JointMotionController,
)
from robot_twin.simulation.robot_simulator import RobotSimulator
from robot_twin.vision.camera import RobotCamera
from robot_twin.vision.obstacle_detector import ObstacleDetector


LOGGER = logging.getLogger(__name__)

DEFAULT_POSITION_TOLERANCE: float = 0.01


@dataclass(frozen=True)
class CartesianMotionConfig:
    """Configuration for Cartesian motion validation."""

    position_tolerance: float = DEFAULT_POSITION_TOLERANCE


@dataclass(frozen=True)
class CartesianMotionResult:
    """Result of a Cartesian motion operation."""

    success: bool
    target_position: tuple[float, float, float]
    actual_position: tuple[float, float, float]
    position_error: float
    safety_action: SafetyAction
    safety_distance: float


class CartesianMotionController:
    """Move the robot end effector toward Cartesian targets."""

    def __init__(
        self,
        simulator: RobotSimulator,
        ik_solver: InverseKinematicsSolver,
        motion_controller: JointMotionController,
        safe_motion_controller: SafeMotionController,
        camera: RobotCamera,
        obstacle_detector: ObstacleDetector,
        safety_controller: SafetyController,
        config: CartesianMotionConfig | None = None,
    ) -> None:
        self._simulator = simulator
        self._ik_solver = ik_solver
        self._motion_controller = motion_controller
        self._safe_motion_controller = safe_motion_controller
        self._camera = camera
        self._obstacle_detector = obstacle_detector
        self._safety_controller = safety_controller
        self._config = config or CartesianMotionConfig()

        self._validate_config()

    def move_to(
        self,
        target_position: tuple[float, float, float],
    ) -> CartesianMotionResult:
        """Move the end effector to a Cartesian target."""

        self._validate_target_position(target_position)

        safety_decision = self._evaluate_safety()

        LOGGER.info(
            "Safety decision: %s | distance=%.2f m",
            safety_decision.action.value,
            safety_decision.distance,
        )

        if safety_decision.action == SafetyAction.STOP:
            self._stop_all_joints()

            actual_position = (
                self._ik_solver.get_end_effector_position()
            )

            position_error = self._calculate_position_error(
                target_position,
                actual_position,
            )

            return CartesianMotionResult(
                success=False,
                target_position=target_position,
                actual_position=actual_position,
                position_error=position_error,
                safety_action=safety_decision.action,
                safety_distance=safety_decision.distance,
            )

        ik_result = self._ik_solver.solve(
            target_position=target_position,
        )

        LOGGER.info(
            "IK solved: target=(%.3f, %.3f, %.3f) joints=%d",
            *target_position,
            len(ik_result.joint_positions),
        )

        if safety_decision.action == SafetyAction.SLOW:
            LOGGER.warning(
                "Safety warning active. "
                "Cartesian motion continues with caution."
            )

        motion_success = self._execute_joint_targets(
            ik_result.joint_positions,
        )

        if not motion_success:
            LOGGER.warning(
                "Joint-level convergence was not reached; "
                "Cartesian position will determine final success."
            )

        actual_position = (
            self._ik_solver.get_end_effector_position()
        )

        position_error = self._calculate_position_error(
            target_position,
            actual_position,
        )

        success = (
            position_error
            <= self._config.position_tolerance
        )

        LOGGER.info(
            "Cartesian motion result: "
            "target=(%.3f, %.3f, %.3f) "
            "actual=(%.3f, %.3f, %.3f) "
            "error=%.4f m success=%s",
            *target_position,
            *actual_position,
            position_error,
            success,
        )

        return CartesianMotionResult(
            success=success,
            target_position=target_position,
            actual_position=actual_position,
            position_error=position_error,
            safety_action=safety_decision.action,
            safety_distance=safety_decision.distance,
        )

    def _execute_joint_targets(
        self,
        joint_positions: tuple[float, ...],
    ) -> bool:
        """Execute all IK-generated joint targets simultaneously."""

        target_positions: dict[int, float] = {}

        for joint_index, target_position in enumerate(
            joint_positions
        ):
            if not math.isfinite(target_position):
                raise ValueError(
                    f"Invalid IK target for joint {joint_index}."
                )

            target_positions[joint_index] = target_position

        LOGGER.info(
            "Executing %d IK joint targets simultaneously.",
            len(target_positions),
        )

        return self._motion_controller.move_joints(
            target_positions=target_positions,
        )

    def _evaluate_safety(self):
        """Capture the environment and evaluate motion safety."""

        frame = self._camera.capture()

        detection = self._obstacle_detector.detect(
            frame,
        )

        return self._safety_controller.evaluate(
            detection,
        )

    def _stop_all_joints(self) -> None:
        """Stop all robot joints."""

        joint_states = self._simulator.get_joint_states()

        for state in joint_states:
            joint_index = int(
                state["joint_index"]
            )

            self._safe_motion_controller.stop_joint(
                joint_index,
            )

    @staticmethod
    def _calculate_position_error(
        target_position: tuple[float, float, float],
        actual_position: tuple[float, float, float],
    ) -> float:
        """Calculate Euclidean Cartesian position error."""

        return math.sqrt(
            sum(
                (
                    target - actual
                ) ** 2
                for target, actual in zip(
                    target_position,
                    actual_position,
                )
            )
        )

    @staticmethod
    def _validate_target_position(
        target_position: tuple[float, float, float],
    ) -> None:
        """Validate Cartesian target coordinates."""

        if len(target_position) != 3:
            raise ValueError(
                "target_position must contain exactly "
                "three coordinates."
            )

        if not all(
            isinstance(value, (int, float))
            and math.isfinite(float(value))
            for value in target_position
        ):
            raise ValueError(
                "Cartesian target coordinates must be "
                "finite numeric values."
            )

    def _validate_config(self) -> None:
        """Validate controller configuration."""

        if self._config.position_tolerance <= 0:
            raise ValueError(
                "position_tolerance must be greater than zero."
            )


def main() -> None:
    """Run a Cartesian motion demonstration."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    simulator = RobotSimulator()

    try:
        simulator.connect()
        simulator.load_world()
        simulator.load_robot()

        motion_controller = JointMotionController(
            simulator=simulator,
        )

        safe_motion_controller = SafeMotionController(
            motion_controller=motion_controller,
        )

        ik_solver = InverseKinematicsSolver(
            simulator=simulator,
        )

        camera = RobotCamera(
            simulator,
        )

        obstacle_detector = ObstacleDetector()

        safety_controller = SafetyController()

        controller = CartesianMotionController(
            simulator=simulator,
            ik_solver=ik_solver,
            motion_controller=motion_controller,
            safe_motion_controller=safe_motion_controller,
            camera=camera,
            obstacle_detector=obstacle_detector,
            safety_controller=safety_controller,
        )

        target_position = (
            0.10,
            0.00,
            1.261,
        )

        result = controller.move_to(
            target_position,
        )

        print()
        print("ROBOTWIN-AI CARTESIAN MOTION")
        print("=" * 60)

        print(
            f"Safety action     : "
            f"{result.safety_action.value}"
        )

        print(
            f"Safety distance   : "
            f"{result.safety_distance:.2f} m"
        )

        print(
            f"Target position   : "
            f"({result.target_position[0]:.3f}, "
            f"{result.target_position[1]:.3f}, "
            f"{result.target_position[2]:.3f})"
        )

        print(
            f"Final position    : "
            f"({result.actual_position[0]:.3f}, "
            f"{result.actual_position[1]:.3f}, "
            f"{result.actual_position[2]:.3f})"
        )

        print(
            f"Position error    : "
            f"{result.position_error:.4f} m"
        )

        print(
            f"Motion successful : "
            f"{result.success}"
        )

    finally:
        simulator.close()


if __name__ == "__main__":
    main()