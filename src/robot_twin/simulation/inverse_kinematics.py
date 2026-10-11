"""Inverse kinematics utilities for RoboTwin-AI."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pybullet as p

from robot_twin.simulation.robot_simulator import RobotSimulator

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class IKConfig:
    """Configuration for inverse kinematics."""

    end_effector_link: int = 6
    max_iterations: int = 100
    residual_threshold: float = 1e-4


@dataclass(frozen=True)
class IKResult:
    """Result returned by the inverse kinematics solver."""

    target_position: tuple[float, float, float]
    joint_positions: tuple[float, ...]
    end_effector_link: int


class InverseKinematicsSolver:
    """Calculate robot joint positions for a Cartesian target."""

    def __init__(
        self,
        simulator: RobotSimulator,
        config: IKConfig | None = None,
    ) -> None:
        self._simulator = simulator
        self._config = config or IKConfig()

        self._validate_config()

    def solve(
        self,
        target_position: tuple[float, float, float],
    ) -> IKResult:
        """Calculate joint positions for the requested Cartesian position."""

        self._validate_target_position(target_position)

        robot_id = self._simulator.robot_id
        client_id = self._simulator.client_id

        joint_indices = self._get_revolute_joint_indices()

        if not joint_indices:
            raise RuntimeError(
                "No revolute joints were found on the robot."
            )

        lower_limits: list[float] = []
        upper_limits: list[float] = []
        joint_ranges: list[float] = []
        rest_poses: list[float] = []

        for joint_index in joint_indices:
            joint_info = p.getJointInfo(
                robot_id,
                joint_index,
                physicsClientId=client_id,
            )

            lower_limit = float(joint_info[8])
            upper_limit = float(joint_info[9])

            current_state = p.getJointState(
                robot_id,
                joint_index,
                physicsClientId=client_id,
            )

            current_position = float(current_state[0])

            if lower_limit >= upper_limit:
                lower_limit = current_position - 3.14
                upper_limit = current_position + 3.14

            lower_limits.append(lower_limit)
            upper_limits.append(upper_limit)
            joint_ranges.append(upper_limit - lower_limit)
            rest_poses.append(current_position)

        joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=robot_id,
            endEffectorLinkIndex=self._config.end_effector_link,
            targetPosition=target_position,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=rest_poses,
            maxNumIterations=self._config.max_iterations,
            residualThreshold=self._config.residual_threshold,
            physicsClientId=client_id,
        )

        result = IKResult(
            target_position=target_position,
            joint_positions=tuple(
                float(value)
                for value in joint_positions[: len(joint_indices)]
            ),
            end_effector_link=self._config.end_effector_link,
        )

        LOGGER.info(
            "IK solved: target=(%.3f, %.3f, %.3f) joints=%d",
            target_position[0],
            target_position[1],
            target_position[2],
            len(result.joint_positions),
        )

        for joint_index, position in zip(
            joint_indices,
            result.joint_positions,
        ):
            LOGGER.debug(
                "IK joint %d target: %.4f rad",
                joint_index,
                position,
            )

        return result

    def get_end_effector_position(self) -> tuple[float, float, float]:
        """Return the current end-effector world position."""

        robot_id = self._simulator.robot_id
        client_id = self._simulator.client_id

        link_state = p.getLinkState(
            robot_id,
            self._config.end_effector_link,
            computeForwardKinematics=True,
            physicsClientId=client_id,
        )

        position = link_state[4]

        return (
            float(position[0]),
            float(position[1]),
            float(position[2]),
        )

    def _get_revolute_joint_indices(self) -> tuple[int, ...]:
        """Return robot joint indices corresponding to revolute joints."""

        robot_id = self._simulator.robot_id
        client_id = self._simulator.client_id

        joint_count = p.getNumJoints(
            robot_id,
            physicsClientId=client_id,
        )

        return tuple(
            joint_index
            for joint_index in range(joint_count)
            if p.getJointInfo(
                robot_id,
                joint_index,
                physicsClientId=client_id,
            )[2]
            == p.JOINT_REVOLUTE
        )

    def _validate_config(self) -> None:
        """Validate inverse kinematics configuration."""

        if self._config.end_effector_link < 0:
            raise ValueError(
                "end_effector_link must be non-negative."
            )

        if self._config.max_iterations <= 0:
            raise ValueError(
                "max_iterations must be greater than zero."
            )

        if self._config.residual_threshold <= 0:
            raise ValueError(
                "residual_threshold must be greater than zero."
            )

    @staticmethod
    def _validate_target_position(
        target_position: tuple[float, float, float],
    ) -> None:
        """Validate a Cartesian target position."""

        if len(target_position) != 3:
            raise ValueError(
                "target_position must contain exactly three values."
            )

        if not all(
            isinstance(value, (int, float))
            for value in target_position
        ):
            raise TypeError(
                "target_position values must be numeric."
            )


def main() -> None:
    """Run an inverse kinematics demonstration."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    with RobotSimulator() as simulator:
        solver = InverseKinematicsSolver(simulator)

        current_position = solver.get_end_effector_position()

        target_position = (
            current_position[0] + 0.10,
            current_position[1],
            current_position[2],
        )

        result = solver.solve(target_position)

        print()
        print("ROBOTWIN-AI INVERSE KINEMATICS")
        print("=" * 60)

        print(
            "Current end-effector : "
            f"({current_position[0]:.3f}, "
            f"{current_position[1]:.3f}, "
            f"{current_position[2]:.3f})"
        )

        print(
            "Target position      : "
            f"({target_position[0]:.3f}, "
            f"{target_position[1]:.3f}, "
            f"{target_position[2]:.3f})"
        )

        print(
            "End-effector link    : "
            f"{result.end_effector_link}"
        )

        print(
            "Calculated joints    : "
            f"{len(result.joint_positions)}"
        )

        for joint_index, position in enumerate(
            result.joint_positions
        ):
            print(
                f"Joint {joint_index}: "
                f"{position:.4f} rad"
            )


if __name__ == "__main__":
    main()