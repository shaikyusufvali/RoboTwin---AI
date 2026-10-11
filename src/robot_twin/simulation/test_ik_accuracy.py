"""Validate inverse kinematics accuracy for RoboTwin-AI."""

from __future__ import annotations

import logging
import math

import pybullet as p

from robot_twin.simulation.inverse_kinematics import (
    InverseKinematicsSolver,
)
from robot_twin.simulation.robot_simulator import RobotSimulator


LOGGER = logging.getLogger(__name__)


def calculate_position_error(
    actual: tuple[float, float, float],
    target: tuple[float, float, float],
) -> float:
    """Calculate Euclidean distance between two Cartesian positions."""

    dx = actual[0] - target[0]
    dy = actual[1] - target[1]
    dz = actual[2] - target[2]

    return math.sqrt(
        dx**2 + dy**2 + dz**2
    )


def main() -> None:
    """Validate the Cartesian accuracy of the IK solution."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    with RobotSimulator() as simulator:
        solver = InverseKinematicsSolver(
            simulator=simulator,
        )

        current_position = (
            solver.get_end_effector_position()
        )

        target_position = (
            current_position[0] + 0.10,
            current_position[1],
            current_position[2],
        )

        ik_result = solver.solve(
            target_position
        )

        LOGGER.info(
            "Applying IK solution directly for validation."
        )

        for joint_index, joint_position in enumerate(
            ik_result.joint_positions
        ):
            p.resetJointState(
                bodyUniqueId=simulator.robot_id,
                jointIndex=joint_index,
                targetValue=joint_position,
                physicsClientId=simulator.client_id,
            )

        simulator.step()

        actual_position = (
            solver.get_end_effector_position()
        )

        error = calculate_position_error(
            actual_position,
            target_position,
        )

        print()
        print("ROBOTWIN-AI IK ACCURACY TEST")
        print("=" * 60)

        print(
            "Initial position : "
            f"({current_position[0]:.4f}, "
            f"{current_position[1]:.4f}, "
            f"{current_position[2]:.4f})"
        )

        print(
            "Target position  : "
            f"({target_position[0]:.4f}, "
            f"{target_position[1]:.4f}, "
            f"{target_position[2]:.4f})"
        )

        print(
            "Actual position  : "
            f"({actual_position[0]:.4f}, "
            f"{actual_position[1]:.4f}, "
            f"{actual_position[2]:.4f})"
        )

        print(
            f"Position error   : {error:.6f} m"
        )

        print(
            f"Joint count      : "
            f"{len(ik_result.joint_positions)}"
        )

        if error <= 0.01:
            print("IK accuracy      : PASS")
        else:
            print("IK accuracy      : FAIL")


if __name__ == "__main__":
    main()