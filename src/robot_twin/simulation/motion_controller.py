"""Robot joint motion control for RoboTwin-AI."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Final

import pybullet as p

from robot_twin.simulation.robot_simulator import RobotSimulator


LOGGER = logging.getLogger(__name__)


DEFAULT_POSITION_GAIN: Final[float] = 0.8
DEFAULT_VELOCITY_GAIN: Final[float] = 0.3
DEFAULT_MAX_FORCE: Final[float] = 120.0
DEFAULT_POSITION_TOLERANCE: Final[float] = 0.01
DEFAULT_MAX_SIMULATION_STEPS: Final[int] = 1500
DEFAULT_SETTLE_STEPS: Final[int] = 10


@dataclass(frozen=True)
class MotionConfig:
    """Configuration for joint motion control."""

    position_gain: float = DEFAULT_POSITION_GAIN
    velocity_gain: float = DEFAULT_VELOCITY_GAIN
    max_force: float = DEFAULT_MAX_FORCE
    position_tolerance: float = DEFAULT_POSITION_TOLERANCE
    max_simulation_steps: int = DEFAULT_MAX_SIMULATION_STEPS
    settle_steps: int = DEFAULT_SETTLE_STEPS


class JointMotionController:
    """Control robot joints using PyBullet position control."""

    def __init__(
        self,
        simulator: RobotSimulator,
        config: MotionConfig | None = None,
    ) -> None:
        self._simulator = simulator
        self._config = config or MotionConfig()

        self._validate_config()

    def command_joint(
        self,
        joint_index: int,
        target_position: float,
    ) -> None:
        """Command one joint without waiting for convergence."""

        self._validate_joint_index(joint_index)
        self._validate_position(target_position)

        self._command_joint(
            joint_index=joint_index,
            target_position=target_position,
        )

        LOGGER.info(
            "Joint %d commanded to %.3f rad.",
            joint_index,
            target_position,
        )

    def command_joints(
        self,
        target_positions: dict[int, float],
    ) -> None:
        """Command multiple joints without waiting for convergence."""

        if not target_positions:
            raise ValueError(
                "target_positions cannot be empty."
            )

        for joint_index, target_position in target_positions.items():
            self._validate_joint_index(joint_index)
            self._validate_position(target_position)

        for joint_index, target_position in target_positions.items():
            self._command_joint(
                joint_index=joint_index,
                target_position=target_position,
            )

            LOGGER.info(
                "Joint %d commanded to %.3f rad.",
                joint_index,
                target_position,
            )

    def move_joint(
        self,
        joint_index: int,
        target_position: float,
    ) -> bool:
        """Move one joint to the requested target position."""

        self.command_joint(
            joint_index=joint_index,
            target_position=target_position,
        )

        return self._wait_for_joint_targets(
            {joint_index: target_position}
        )

    def move_joints(
        self,
        target_positions: dict[int, float],
    ) -> bool:
        """Move multiple joints simultaneously."""

        self.command_joints(target_positions)

        return self._wait_for_joint_targets(
            target_positions
        )

    def stop_joint(self, joint_index: int) -> None:
        """Stop a single joint at its current position."""

        self._validate_joint_index(joint_index)

        current_position = self.get_joint_position(
            joint_index
        )

        p.setJointMotorControl2(
            bodyUniqueId=self._simulator.robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=current_position,
            positionGain=self._config.position_gain,
            velocityGain=self._config.velocity_gain,
            force=self._config.max_force,
            physicsClientId=self._simulator.client_id,
        )

        LOGGER.info(
            "Joint %d motion stopped.",
            joint_index,
        )

    def stop_joints(self) -> None:
        """Stop all revolute joints."""

        for joint_index in self.get_joint_positions():
            self.stop_joint(joint_index)

    def get_joint_position(
        self,
        joint_index: int,
    ) -> float:
        """Return the current position of one joint."""

        self._validate_joint_index(joint_index)

        state = p.getJointState(
            self._simulator.robot_id,
            joint_index,
            physicsClientId=self._simulator.client_id,
        )

        return float(state[0])

    def get_joint_positions(self) -> dict[int, float]:
        """Return current positions of all revolute joints."""

        joint_count = p.getNumJoints(
            self._simulator.robot_id,
            physicsClientId=self._simulator.client_id,
        )

        positions: dict[int, float] = {}

        for joint_index in range(joint_count):
            joint_info = p.getJointInfo(
                self._simulator.robot_id,
                joint_index,
                physicsClientId=self._simulator.client_id,
            )

            if joint_info[2] == p.JOINT_REVOLUTE:
                positions[joint_index] = (
                    self.get_joint_position(joint_index)
                )

        return positions

    def _command_joint(
        self,
        joint_index: int,
        target_position: float,
    ) -> None:
        """Send a position-control command to one joint."""

        p.setJointMotorControl2(
            bodyUniqueId=self._simulator.robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_position,
            positionGain=self._config.position_gain,
            velocityGain=self._config.velocity_gain,
            force=self._config.max_force,
            physicsClientId=self._simulator.client_id,
        )

    def _wait_for_joint_targets(
        self,
        target_positions: dict[int, float],
    ) -> bool:
        """Run simulation until targets converge or timeout."""

        consecutive_success_steps = 0

        for _ in range(
            self._config.max_simulation_steps
        ):
            self._simulator.step()

            if self._all_targets_reached(target_positions):
                consecutive_success_steps += 1

                if (
                    consecutive_success_steps
                    >= self._config.settle_steps
                ):
                    LOGGER.info(
                        "All %d joints reached their targets "
                        "and remained stable.",
                        len(target_positions),
                    )
                    return True
            else:
                consecutive_success_steps = 0

        LOGGER.warning(
            "Joint motion did not converge within "
            "%d simulation steps.",
            self._config.max_simulation_steps,
        )

        for (
            joint_index,
            target_position,
        ) in target_positions.items():
            actual_position = self.get_joint_position(
                joint_index
            )

            error = abs(
                target_position - actual_position
            )

            LOGGER.warning(
                "Joint %d final error: %.4f rad "
                "(target=%.4f, actual=%.4f)",
                joint_index,
                error,
                target_position,
                actual_position,
            )

        return False

    def _all_targets_reached(
        self,
        target_positions: dict[int, float],
    ) -> bool:
        """Check whether all target positions are within tolerance."""

        for (
            joint_index,
            target_position,
        ) in target_positions.items():
            actual_position = self.get_joint_position(
                joint_index
            )

            if (
                abs(target_position - actual_position)
                > self._config.position_tolerance
            ):
                return False

        return True

    def _validate_joint_index(
        self,
        joint_index: int,
    ) -> None:
        """Validate a robot joint index."""

        if not isinstance(joint_index, int):
            raise TypeError(
                "joint_index must be an integer."
            )

        joint_count = p.getNumJoints(
            self._simulator.robot_id,
            physicsClientId=self._simulator.client_id,
        )

        if not 0 <= joint_index < joint_count:
            raise IndexError(
                f"Invalid joint index: {joint_index}."
            )

        joint_type = p.getJointInfo(
            self._simulator.robot_id,
            joint_index,
            physicsClientId=self._simulator.client_id,
        )[2]

        if joint_type != p.JOINT_REVOLUTE:
            raise ValueError(
                f"Joint {joint_index} is not a revolute joint."
            )

    @staticmethod
    def _validate_position(
        target_position: float,
    ) -> None:
        """Validate a joint target position."""

        if not isinstance(
            target_position,
            (int, float),
        ):
            raise TypeError(
                "target_position must be numeric."
            )

        if not math.isfinite(target_position):
            raise ValueError(
                "target_position must be finite."
            )

    def _validate_config(self) -> None:
        """Validate motion controller configuration."""

        if self._config.position_gain <= 0:
            raise ValueError(
                "position_gain must be greater than zero."
            )

        if self._config.velocity_gain < 0:
            raise ValueError(
                "velocity_gain cannot be negative."
            )

        if self._config.max_force <= 0:
            raise ValueError(
                "max_force must be greater than zero."
            )

        if self._config.position_tolerance <= 0:
            raise ValueError(
                "position_tolerance must be greater than zero."
            )

        if self._config.max_simulation_steps <= 0:
            raise ValueError(
                "max_simulation_steps must be greater than zero."
            )

        if self._config.settle_steps <= 0:
            raise ValueError(
                "settle_steps must be greater than zero."
            )

        if (
            self._config.settle_steps
            > self._config.max_simulation_steps
        ):
            raise ValueError(
                "settle_steps cannot exceed "
                "max_simulation_steps."
            )


def main() -> None:
    """Run a basic joint motion demonstration."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    with RobotSimulator() as simulator:
        controller = JointMotionController(simulator)

        target_position = 0.5

        success = controller.move_joint(
            joint_index=1,
            target_position=target_position,
        )

        final_position = controller.get_joint_position(1)

        print()
        print("ROBOTWIN-AI JOINT MOTION")
        print("=" * 60)
        print(
            f"Target position : "
            f"{target_position:.4f} rad"
        )
        print(
            f"Final position  : "
            f"{final_position:.4f} rad"
        )
        print(
            f"Position error  : "
            f"{abs(target_position - final_position):.4f} rad"
        )
        print(
            f"Motion success  : {success}"
        )


if __name__ == "__main__":
    main()