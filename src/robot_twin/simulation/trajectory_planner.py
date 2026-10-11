"""Trajectory generation for RoboTwin-AI motion planning."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum

from robot_twin.simulation.motion_planner import (
    MotionDirection,
    MotionTarget,
)


LOGGER = logging.getLogger(__name__)


class TrajectoryType(str, Enum):
    """Types of trajectories supported by the planner."""

    JOINT = "joint"
    CARTESIAN = "cartesian"


@dataclass(frozen=True)
class TrajectoryPoint:
    """A single point in a robot trajectory."""

    step: int
    x: float
    y: float
    z: float
    joint_index: int | None = None
    joint_position: float | None = None


@dataclass(frozen=True)
class Trajectory:
    """Generated robot trajectory."""

    trajectory_type: TrajectoryType
    points: tuple[TrajectoryPoint, ...]


@dataclass(frozen=True)
class TrajectoryConfig:
    """Configuration for trajectory generation."""

    step_distance: float = 0.25
    joint_step: float = 0.10


class TrajectoryPlanner:
    """Generate controlled robot motion trajectories."""

    def __init__(
        self,
        config: TrajectoryConfig | None = None,
    ) -> None:
        self._config = config or TrajectoryConfig()

        if self._config.step_distance <= 0:
            raise ValueError(
                "step_distance must be greater than zero."
            )

        if self._config.joint_step <= 0:
            raise ValueError(
                "joint_step must be greater than zero."
            )

    def generate(
        self,
        target: MotionTarget,
    ) -> Trajectory:
        """Generate a trajectory from a motion target."""

        if target.joint_index is not None:
            return self._generate_joint_trajectory(target)

        if target.direction is not None:
            return self._generate_cartesian_trajectory(target)

        return Trajectory(
            trajectory_type=TrajectoryType.CARTESIAN,
            points=(
                TrajectoryPoint(
                    step=0,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                ),
            ),
        )

    def _generate_joint_trajectory(
        self,
        target: MotionTarget,
    ) -> Trajectory:
        """Generate a gradual joint-space trajectory."""

        if target.joint_index is None:
            raise ValueError(
                "Joint trajectory requires a joint index."
            )

        if target.joint_position is None:
            raise ValueError(
                "Joint trajectory requires a target position."
            )

        target_degrees = target.joint_position

        target_radians = math.radians(
            target_degrees
        )

        point_count = max(
            1,
            math.ceil(
                abs(target_radians)
                / self._config.joint_step
            ),
        )

        points: list[TrajectoryPoint] = []

        for step in range(1, point_count + 1):
            progress = step / point_count

            position = (
                target_radians * progress
            )

            points.append(
                TrajectoryPoint(
                    step=step,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    joint_index=target.joint_index,
                    joint_position=position,
                )
            )

        LOGGER.info(
            "Generated joint trajectory: "
            "joint=%d target=%.2f degrees "
            "(%.3f radians) points=%d",
            target.joint_index,
            target_degrees,
            target_radians,
            len(points),
        )

        return Trajectory(
            trajectory_type=TrajectoryType.JOINT,
            points=tuple(points),
        )

    def _generate_cartesian_trajectory(
        self,
        target: MotionTarget,
    ) -> Trajectory:
        """Generate a gradual Cartesian trajectory."""

        if target.direction is None:
            raise ValueError(
                "Cartesian trajectory requires a direction."
            )

        if target.distance is None:
            raise ValueError(
                "Cartesian trajectory requires a distance."
            )

        distance = target.distance

        if distance <= 0:
            raise ValueError(
                "Trajectory distance must be greater than zero."
            )

        point_count = max(
            1,
            math.ceil(
                distance
                / self._config.step_distance
            ),
        )

        points: list[TrajectoryPoint] = []

        for step in range(1, point_count + 1):
            progress = step / point_count
            travelled = distance * progress

            x, y, z = self._calculate_position(
                target.direction,
                travelled,
            )

            points.append(
                TrajectoryPoint(
                    step=step,
                    x=x,
                    y=y,
                    z=z,
                )
            )

        LOGGER.info(
            "Generated Cartesian trajectory: "
            "direction=%s distance=%.2f m points=%d",
            target.direction.value,
            distance,
            len(points),
        )

        return Trajectory(
            trajectory_type=TrajectoryType.CARTESIAN,
            points=tuple(points),
        )

    @staticmethod
    def _calculate_position(
        direction: MotionDirection,
        distance: float,
    ) -> tuple[float, float, float]:
        """Calculate Cartesian position for a direction."""

        if direction == MotionDirection.FORWARD:
            return 0.0, distance, 0.0

        if direction == MotionDirection.BACKWARD:
            return 0.0, -distance, 0.0

        if direction == MotionDirection.LEFT:
            return -distance, 0.0, 0.0

        if direction == MotionDirection.RIGHT:
            return distance, 0.0, 0.0

        raise ValueError(
            f"Unsupported motion direction: {direction}"
        )


def main() -> None:
    """Demonstrate trajectory generation."""

    from robot_twin.command.command_interpreter import (
        CommandInterpreter,
    )
    from robot_twin.simulation.motion_planner import (
        MotionPlanner,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    interpreter = CommandInterpreter()
    motion_planner = MotionPlanner()
    trajectory_planner = TrajectoryPlanner()

    commands = (
        "Move joint 2 to 30 degrees",
        "Move forward 2 meters",
        "Go left 1 meter",
    )

    print()
    print("ROBOTWIN-AI TRAJECTORY PLANNER")
    print("=" * 60)

    for text in commands:
        command = interpreter.interpret(text)
        target = motion_planner.plan(command)
        trajectory = trajectory_planner.generate(
            target
        )

        print()
        print(f"Human command   : {text}")
        print(
            f"Trajectory type : "
            f"{trajectory.trajectory_type.value}"
        )
        print(
            f"Trajectory points: "
            f"{len(trajectory.points)}"
        )

        if trajectory.points:
            first = trajectory.points[0]
            last = trajectory.points[-1]

            print(f"First point     : {first}")
            print(f"Last point      : {last}")

            if last.joint_position is not None:
                print(
                    f"Final position  : "
                    f"{last.joint_position:.3f} rad"
                )


if __name__ == "__main__":
    main()