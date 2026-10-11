"""Generate a healthy motion baseline for RoboTwin-AI."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Final

from robot_twin.simulation.motion_controller import (
    JointMotionController,
)
from robot_twin.simulation.robot_simulator import (
    RobotSimulator,
)
from robot_twin.telemetry.pybullet_adapter import (
    PyBulletRobotAdapter,
)


LOGGER = logging.getLogger(__name__)


DEFAULT_OUTPUT_FILE: Final[Path] = (
    Path("data") / "healthy_baseline.json"
)

DEFAULT_SAMPLE_COUNT: Final[int] = 1200
DEFAULT_SETTLE_STEPS: Final[int] = 120

DEFAULT_TARGET_POSITIONS: Final[tuple[float, ...]] = (
    0.25,
    -0.20,
    0.20,
    -0.15,
    0.15,
    -0.10,
    0.10,
)

DEFAULT_RETURN_POSITIONS: Final[tuple[float, ...]] = (
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)


@dataclass(frozen=True)
class FeatureStatistics:
    """Statistical description of one healthy telemetry feature."""

    mean: float
    std: float


@dataclass(frozen=True)
class JointBaseline:
    """Healthy baseline statistics for one robot joint."""

    joint_index: int
    sample_count: int
    position: FeatureStatistics
    velocity: FeatureStatistics
    force: FeatureStatistics
    motor_current: FeatureStatistics
    temperature: FeatureStatistics
    position_error: FeatureStatistics


class HealthyBaselineGenerator:
    """Collect healthy motion telemetry and build a baseline."""

    def __init__(
        self,
        adapter: PyBulletRobotAdapter,
        motion_controller: JointMotionController,
        sample_count: int = DEFAULT_SAMPLE_COUNT,
        settle_steps: int = DEFAULT_SETTLE_STEPS,
    ) -> None:
        self._adapter = adapter
        self._motion_controller = motion_controller
        self._sample_count = sample_count
        self._settle_steps = settle_steps

        self._validate_config()

    def generate(
        self,
        output_file: Path = DEFAULT_OUTPUT_FILE,
    ) -> list[JointBaseline]:
        """Collect healthy motion samples and save the baseline."""

        LOGGER.info("Starting healthy baseline generation.")

        joint_indices = tuple(
            self._motion_controller.get_joint_positions()
        )

        if not joint_indices:
            raise RuntimeError(
                "No revolute joints were found."
            )

        samples = {
            joint_index: {
                "position": [],
                "velocity": [],
                "force": [],
                "motor_current": [],
                "temperature": [],
                "position_error": [],
            }
            for joint_index in joint_indices
        }

        target_positions = self._build_target_positions(
            joint_indices
        )

        return_positions = self._build_return_positions(
            joint_indices
        )

        LOGGER.info(
            "Healthy motion targets: %s",
            target_positions,
        )

        self._collect_motion_cycle(
            target_positions=target_positions,
            samples=samples,
        )

        LOGGER.info(
            "Returning robot to neutral position."
        )

        self._collect_motion_cycle(
            target_positions=return_positions,
            samples=samples,
        )

        baselines = self._build_baselines(
            samples=samples,
        )

        self._save_baseline(
            baselines=baselines,
            output_file=output_file,
        )

        LOGGER.info(
            "Healthy baseline generation completed."
        )

        return baselines

    def _collect_motion_cycle(
        self,
        target_positions: dict[int, float],
        samples: dict[int, dict[str, list[float]]],
    ) -> None:
        """Command motion and collect telemetry during movement."""

        self._motion_controller.command_joints(
            target_positions
        )

        for sample_index in range(self._sample_count):
            self._simulator_step()

            for joint_index in samples:
                telemetry = (
                    self._adapter.get_joint_telemetry(
                        joint_index=joint_index
                    )
                )

                target_position = target_positions[
                    joint_index
                ]

                samples[joint_index]["position"].append(
                    telemetry.position
                )

                samples[joint_index]["velocity"].append(
                    telemetry.velocity
                )

                samples[joint_index]["force"].append(
                    telemetry.force
                )

                samples[joint_index]["motor_current"].append(
                    telemetry.motor_current
                )

                samples[joint_index]["temperature"].append(
                    telemetry.temperature
                )

                samples[joint_index]["position_error"].append(
                    telemetry.position - target_position
                )

            if (
                sample_index + 1
            ) % 200 == 0:
                LOGGER.info(
                    "Collected %d/%d telemetry samples.",
                    sample_index + 1,
                    self._sample_count,
                )

        for _ in range(self._settle_steps):
            self._simulator_step()

    def _simulator_step(self) -> None:
        """Advance the underlying simulation."""

        self._adapter._simulator.step()

    def _build_baselines(
        self,
        samples: dict[int, dict[str, list[float]]],
    ) -> list[JointBaseline]:
        """Convert raw telemetry samples into statistics."""

        baselines: list[JointBaseline] = []

        for joint_index, joint_samples in samples.items():
            baseline = JointBaseline(
                joint_index=joint_index,
                sample_count=len(
                    joint_samples["position"]
                ),
                position=self._statistics(
                    joint_samples["position"]
                ),
                velocity=self._statistics(
                    joint_samples["velocity"]
                ),
                force=self._statistics(
                    joint_samples["force"]
                ),
                motor_current=self._statistics(
                    joint_samples["motor_current"]
                ),
                temperature=self._statistics(
                    joint_samples["temperature"]
                ),
                position_error=self._statistics(
                    joint_samples["position_error"]
                ),
            )

            baselines.append(baseline)

        return baselines

    @staticmethod
    def _statistics(
        values: list[float],
    ) -> FeatureStatistics:
        """Calculate mean and population standard deviation."""

        if not values:
            raise ValueError(
                "Cannot calculate statistics from empty data."
            )

        standard_deviation = pstdev(values)

        if standard_deviation == 0:
            standard_deviation = 1e-6

        return FeatureStatistics(
            mean=float(mean(values)),
            std=float(standard_deviation),
        )

    @staticmethod
    def _build_target_positions(
        joint_indices: tuple[int, ...],
    ) -> dict[int, float]:
        """Build safe baseline motion targets."""

        targets: dict[int, float] = {}

        for joint_index in joint_indices:
            if joint_index >= len(
                DEFAULT_TARGET_POSITIONS
            ):
                raise ValueError(
                    "Target configuration does not contain "
                    f"joint {joint_index}."
                )

            targets[joint_index] = (
                DEFAULT_TARGET_POSITIONS[joint_index]
            )

        return targets

    @staticmethod
    def _build_return_positions(
        joint_indices: tuple[int, ...],
    ) -> dict[int, float]:
        """Build neutral return positions."""

        positions: dict[int, float] = {}

        for joint_index in joint_indices:
            if joint_index >= len(
                DEFAULT_RETURN_POSITIONS
            ):
                raise ValueError(
                    "Return configuration does not contain "
                    f"joint {joint_index}."
                )

            positions[joint_index] = (
                DEFAULT_RETURN_POSITIONS[joint_index]
            )

        return positions

    def _save_baseline(
        self,
        baselines: list[JointBaseline],
        output_file: Path,
    ) -> None:
        """Save baseline statistics as JSON."""

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "version": "3.0",
            "description": (
                "Healthy KUKA motion telemetry baseline "
                "generated from PyBullet simulation."
            ),
            "sample_count_per_joint": self._sample_count * 2,
            "joints": [
                asdict(baseline)
                for baseline in baselines
            ],
        }

        with output_file.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=2,
            )

        LOGGER.info(
            "Healthy baseline saved to %s",
            output_file,
        )

    def _validate_config(self) -> None:
        """Validate generator configuration."""

        if self._sample_count <= 0:
            raise ValueError(
                "sample_count must be greater than zero."
            )

        if self._settle_steps < 0:
            raise ValueError(
                "settle_steps cannot be negative."
            )


def print_summary(
    baselines: list[JointBaseline],
) -> None:
    """Print a compact baseline summary."""

    print()
    print("ROBOTWIN-AI HEALTHY BASELINE")
    print("=" * 70)
    print(
        f"{'Joint':<8}"
        f"{'Samples':<10}"
        f"{'Velocity Mean':<18}"
        f"{'Current Mean':<18}"
        f"{'Temperature Mean':<18}"
    )
    print("-" * 70)

    for baseline in baselines:
        print(
            f"{baseline.joint_index:<8}"
            f"{baseline.sample_count:<10}"
            f"{baseline.velocity.mean:<18.4f}"
            f"{baseline.motor_current.mean:<18.4f}"
            f"{baseline.temperature.mean:<18.4f}"
        )

    print("=" * 70)
    print(
        "Healthy motion baseline generated successfully."
    )


def main() -> None:
    """Run healthy baseline generation."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    simulator = RobotSimulator()
    adapter = PyBulletRobotAdapter(simulator)

    try:
        adapter.connect()

        motion_controller = JointMotionController(
            simulator=simulator,
        )

        generator = HealthyBaselineGenerator(
            adapter=adapter,
            motion_controller=motion_controller,
        )

        baselines = generator.generate()

        print_summary(baselines)

    finally:
        adapter.disconnect()


if __name__ == "__main__":
    main()