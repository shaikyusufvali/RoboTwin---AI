"""Motion-aware telemetry collection for RoboTwin-AI."""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Final

from robot_twin.simulation.motion_controller import (
    JointMotionController,
)
from robot_twin.simulation.robot_simulator import (
    RobotSimulator,
    SimulationConfig,
)
from robot_twin.telemetry.pybullet_adapter import (
    PyBulletRobotAdapter,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_FILE: Final[Path] = (
    Path("data") / "motion_robot_telemetry.csv"
)
DEFAULT_JOINT_INDEX: Final[int] = 2
DEFAULT_SAMPLE_INTERVAL: Final[float] = 0.1
DEFAULT_DURATION: Final[float] = 20.0

TARGET_POSITIONS: Final[tuple[float, ...]] = (
    0.8,
    -0.8,
    0.8,
    -0.8,
    0.0,
)


class MotionTelemetryCollector:
    """Collect telemetry while a simulated joint follows targets."""

    def __init__(
        self,
        simulator: RobotSimulator,
        adapter: PyBulletRobotAdapter,
        controller: JointMotionController,
        output_file: Path = DEFAULT_OUTPUT_FILE,
        joint_index: int = DEFAULT_JOINT_INDEX,
        sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
    ) -> None:
        if sample_interval <= 0:
            raise ValueError(
                "sample_interval must be greater than zero."
            )

        self._simulator = simulator
        self._adapter = adapter
        self._controller = controller
        self._output_file = output_file
        self._joint_index = joint_index
        self._sample_interval = sample_interval

    def collect(
        self,
        duration_seconds: float = DEFAULT_DURATION,
    ) -> int:
        """Collect telemetry during controlled robot motion."""

        if duration_seconds <= 0:
            raise ValueError(
                "duration_seconds must be greater than zero."
            )

        if not self._adapter.is_connected():
            raise RuntimeError(
                "Robot adapter is not connected."
            )

        self._output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = [
            "timestamp",
            "joint_index",
            "position",
            "velocity",
            "force",
            "motor_current",
            "temperature",
            "target_position",
        ]

        records_written = 0
        target_index = 0
        target_started_at = time.monotonic()

        self._controller.move_joint(
            self._joint_index,
            TARGET_POSITIONS[target_index],
        )

        start_time = time.monotonic()

        with self._output_file.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )
            writer.writeheader()

            while (
                time.monotonic() - start_time
                < duration_seconds
            ):
                self._simulator.step()

                elapsed_target_time = (
                    time.monotonic() - target_started_at
                )

                if elapsed_target_time >= 4.0:
                    target_index = (
                        target_index + 1
                    ) % len(TARGET_POSITIONS)

                    self._controller.move_joint(
                        self._joint_index,
                        TARGET_POSITIONS[target_index],
                    )

                    target_started_at = time.monotonic()

                telemetry = (
                    self._adapter.get_joint_telemetry(
                        self._joint_index
                    )
                )

                row = {
                    "timestamp": time.time(),
                    **asdict(telemetry),
                    "target_position": (
                        TARGET_POSITIONS[target_index]
                    ),
                }

                writer.writerow(row)
                records_written += 1

                LOGGER.info(
                    (
                        "Joint %d | target=%.3f | "
                        "position=%.3f | velocity=%.3f | "
                        "force=%.3f | current=%.3f | "
                        "temperature=%.2f"
                    ),
                    telemetry.joint_index,
                    TARGET_POSITIONS[target_index],
                    telemetry.position,
                    telemetry.velocity,
                    telemetry.force,
                    telemetry.motor_current,
                    telemetry.temperature,
                )

                time.sleep(self._sample_interval)

        LOGGER.info(
            "Motion telemetry collection completed. "
            "Records: %d",
            records_written,
        )

        return records_written


def main() -> None:
    """Run motion-aware telemetry collection."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    simulator = RobotSimulator(
        SimulationConfig(gui=False)
    )

    adapter = PyBulletRobotAdapter(
        simulator=simulator
    )

    controller = JointMotionController(
        simulator=simulator
    )

    collector = MotionTelemetryCollector(
        simulator=simulator,
        adapter=adapter,
        controller=controller,
    )

    try:
        adapter.connect()

        records = collector.collect(
            duration_seconds=DEFAULT_DURATION
        )

        print(
            f"Motion telemetry collection completed: "
            f"{records} records"
        )
        print(
            f"Saved to: {DEFAULT_OUTPUT_FILE}"
        )

    finally:
        adapter.disconnect()


if __name__ == "__main__":
    main()