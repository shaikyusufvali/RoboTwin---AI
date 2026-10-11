"""Live robot telemetry streaming for RoboTwin-AI."""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import asdict
from pathlib import Path

from robot_twin.telemetry.pybullet_adapter import PyBulletRobotAdapter


LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_FILE = Path("data") / "live_robot_telemetry.csv"
DEFAULT_JOINT_INDEX = 2
DEFAULT_SAMPLE_INTERVAL = 0.1
DEFAULT_DURATION = 10.0


class TelemetryStream:
    """Collect live telemetry from a robot adapter."""

    def __init__(
        self,
        adapter: PyBulletRobotAdapter,
        output_file: Path = DEFAULT_OUTPUT_FILE,
        joint_index: int = DEFAULT_JOINT_INDEX,
        sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
    ) -> None:
        if sample_interval <= 0:
            raise ValueError(
                "sample_interval must be greater than zero."
            )

        self._adapter = adapter
        self._output_file = output_file
        self._joint_index = joint_index
        self._sample_interval = sample_interval

    def collect(
        self,
        duration_seconds: float = DEFAULT_DURATION,
    ) -> int:
        """Collect telemetry for the requested duration."""

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
        ]

        records_written = 0
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
                self._adapter._simulator.step()

                telemetry = (
                    self._adapter.get_joint_telemetry(
                        self._joint_index
                    )
                )

                row = {
                    "timestamp": time.time(),
                    **asdict(telemetry),
                }

                writer.writerow(row)
                records_written += 1

                LOGGER.info(
                    "Joint %d | position=%.4f | "
                    "velocity=%.4f | force=%.4f | "
                    "current=%.4f | temperature=%.2f",
                    telemetry.joint_index,
                    telemetry.position,
                    telemetry.velocity,
                    telemetry.force,
                    telemetry.motor_current,
                    telemetry.temperature,
                )

                time.sleep(self._sample_interval)

        LOGGER.info(
            "Telemetry collection completed. "
            "Records written: %d",
            records_written,
        )

        return records_written


def main() -> None:
    """Run the live telemetry collection."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    from robot_twin.simulation.robot_simulator import (
        RobotSimulator,
        SimulationConfig,
    )

    simulator = RobotSimulator(
        SimulationConfig(gui=False)
    )

    adapter = PyBulletRobotAdapter(simulator)

    try:
        adapter.connect()

        stream = TelemetryStream(
            adapter=adapter,
            output_file=DEFAULT_OUTPUT_FILE,
            joint_index=DEFAULT_JOINT_INDEX,
            sample_interval=DEFAULT_SAMPLE_INTERVAL,
        )

        records = stream.collect(
            duration_seconds=DEFAULT_DURATION
        )

        print(
            f"Telemetry collection completed: "
            f"{records} records"
        )
        print(
            f"Saved to: {DEFAULT_OUTPUT_FILE}"
        )

    finally:
        adapter.disconnect()


if __name__ == "__main__":
    main()