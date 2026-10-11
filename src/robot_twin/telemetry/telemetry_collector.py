"""Telemetry collection utilities for RoboTwin-AI."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from robot_twin.simulation.robot_simulator import RobotSimulator


LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_FILE: Final[Path] = Path("data") / "robot_telemetry.csv"


@dataclass(frozen=True)
class TelemetryRecord:
    """Snapshot of a single robot joint."""

    timestamp: str
    joint_index: int
    position: float
    velocity: float
    force: float


class TelemetryCollector:
    """Collect and persist robot telemetry."""

    def __init__(self, output_file: Path = DEFAULT_OUTPUT_FILE) -> None:
        self.output_file = output_file

    def collect(
        self,
        simulator: RobotSimulator,
        duration_seconds: float,
        sample_interval: float = 0.1,
    ) -> list[TelemetryRecord]:
        """Collect joint telemetry for the requested duration."""
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be greater than zero.")

        if sample_interval <= 0:
            raise ValueError("sample_interval must be greater than zero.")

        records: list[TelemetryRecord] = []
        elapsed = 0.0

        while elapsed < duration_seconds:
            for state in simulator.get_joint_states():
                records.append(
                    TelemetryRecord(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        joint_index=int(state["joint_index"]),
                        position=float(state["position"]),
                        velocity=float(state["velocity"]),
                        force=float(state["force"]),
                    )
                )

            simulator.step()

            elapsed += sample_interval

        return records

    def save(self, records: list[TelemetryRecord]) -> None:
        """Save telemetry records to a CSV file."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        with self.output_file.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "timestamp",
                    "joint_index",
                    "position",
                    "velocity",
                    "force",
                ],
            )

            writer.writeheader()

            for record in records:
                writer.writerow(
                    {
                        "timestamp": record.timestamp,
                        "joint_index": record.joint_index,
                        "position": record.position,
                        "velocity": record.velocity,
                        "force": record.force,
                    }
                )

        LOGGER.info(
            "Saved %d telemetry records to %s",
            len(records),
            self.output_file,
        )


def main() -> None:
    """Run a telemetry collection session."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    simulator = RobotSimulator()

    try:
        simulator.connect()
        simulator.load_world()
        simulator.load_robot()

        collector = TelemetryCollector()

        records = collector.collect(
            simulator=simulator,
            duration_seconds=30.0,
            sample_interval=0.1,
        )

        collector.save(records)

    finally:
        simulator.close()


if __name__ == "__main__":
    main()
