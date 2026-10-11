"""Build a healthy telemetry baseline for RoboTwin-AI."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Final


LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT_FILE: Final[Path] = (
    Path("data") / "motion_robot_telemetry.csv"
)

DEFAULT_OUTPUT_FILE: Final[Path] = (
    Path("data") / "healthy_baseline.json"
)

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "joint_index",
    "position",
    "velocity",
    "force",
    "motor_current",
    "temperature",
    "target_position",
)


@dataclass(frozen=True)
class FeatureStatistics:
    """Statistical description of one healthy feature."""

    mean: float
    standard_deviation: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class JointBaseline:
    """Healthy baseline for one robot joint."""

    joint_index: int
    samples: int
    position: FeatureStatistics
    velocity: FeatureStatistics
    force: FeatureStatistics
    motor_current: FeatureStatistics
    temperature: FeatureStatistics
    position_error: FeatureStatistics


class BaselineBuilder:
    """Build healthy-operation statistics from telemetry."""

    def build(
        self,
        input_file: Path,
        output_file: Path,
    ) -> list[JointBaseline]:
        """Build and save a healthy baseline."""

        if not input_file.exists():
            raise FileNotFoundError(
                f"Telemetry file not found: {input_file}"
            )

        grouped_data: dict[int, dict[str, list[float]]] = {}

        with input_file.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as source:
            reader = csv.DictReader(source)

            if reader.fieldnames is None:
                raise ValueError(
                    "Telemetry file has no header."
                )

            missing_columns = (
                set(REQUIRED_COLUMNS)
                - set(reader.fieldnames)
            )

            if missing_columns:
                raise ValueError(
                    "Missing telemetry columns: "
                    + ", ".join(sorted(missing_columns))
                )

            for row in reader:
                joint_index = int(row["joint_index"])

                if joint_index not in grouped_data:
                    grouped_data[joint_index] = {
                        "position": [],
                        "velocity": [],
                        "force": [],
                        "motor_current": [],
                        "temperature": [],
                        "position_error": [],
                    }

                position = float(row["position"])
                target_position = float(
                    row["target_position"]
                )

                grouped_data[joint_index][
                    "position"
                ].append(position)

                grouped_data[joint_index][
                    "velocity"
                ].append(
                    float(row["velocity"])
                )

                grouped_data[joint_index][
                    "force"
                ].append(
                    float(row["force"])
                )

                grouped_data[joint_index][
                    "motor_current"
                ].append(
                    float(row["motor_current"])
                )

                grouped_data[joint_index][
                    "temperature"
                ].append(
                    float(row["temperature"])
                )

                grouped_data[joint_index][
                    "position_error"
                ].append(
                    target_position - position
                )

        baselines = [
            self._build_joint_baseline(
                joint_index=joint_index,
                values=values,
            )
            for joint_index, values
            in sorted(grouped_data.items())
        ]

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._save_baseline(
            baselines=baselines,
            output_file=output_file,
        )

        LOGGER.info(
            "Healthy baseline created for %d joints.",
            len(baselines),
        )

        return baselines

    @staticmethod
    def _build_joint_baseline(
        joint_index: int,
        values: dict[str, list[float]],
    ) -> JointBaseline:
        return JointBaseline(
            joint_index=joint_index,
            samples=len(values["position"]),
            position=BaselineBuilder._calculate_statistics(
                values["position"]
            ),
            velocity=BaselineBuilder._calculate_statistics(
                values["velocity"]
            ),
            force=BaselineBuilder._calculate_statistics(
                values["force"]
            ),
            motor_current=BaselineBuilder._calculate_statistics(
                values["motor_current"]
            ),
            temperature=BaselineBuilder._calculate_statistics(
                values["temperature"]
            ),
            position_error=BaselineBuilder._calculate_statistics(
                values["position_error"]
            ),
        )

    @staticmethod
    def _calculate_statistics(
        values: list[float],
    ) -> FeatureStatistics:
        if not values:
            raise ValueError(
                "Cannot calculate statistics from empty data."
            )

        standard_deviation = (
            stdev(values)
            if len(values) > 1
            else 0.0
        )

        return FeatureStatistics(
            mean=mean(values),
            standard_deviation=standard_deviation,
            minimum=min(values),
            maximum=max(values),
        )

    @staticmethod
    def _save_baseline(
        baselines: list[JointBaseline],
        output_file: Path,
    ) -> None:
        payload = {
            "baseline_type": "healthy_telemetry",
            "version": "1.0",
            "joints": [
                asdict(baseline)
                for baseline in baselines
            ],
        }

        with output_file.open(
            "w",
            encoding="utf-8",
        ) as destination:
            json.dump(
                payload,
                destination,
                indent=4,
            )


def main() -> None:
    """Build the healthy robot baseline."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    builder = BaselineBuilder()

    baselines = builder.build(
        input_file=DEFAULT_INPUT_FILE,
        output_file=DEFAULT_OUTPUT_FILE,
    )

    print()
    print("ROBOTWIN-AI HEALTHY BASELINE")
    print("=" * 45)
    print(f"Joints analyzed : {len(baselines)}")
    print(
        f"Output          : {DEFAULT_OUTPUT_FILE}"
    )

    for baseline in baselines:
        print(
            f"Joint {baseline.joint_index}: "
            f"{baseline.samples} samples"
        )


if __name__ == "__main__":
    main()