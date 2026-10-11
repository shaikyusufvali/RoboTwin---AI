"""Prepare affected-joint data for RoboTwin-AI fault diagnosis."""

from __future__ import annotations

import csv
import logging
from pathlib import Path


LOGGER = logging.getLogger(__name__)

INPUT_FILE = Path("data") / "robot_diagnostics_dataset_v3.csv"
OUTPUT_FILE = Path("data") / "fault_joint_dataset.csv"

FEATURE_COLUMNS = [
    "position",
    "position_error",
    "velocity",
    "acceleration",
    "force",
    "motor_current",
    "temperature",
]

REQUIRED_COLUMNS = {
    "episode_id",
    "joint_index",
    "fault_joint_index",
    "fault_type",
    *FEATURE_COLUMNS,
}


def prepare_dataset(
    input_path: Path,
    output_path: Path,
) -> None:
    """Keep telemetry records belonging to the affected joint."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    selected_rows: list[dict[str, str]] = []

    with input_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        available_columns = set(
            reader.fieldnames or []
        )

        missing_columns = (
            REQUIRED_COLUMNS - available_columns
        )

        if missing_columns:
            raise ValueError(
                "Missing dataset columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:
            joint_index = int(
                row["joint_index"]
            )

            fault_joint_index = int(
                row["fault_joint_index"]
            )

            if joint_index != fault_joint_index:
                continue

            selected_rows.append(row)

    if not selected_rows:
        raise ValueError(
            "No affected-joint records were found."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        selected_rows[0].keys()
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(selected_rows)

    LOGGER.info(
        "Saved affected-joint dataset: %s",
        output_path,
    )

    print()
    print("=" * 72)
    print("ROBOTWIN-AI AFFECTED JOINT DATASET")
    print("=" * 72)
    print(f"Input records: {sum(1 for _ in _read_rows(input_path))}")
    print(f"Affected-joint records: {len(selected_rows)}")

    print()
    print("Class distribution:")

    class_counts: dict[str, int] = {}

    for row in selected_rows:
        fault_type = row["fault_type"]

        class_counts[fault_type] = (
            class_counts.get(fault_type, 0) + 1
        )

    for fault_type in sorted(class_counts):
        print(
            f"{fault_type}: "
            f"{class_counts[fault_type]}"
        )


def _read_rows(
    path: Path,
):
    """Read dataset rows for counting purposes."""

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        yield from csv.DictReader(file)


def main() -> None:
    """Run the affected-joint dataset preparation."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    prepare_dataset(
        INPUT_FILE,
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()