"""Feature engineering for RoboTwin-AI fault diagnosis."""

from __future__ import annotations

import csv
import logging
from pathlib import Path


LOGGER = logging.getLogger(__name__)

INPUT_FILE = Path("data") / "robot_diagnostics_dataset.csv"
OUTPUT_FILE = Path("data") / "robot_diagnostics_features.csv"

ENGINEERED_FEATURES = (
    "abs_position_error",
    "abs_velocity",
    "abs_acceleration",
    "abs_force",
    "abs_motor_current",
    "temperature_current_ratio",
    "force_current_ratio",
    "error_velocity_interaction",
)


def load_rows(path: Path) -> list[dict[str, str]]:
    """Load diagnostic telemetry records from a CSV file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise ValueError(
            f"Dataset is empty: {path}"
        )

    return rows


def add_engineered_features(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Create diagnostic features from raw telemetry."""

    engineered_rows: list[dict[str, str]] = []

    for row in rows:
        position_error = float(row["position_error"])
        velocity = float(row["velocity"])
        acceleration = float(row["acceleration"])
        force = float(row["force"])
        motor_current = float(row["motor_current"])
        temperature = float(row["temperature"])

        absolute_force = abs(force)
        absolute_current = abs(motor_current)

        new_row = dict(row)

        new_row["abs_position_error"] = str(
            abs(position_error)
        )

        new_row["abs_velocity"] = str(
            abs(velocity)
        )

        new_row["abs_acceleration"] = str(
            abs(acceleration)
        )

        new_row["abs_force"] = str(
            absolute_force
        )

        new_row["abs_motor_current"] = str(
            absolute_current
        )

        new_row["temperature_current_ratio"] = str(
            temperature / max(absolute_current, 1e-6)
        )

        new_row["force_current_ratio"] = str(
            absolute_force / max(absolute_current, 1e-6)
        )

        new_row["error_velocity_interaction"] = str(
            abs(position_error) * abs(velocity)
        )

        engineered_rows.append(new_row)

    return engineered_rows


def save_rows(
    rows: list[dict[str, str]],
    path: Path,
) -> None:
    """Save engineered telemetry records to CSV."""

    if not rows:
        raise ValueError(
            "Cannot save an empty dataset."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def print_feature_summary(
    rows: list[dict[str, str]],
) -> None:
    """Print the generated feature summary."""

    print()
    print("=" * 72)
    print("ROBOTWIN-AI FEATURE ENGINEERING")
    print("=" * 72)

    print(f"Records      : {len(rows)}")
    print(f"Total columns: {len(rows[0])}")

    print()
    print("Generated features:")

    for feature in ENGINEERED_FEATURES:
        print(f"  - {feature}")


def main() -> None:
    """Run the feature engineering pipeline."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    rows = load_rows(INPUT_FILE)

    LOGGER.info(
        "Loaded %d diagnostic records.",
        len(rows),
    )

    engineered_rows = add_engineered_features(rows)

    save_rows(
        engineered_rows,
        OUTPUT_FILE,
    )

    LOGGER.info(
        "Feature dataset saved to: %s",
        OUTPUT_FILE,
    )

    print_feature_summary(engineered_rows)


if __name__ == "__main__":
    main()