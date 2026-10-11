
from __future__ import annotations

import re
from typing import Any

import pandas as pd


# ============================================================
# Runtime Output Parser
# Parses actual output from RoboTwin-AI diagnostic runtime.
# Missing values remain None; no sensor values are fabricated.
# ============================================================

FIELD_PATTERNS = {
    "ground_truth_fault": [
        r"Ground truth fault\s*:\s*(.+)",
        r"Ground truth\s*:\s*(.+)",
    ],
    "telemetry_fault": [
        r"Telemetry diagnosis\s*:\s*(.+)",
        r"Telemetry fault\s*:\s*(.+)",
    ],
    "multimodal_fault": [
        r"Multimodal diagnosis\s*:\s*(.+)",
        r"Multimodal fault\s*:\s*(.+)",
    ],
    "temporal_fault": [
        r"Temporal diagnosis\s*:\s*(.+)",
        r"Diagnosed fault\s*:\s*(.+)",
        r"Final diagnosis\s*:\s*(.+)",
    ],
    "telemetry_confidence": [
        r"Telemetry confidence\s*:\s*([0-9.eE+-]+)",
    ],
    "multimodal_confidence": [
        r"Multimodal confidence\s*:\s*([0-9.eE+-]+)",
    ],
    "temporal_confidence": [
        r"Temporal confidence\s*:\s*([0-9.eE+-]+)",
        r"Confidence\s*:\s*([0-9.eE+-]+)",
    ],
    "temporal_state": [
        r"Temporal state\s*:\s*(.+)",
        r"State\s*:\s*(SINGLE_OBSERVATION|STABLE_FAULT|"
        r"PERSISTENT_FAULT|DEVELOPING_FAULT|UNKNOWN)",
    ],
    "confidence_trend": [
        r"Confidence trend\s*:\s*(.+)",
        r"Trend\s*:\s*(STABLE|INCREASING|DECREASING|UNKNOWN)",
    ],
    "persistence_score": [
        r"Persistence(?: score)?\s*:\s*([0-9.eE+-]+)",
    ],
    "position": [
        r"Position\s*:\s*([0-9.eE+-]+)",
        r"Actual position\s*:\s*([0-9.eE+-]+)",
    ],
    "target_position": [
        r"Target(?: position)?\s*:\s*([0-9.eE+-]+)",
        r"Commanded position\s*:\s*([0-9.eE+-]+)",
    ],
    "position_error": [
        r"Position error\s*:\s*([0-9.eE+-]+)",
    ],
    "velocity": [
        r"Velocity\s*:\s*([0-9.eE+-]+)",
    ],
    "force": [
        r"Force(?:/Torque)?\s*:\s*([0-9.eE+-]+)",
        r"Torque\s*:\s*([0-9.eE+-]+)",
    ],
    "motor_current": [
        r"Motor current\s*:\s*([0-9.eE+-]+)",
        r"Current\s*:\s*([0-9.eE+-]+)",
    ],
    "temperature": [
        r"Temperature\s*:\s*([0-9.eE+-]+)",
    ],
    "obstacle_detected": [
        r"Obstacle detected\s*:\s*(True|False|Yes|No)",
        r"Vision obstacle\s*:\s*(True|False|Yes|No)",
    ],
}


def _to_number(value: Any) -> float | None:
    """Convert a value to float safely."""

    if value is None:
        return None

    try:
        number = float(str(value).strip())

        if pd.isna(number):
            return None

        return number

    except (TypeError, ValueError):
        return None


def _normalize_fault(value: Any) -> str | None:
    """Normalize a detected fault label."""

    if value is None:
        return None

    text = str(value).strip()

    text = re.sub(
        r"\s*\(.*?\)\s*$",
        "",
        text,
    ).strip()

    if not text:
        return None

    return text.upper().replace(" ", "_")


def _to_bool(value: Any) -> bool | None:
    """Convert common boolean representations."""

    if value is None:
        return None

    value = str(value).strip().lower()

    if value in {"true", "yes", "1"}:
        return True

    if value in {"false", "no", "0"}:
        return False

    return None


def _search_patterns(
    text: str,
    patterns: list[str],
) -> str | None:
    """Return the first matching captured value."""

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        if match:
            return match.group(1).strip()

    return None


def parse_runtime_lines(
    lines: list[str],
) -> list[dict[str, Any]]:
    """
    Parse actual runtime log lines into sample records.

    A new observation usually begins with a line such as:
        Observation 1

    Also supports:
        Sample 1
        Temporal Sample 1/6

    If the runtime output format differs, unrecognized fields
    remain missing instead of being invented.
    """

    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    sample_counter = 0

    observation_pattern = re.compile(
        r"^\s*(?:Observation|Temporal Sample|Sample)"
        r"\s*#?\s*(\d+)",
        flags=re.IGNORECASE,
    )

    def finalize(record: dict[str, Any] | None) -> None:
        if record is None:
            return

        # Retain only records with at least one useful field.
        useful_fields = [
            key
            for key, value in record.items()
            if key != "sample_index" and value is not None
        ]

        if useful_fields:
            records.append(record)

    for raw_line in lines:
        line = str(raw_line).strip()

        if not line:
            continue

        observation_match = observation_pattern.search(line)

        if observation_match:
            finalize(current)

            sample_counter += 1

            try:
                sample_index = int(
                    observation_match.group(1)
                )
            except ValueError:
                sample_index = sample_counter

            current = {
                "sample_index": sample_index,
            }

            continue

        # Ignore non-observation logs until a sample starts.
        if current is None:
            continue

        for field, patterns in FIELD_PATTERNS.items():
            value = _search_patterns(
                line,
                patterns,
            )

            if value is None:
                continue

            if field in {
                "position",
                "target_position",
                "position_error",
                "velocity",
                "force",
                "motor_current",
                "temperature",
                "telemetry_confidence",
                "multimodal_confidence",
                "temporal_confidence",
                "persistence_score",
            }:
                number = _to_number(value)

                if number is not None:
                    current[field] = number

            elif field == "obstacle_detected":
                current[field] = _to_bool(value)

            elif field.endswith("_fault"):
                current[field] = _normalize_fault(value)

            else:
                current[field] = value.strip()

    finalize(current)

    # If position error is absent but both positions exist,
    # calculate it using the project's established convention:
    # actual position - commanded/target position.
    for record in records:
        if (
            record.get("position_error") is None
            and record.get("position") is not None
            and record.get("target_position") is not None
        ):
            record["position_error"] = (
                record["position"]
                - record["target_position"]
            )

    return records


def latest_sample(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the most recent parsed observation."""

    if not records:
        return {}

    return records[-1]


def to_frame(
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Convert parsed observations into a DataFrame."""

    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame(records)

    if "sample_index" not in frame.columns:
        frame.insert(
            0,
            "sample_index",
            range(1, len(frame) + 1),
        )

    return frame

