
from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any


# Dashboard data directory
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "dashboard"
EVENTS_FILE = DATA_DIR / "events.json"


def _ensure_data_directory() -> None:
    """Create the dashboard data directory when needed."""
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_events() -> list[dict[str, Any]]:
    """Load previously recorded dashboard events."""

    _ensure_data_directory()

    if not EVENTS_FILE.exists():
        return []

    try:
        with EVENTS_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if isinstance(data, list):
            return [
                item
                for item in data
                if isinstance(item, dict)
            ]

        return []

    except (OSError, json.JSONDecodeError):
        return []


def append_event(
    scenario: str,
    event_type: str,
    message: str,
) -> None:
    """Append a dashboard event with a timestamp."""

    _ensure_data_directory()

    events = load_events()

    event = {
        "timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "scenario": scenario,
        "event_type": event_type,
        "message": message,
    }

    events.append(event)

    # Retain the most recent 2000 events.
    events = events[-2000:]

    temporary_file = EVENTS_FILE.with_suffix(".tmp")

    with temporary_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            events,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temporary_file.replace(EVENTS_FILE)


def csv_bytes(
    records: list[dict[str, Any]],
) -> bytes:
    """Convert parsed runtime observations into downloadable CSV."""

    if not records:
        return b""

    output = io.StringIO(newline="")

    # Preserve all fields found across the observations.
    fieldnames: list[str] = []

    for record in records:
        for field in record:
            if field not in fieldnames:
                fieldnames.append(field)

    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        extrasaction="ignore",
    )

    writer.writeheader()
    writer.writerows(records)

    return output.getvalue().encode("utf-8-sig")
