"""Natural-language command interpretation for RoboTwin-AI."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Final


LOGGER = logging.getLogger(__name__)


class CommandType(str, Enum):
    """Supported high-level robot commands."""

    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    STOP = "stop"
    MOVE_JOINT = "move_joint"


@dataclass(frozen=True)
class RobotCommand:
    """Structured command understood by the robot system."""

    command_type: CommandType
    distance: float | None = None
    joint_index: int | None = None
    target_position: float | None = None


class CommandInterpreter:
    """Convert human language into structured robot commands."""

    _DISTANCE_PATTERN: Final[re.Pattern[str]] = re.compile(
        r"(\d+(?:\.\d+)?)\s*(?:meters?|m)\b",
        re.IGNORECASE,
    )

    _JOINT_PATTERN: Final[re.Pattern[str]] = re.compile(
        r"joint\s*(\d+).*?"
        r"(-?\d+(?:\.\d+)?)\s*(?:degrees?|deg|°)",
        re.IGNORECASE,
    )

    def interpret(self, text: str) -> RobotCommand:
        """Interpret a human command."""

        normalized = text.strip().lower()

        if not normalized:
            raise ValueError("Command cannot be empty.")

        if self._contains_any(
            normalized,
            ("stop", "halt", "emergency stop"),
        ):
            return RobotCommand(
                command_type=CommandType.STOP,
            )

        joint_match = self._JOINT_PATTERN.search(normalized)

        if joint_match:
            joint_index = int(joint_match.group(1))
            target_degrees = float(joint_match.group(2))

            return RobotCommand(
                command_type=CommandType.MOVE_JOINT,
                joint_index=joint_index,
                target_position=target_degrees,
            )

        distance = self._extract_distance(normalized)

        direction_map = {
            CommandType.MOVE_FORWARD: (
                "forward",
                "move forward",
                "go forward",
            ),
            CommandType.MOVE_BACKWARD: (
                "backward",
                "move backward",
                "go backward",
                "back",
            ),
            CommandType.MOVE_LEFT: (
                "left",
                "move left",
                "go left",
            ),
            CommandType.MOVE_RIGHT: (
                "right",
                "move right",
                "go right",
            ),
        }

        for command_type, keywords in direction_map.items():
            if self._contains_any(normalized, keywords):
                return RobotCommand(
                    command_type=command_type,
                    distance=distance,
                )

        raise ValueError(
            f"Unsupported robot command: {text}"
        )

    @staticmethod
    def _contains_any(
        text: str,
        keywords: tuple[str, ...],
    ) -> bool:
        """Return whether any keyword occurs in the text."""

        return any(keyword in text for keyword in keywords)

    def _extract_distance(
        self,
        text: str,
    ) -> float | None:
        """Extract a distance value from a command."""

        match = self._DISTANCE_PATTERN.search(text)

        if match is None:
            return None

        distance = float(match.group(1))

        if distance <= 0:
            raise ValueError(
                "Distance must be greater than zero."
            )

        return distance


def main() -> None:
    """Run a command interpretation demonstration."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    interpreter = CommandInterpreter()

    commands = (
        "Move forward 2 meters",
        "Go left 1 meter",
        "Stop the robot",
        "Move joint 2 to 30 degrees",
    )

    for text in commands:
        command = interpreter.interpret(text)

        LOGGER.info(
            "Human command: %s | Structured command: %s",
            text,
            command,
        )

        print()
        print(f"Human: {text}")
        print(f"AI command: {command}")


if __name__ == "__main__":
    main()