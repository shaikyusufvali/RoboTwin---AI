"""Robot telemetry adapter interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class JointTelemetry:
    """Telemetry returned by a robot joint."""

    joint_index: int
    position: float
    velocity: float
    force: float
    motor_current: float
    temperature: float


class RobotAdapter(ABC):
    """Common interface for simulated and real robots."""

    @abstractmethod
    def connect(self) -> None:
        """Connect to the robot."""

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the robot."""

    @abstractmethod
    def get_joint_telemetry(
        self,
        joint_index: int,
    ) -> JointTelemetry:
        """Read telemetry for a robot joint."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Return whether the adapter is connected."""