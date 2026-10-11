"""PyBullet robot simulator for RoboTwin-AI."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Final

import pybullet as p
import pybullet_data


LOGGER = logging.getLogger(__name__)

DEFAULT_TIME_STEP: Final[float] = 1.0 / 240.0
DEFAULT_GRAVITY: Final[tuple[float, float, float]] = (0.0, 0.0, -9.81)
DEFAULT_ROBOT_POSITION: Final[tuple[float, float, float]] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the robot simulation."""

    gui: bool = True
    time_step: float = DEFAULT_TIME_STEP
    gravity: tuple[float, float, float] = DEFAULT_GRAVITY
    robot_position: tuple[float, float, float] = DEFAULT_ROBOT_POSITION
    real_time: bool = False


class RobotSimulator:
    """Control the PyBullet robot simulation."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self._client_id: int | None = None
        self._robot_id: int | None = None

    @property
    def client_id(self) -> int:
        """Return the active PyBullet client ID."""
        self._require_connection()
        return self._client_id

    @property
    def robot_id(self) -> int:
        """Return the loaded PyBullet robot ID."""
        self._require_robot()
        return self._robot_id

    def connect(self) -> None:
        """Connect to the PyBullet physics server."""
        if self._client_id is not None:
            return

        mode = p.GUI if self.config.gui else p.DIRECT

        try:
            client_id = p.connect(mode)
        except Exception as exc:
            LOGGER.exception("Failed to connect to PyBullet.")
            raise RuntimeError(
                "Failed to start PyBullet simulation."
            ) from exc

        if client_id < 0:
            raise RuntimeError("Failed to connect to PyBullet.")

        self._client_id = client_id

        LOGGER.info(
            "Connected to PyBullet. Client ID: %s",
            self._client_id,
        )

    def load_world(self) -> None:
        """Configure physics and load the ground plane."""
        self._require_connection()

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*self.config.gravity)
        p.setTimeStep(self.config.time_step)
        p.setRealTimeSimulation(int(self.config.real_time))

        plane_id = p.loadURDF("plane.urdf")

        LOGGER.info(
            "Simulation world loaded. Plane ID: %s",
            plane_id,
        )

    def load_robot(self) -> None:
        """Load the KUKA iiwa robot model."""
        self._require_connection()

        self._robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=self.config.robot_position,
            useFixedBase=True,
        )

        LOGGER.info(
            "KUKA iiwa robot loaded. Robot ID: %s",
            self._robot_id,
        )

    def get_joint_states(self) -> list[dict[str, float | int]]:
        """Return current states for all revolute joints."""
        self._require_robot()

        states: list[dict[str, float | int]] = []

        for joint_index in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(
                self.robot_id,
                joint_index,
            )

            if joint_info[2] != p.JOINT_REVOLUTE:
                continue

            state = p.getJointState(
                self.robot_id,
                joint_index,
            )

            states.append(
                {
                    "joint_index": joint_index,
                    "position": float(state[0]),
                    "velocity": float(state[1]),
                    "force": float(state[3]),
                }
            )

        return states

    def step(self) -> None:
        """Advance the simulation by one timestep."""
        self._require_connection()
        p.stepSimulation()

    def run(self, duration_seconds: float = 30.0) -> None:
        """Run the simulation for a fixed duration."""
        if duration_seconds <= 0:
            raise ValueError(
                "duration_seconds must be greater than zero."
            )

        self._require_robot()

        start_time = time.monotonic()

        while time.monotonic() - start_time < duration_seconds:
            self.step()

            if not self.config.real_time:
                time.sleep(self.config.time_step)

    def close(self) -> None:
        """Close the PyBullet connection."""
        if self._client_id is None:
            return

        try:
            p.disconnect(self._client_id)
            LOGGER.info("Simulation closed.")
        finally:
            self._client_id = None
            self._robot_id = None

    def _require_connection(self) -> None:
        """Ensure that PyBullet is connected."""
        if self._client_id is None:
            raise RuntimeError(
                "PyBullet is not connected."
            )

    def _require_robot(self) -> None:
        """Ensure that the robot has been loaded."""
        self._require_connection()

        if self._robot_id is None:
            raise RuntimeError(
                "Robot has not been loaded."
            )

    def __enter__(self) -> "RobotSimulator":
        """Initialize the simulation context."""
        self.connect()
        self.load_world()
        self.load_robot()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        """Close the simulation context."""
        self.close()


def main() -> None:
    """Start the RoboTwin-AI robot simulation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    with RobotSimulator() as simulator:
        simulator.run(duration_seconds=30.0)


if __name__ == "__main__":
    main()