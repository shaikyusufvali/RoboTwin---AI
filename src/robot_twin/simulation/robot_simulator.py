"""
PyBullet robot simulator for RoboTwin-AI.

Provides a stable digital-twin simulation environment for the KUKA iiwa
robot while preserving the existing RoboTwin-AI simulator API.

The simulator includes an explicit numerical safety layer because
PyBullet's motor maxVelocity parameter is not treated as a hard physical
velocity limiter.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Final

import pybullet as p
import pybullet_data


LOGGER = logging.getLogger(__name__)


# ======================================================================
# Simulation defaults
# ======================================================================

DEFAULT_TIME_STEP: Final[float] = 1.0 / 240.0

DEFAULT_GRAVITY: Final[
    tuple[float, float, float]
] = (
    0.0,
    0.0,
    -9.81,
)

DEFAULT_ROBOT_POSITION: Final[
    tuple[float, float, float]
] = (
    0.0,
    0.0,
    0.0,
)


# ======================================================================
# Controller defaults
# ======================================================================

DEFAULT_POSITION_GAIN: Final[float] = 0.35

DEFAULT_VELOCITY_GAIN: Final[float] = 1.0

# This is our RoboTwin safety limit.
#
# It is intentionally lower than the KUKA URDF velocity capability.
DEFAULT_MAX_JOINT_VELOCITY: Final[float] = 1.50


# ======================================================================
# Physics defaults
# ======================================================================

DEFAULT_NUM_SOLVER_ITERATIONS: Final[int] = 100

DEFAULT_JOINT_DAMPING: Final[float] = 0.5

DEFAULT_LINEAR_DAMPING: Final[float] = 0.04

DEFAULT_ANGULAR_DAMPING: Final[float] = 0.04


# ======================================================================
# Safety defaults
# ======================================================================

FALLBACK_MIN_JOINT_POSITION: Final[float] = -2.5

FALLBACK_MAX_JOINT_POSITION: Final[float] = 2.5

SAFETY_POSITION_MARGIN: Final[float] = 0.02

SAFETY_VELOCITY_LIMIT: Final[float] = 1.50


# ======================================================================
# Configuration
# ======================================================================


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the robot simulation."""

    gui: bool = True

    time_step: float = DEFAULT_TIME_STEP

    gravity: tuple[float, float, float] = (
        DEFAULT_GRAVITY
    )

    robot_position: tuple[float, float, float] = (
        DEFAULT_ROBOT_POSITION
    )

    real_time: bool = False

    position_gain: float = (
        DEFAULT_POSITION_GAIN
    )

    velocity_gain: float = (
        DEFAULT_VELOCITY_GAIN
    )

    max_joint_velocity: float = (
        DEFAULT_MAX_JOINT_VELOCITY
    )

    num_solver_iterations: int = (
        DEFAULT_NUM_SOLVER_ITERATIONS
    )

    joint_damping: float = (
        DEFAULT_JOINT_DAMPING
    )

    # Explicit switch for the numerical safety layer.
    #
    # True:
    #     Safety correction is applied after every physics step.
    #
    # False:
    #     Raw PyBullet simulation state is preserved.
    #
    # This is useful for controlled diagnostics and debugging.
    enable_joint_safety: bool = True

    linear_damping: float = (
        DEFAULT_LINEAR_DAMPING
    )

    angular_damping: float = (
        DEFAULT_ANGULAR_DAMPING
    )


# ======================================================================
# Robot Simulator
# ======================================================================


class RobotSimulator:
    """Control the PyBullet robot simulation."""

    def __init__(
        self,
        config: SimulationConfig | None = None,
    ) -> None:

        self.config = (
            config
            or SimulationConfig()
        )

        self._client_id: int | None = None

        self._robot_id: int | None = None

        self._joint_limits: dict[
            int,
            tuple[float, float],
        ] = {}

        self._safety_corrections: int = 0

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def client_id(self) -> int:
        """Return the active PyBullet client ID."""

        if self._client_id is None:
            raise RuntimeError(
                "Simulation is not connected."
            )

        return self._client_id

    @property
    def robot_id(self) -> int:
        """Return the loaded PyBullet robot ID."""

        if self._robot_id is None:
            raise RuntimeError(
                "Robot has not been loaded."
            )

        return self._robot_id

    # ==================================================================
    # Connection
    # ==================================================================

    def connect(self) -> None:
        """Connect to the PyBullet physics server."""

        if self._client_id is not None:
            return

        mode = (
            p.GUI
            if self.config.gui
            else p.DIRECT
        )

        try:
            client_id = p.connect(
                mode
            )

        except Exception as exc:
            LOGGER.exception(
                "Failed to connect to PyBullet."
            )

            raise RuntimeError(
                "Failed to start PyBullet simulation."
            ) from exc

        if client_id < 0:
            raise RuntimeError(
                "Failed to connect to PyBullet."
            )

        self._client_id = client_id

        LOGGER.info(
            "Connected to PyBullet. Client ID: %s",
            self._client_id,
        )

    # ==================================================================
    # World
    # ==================================================================

    def load_world(self) -> None:
        """Configure physics and load the ground plane."""

        self._require_connection()

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        p.setGravity(
            *self.config.gravity
        )

        p.setTimeStep(
            self.config.time_step
        )

        p.setRealTimeSimulation(
            int(self.config.real_time)
        )

        p.setPhysicsEngineParameter(
            numSolverIterations=(
                self.config.num_solver_iterations
            ),
            fixedTimeStep=(
                self.config.time_step
            ),
            physicsClientId=self.client_id,
        )

        plane_id = p.loadURDF(
            "plane.urdf"
        )

        LOGGER.info(
            "Simulation world loaded. Plane ID: %s",
            plane_id,
        )

    # ==================================================================
    # Robot
    # ==================================================================

    def load_robot(self) -> None:
        """Load and initialize the KUKA iiwa robot."""

        self._require_connection()

        self._robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=self.config.robot_position,
            useFixedBase=True,
        )

        self._load_joint_limits()

        self._initialize_robot_joints()

        LOGGER.info(
            "KUKA iiwa robot loaded. Robot ID: %s",
            self._robot_id,
        )

    # ==================================================================
    # Joint limits
    # ==================================================================

    def _load_joint_limits(self) -> None:
        """Read actual revolute-joint limits from the loaded URDF."""

        self._require_robot()

        self._joint_limits.clear()

        joint_count = p.getNumJoints(
            self.robot_id,
            self.client_id,
        )

        for joint_index in range(
            joint_count
        ):

            info = p.getJointInfo(
                self.robot_id,
                joint_index,
                physicsClientId=self.client_id,
            )

            if info[2] != p.JOINT_REVOLUTE:
                continue

            lower = float(info[8])
            upper = float(info[9])

            if (
                not math.isfinite(lower)
                or not math.isfinite(upper)
                or lower >= upper
            ):
                lower = (
                    FALLBACK_MIN_JOINT_POSITION
                )
                upper = (
                    FALLBACK_MAX_JOINT_POSITION
                )

            self._joint_limits[
                joint_index
            ] = (
                lower,
                upper,
            )

            LOGGER.debug(
                (
                    "Joint %d limits: "
                    "%.4f -> %.4f"
                ),
                joint_index,
                lower,
                upper,
            )

    # ==================================================================
    # Joint limits API
    # ==================================================================

    def get_joint_limits(
        self,
        joint_index: int,
    ) -> tuple[float, float]:
        """Return safe position limits for a revolute joint."""

        if joint_index in self._joint_limits:
            return self._joint_limits[
                joint_index
            ]

        return (
            FALLBACK_MIN_JOINT_POSITION,
            FALLBACK_MAX_JOINT_POSITION,
        )

    # ==================================================================
    # Joint initialization
    # ==================================================================

    def _initialize_robot_joints(self) -> None:
        """Initialize every revolute joint in a deterministic state."""

        self._require_robot()

        joint_count = p.getNumJoints(
            self.robot_id,
            self.client_id,
        )

        for joint_index in range(
            joint_count
        ):

            info = p.getJointInfo(
                self.robot_id,
                joint_index,
                physicsClientId=self.client_id,
            )

            if info[2] != p.JOINT_REVOLUTE:
                continue

            lower, upper = (
                self.get_joint_limits(
                    joint_index
                )
            )

            initial_position = max(
                lower,
                min(
                    upper,
                    0.0,
                ),
            )

            p.resetJointState(
                self.robot_id,
                joint_index,
                targetValue=initial_position,
                targetVelocity=0.0,
                physicsClientId=self.client_id,
            )

            # Keep the KUKA's URDF-level damping.
            p.changeDynamics(
                self.robot_id,
                joint_index,
                linearDamping=(
                    self.config.linear_damping
                ),
                angularDamping=(
                    self.config.angular_damping
                ),
                jointDamping=(
                    self.config.joint_damping
                ),
                physicsClientId=self.client_id,
            )

            # Disable inherited motor command.
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0.0,
                force=0.0,
                physicsClientId=self.client_id,
            )

    # ==================================================================
    # Safety layer
    # ==================================================================

    def _enforce_joint_safety(self) -> None:
        """
        Enforce explicit numerical safety after every physics step.

        PyBullet's maxVelocity motor argument is not treated as a hard
        velocity limiter. Therefore RoboTwin-AI validates the actual
        simulated state after every step.

        If a joint exceeds the allowed numerical velocity, its state is
        corrected using resetJointState.

        If a joint crosses the physical URDF limits, its position is
        clamped to a small interior safety margin.
        """

        self._require_robot()

        joint_count = p.getNumJoints(
            self.robot_id,
            self.client_id,
        )

        velocity_limit = min(
            float(
                self.config.max_joint_velocity
            ),
            SAFETY_VELOCITY_LIMIT,
        )

        for joint_index in range(
            joint_count
        ):

            info = p.getJointInfo(
                self.robot_id,
                joint_index,
                physicsClientId=self.client_id,
            )

            if info[2] != p.JOINT_REVOLUTE:
                continue

            state = p.getJointState(
                self.robot_id,
                joint_index,
                physicsClientId=self.client_id,
            )

            position = float(
                state[0]
            )

            velocity = float(
                state[1]
            )

            lower, upper = (
                self.get_joint_limits(
                    joint_index
                )
            )

            safe_lower = (
                lower
                + SAFETY_POSITION_MARGIN
            )

            safe_upper = (
                upper
                - SAFETY_POSITION_MARGIN
            )

            corrected_position = position
            corrected_velocity = velocity

            changed = False

            # ----------------------------------------------------------
            # Numerical validity
            # ----------------------------------------------------------

            if not math.isfinite(
                position
            ):
                corrected_position = 0.0
                changed = True

            if not math.isfinite(
                velocity
            ):
                corrected_velocity = 0.0
                changed = True

            # ----------------------------------------------------------
            # Physical position limits
            # ----------------------------------------------------------

            if (
                corrected_position
                < safe_lower
            ):
                corrected_position = safe_lower
                changed = True

            elif (
                corrected_position
                > safe_upper
            ):
                corrected_position = safe_upper
                changed = True

            # ----------------------------------------------------------
            # Explicit velocity limit
            # ----------------------------------------------------------

            if (
                corrected_velocity
                < -velocity_limit
            ):
                corrected_velocity = (
                    -velocity_limit
                )
                changed = True

            elif (
                corrected_velocity
                > velocity_limit
            ):
                corrected_velocity = (
                    velocity_limit
                )
                changed = True

            # ----------------------------------------------------------
            # Apply correction
            # ----------------------------------------------------------

            if changed:

                p.resetJointState(
                    self.robot_id,
                    joint_index,
                    targetValue=(
                        corrected_position
                    ),
                    targetVelocity=(
                        corrected_velocity
                    ),
                    physicsClientId=self.client_id,
                )

                self._safety_corrections += 1

                LOGGER.warning(
                    (
                        "Joint safety correction | "
                        "joint=%d | "
                        "position=%.4f -> %.4f | "
                        "velocity=%.4f -> %.4f"
                    ),
                    joint_index,
                    position,
                    corrected_position,
                    velocity,
                    corrected_velocity,
                )

    # ==================================================================
    # Joint telemetry
    # ==================================================================

    def get_joint_states(
        self,
    ) -> list[dict[str, float | int]]:
        """
        Return current states for all revolute joints.

        state[0] -> position
        state[1] -> velocity
        state[2] -> joint reaction forces
        state[3] -> applied motor torque

        The existing RoboTwin-AI ``force`` field continues to expose
        state[3] for backward compatibility.
        """

        self._require_robot()

        states: list[
            dict[str, float | int]
        ] = []

        joint_count = p.getNumJoints(
            self.robot_id,
            self.client_id,
        )

        for joint_index in range(
            joint_count
        ):

            info = p.getJointInfo(
                self.robot_id,
                joint_index,
                physicsClientId=self.client_id,
            )

            if info[2] != p.JOINT_REVOLUTE:
                continue

            state = p.getJointState(
                self.robot_id,
                joint_index,
                physicsClientId=self.client_id,
            )

            states.append(
                {
                    "joint_index": joint_index,
                    "position": float(
                        state[0]
                    ),
                    "velocity": float(
                        state[1]
                    ),
                    "force": float(
                        state[3]
                    ),
                }
            )

        return states

    # ==================================================================
    # Single joint telemetry
    # ==================================================================

    def get_joint_state(
        self,
        joint_index: int,
    ) -> dict[str, float | int]:
        """Return telemetry for one revolute joint."""

        self._require_robot()

        info = p.getJointInfo(
            self.robot_id,
            joint_index,
            physicsClientId=self.client_id,
        )

        if info[2] != p.JOINT_REVOLUTE:
            raise ValueError(
                f"Joint {joint_index} is not revolute."
            )

        state = p.getJointState(
            self.robot_id,
            joint_index,
            physicsClientId=self.client_id,
        )

        return {
            "joint_index": joint_index,
            "position": float(
                state[0]
            ),
            "velocity": float(
                state[1]
            ),
            "force": float(
                state[3]
            ),
        }

    # ==================================================================
    # Simulation
    # ==================================================================

    def step(self) -> None:
        """
        Advance simulation by one physics step.

        The safety layer is enabled by default but can be disabled
        through SimulationConfig for controlled diagnostics.
        """

        self._require_connection()

        p.stepSimulation(
            physicsClientId=self.client_id,
        )

        if (
            self._robot_id is not None
            and self.config.enable_joint_safety
        ):
            self._enforce_joint_safety()

    # ==================================================================
    # Multiple steps
    # ==================================================================

    def step_many(
        self,
        steps: int,
    ) -> None:
        """Advance simulation by multiple deterministic steps."""

        if steps <= 0:
            raise ValueError(
                "steps must be greater than zero."
            )

        for _ in range(steps):
            self.step()

    # ==================================================================
    # Run
    # ==================================================================

    def run(
        self,
        duration_seconds: float = 30.0,
    ) -> None:
        """Run the simulation for a fixed duration."""

        if duration_seconds <= 0:
            raise ValueError(
                "duration_seconds must be greater than zero."
            )

        self._require_robot()

        start_time = time.monotonic()

        while (
            time.monotonic()
            - start_time
            < duration_seconds
        ):

            self.step()

            if not self.config.real_time:
                time.sleep(
                    self.config.time_step
                )

    # ==================================================================
    # Safety diagnostics
    # ==================================================================

    @property
    def safety_corrections(self) -> int:
        """Return number of numerical safety corrections."""

        return self._safety_corrections

    # ==================================================================
    # Shutdown
    # ==================================================================

    def close(self) -> None:
        """Close the PyBullet connection."""

        if self._client_id is None:
            return

        try:

            p.disconnect(
                self._client_id
            )

            LOGGER.info(
                "Simulation closed."
            )

        finally:

            self._client_id = None
            self._robot_id = None
            self._joint_limits.clear()
            self._safety_corrections = 0

    # ==================================================================
    # Validation
    # ==================================================================

    def _require_connection(self) -> None:
        """Ensure PyBullet is connected."""

        if self._client_id is None:
            raise RuntimeError(
                "PyBullet is not connected."
            )

    def _require_robot(self) -> None:
        """Ensure the robot has been loaded."""

        self._require_connection()

        if self._robot_id is None:
            raise RuntimeError(
                "Robot has not been loaded."
            )

    # ==================================================================
    # Context manager
    # ==================================================================

    def __enter__(
        self,
    ) -> "RobotSimulator":

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

        self.close()


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    """Start the RoboTwin-AI simulation."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    with RobotSimulator() as simulator:

        simulator.run(
            duration_seconds=30.0
        )


if __name__ == "__main__":
    main()