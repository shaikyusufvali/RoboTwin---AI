"""
Generate labeled robot telemetry for RoboTwin-AI diagnostics.

Design:
    PyBullet simulation
        ↓
    FaultInjector
        ↓
    PyBulletRobotAdapter
        ↓
    SensorModel
        ↓
    Dataset
        ↓
    FaultClassifier

The training telemetry path intentionally matches the runtime telemetry
path so that the classifier sees the same feature representation during
training and inference.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pybullet as p

from robot_twin.diagnostics.fault_injector import (
    FaultInjector,
    FaultScenario,
    FaultType,
)
from robot_twin.simulation.robot_simulator import RobotSimulator
from robot_twin.telemetry.pybullet_adapter import PyBulletRobotAdapter


LOGGER = logging.getLogger(__name__)


DEFAULT_OUTPUT_FILE = (
    Path("data") / "robot_diagnostics_dataset.csv"
)

DEFAULT_SCENARIO_DURATION = 8.0
DEFAULT_SAMPLE_INTERVAL = 0.1
DEFAULT_FAULT_SEVERITY = 0.75

PHYSICS_TIME_STEP = 1.0 / 240.0

JOINT_TARGETS = (
    0.4,
    -0.5,
    0.6,
    -0.4,
    0.5,
    -0.3,
    0.2,
)

FAULT_JOINTS = tuple(range(7))

FAULT_TYPES = (
    FaultType.NORMAL,
    FaultType.MOTOR_RESISTANCE,
    FaultType.LOW_MOTOR_POWER,
    FaultType.POSITION_DRIFT,
)


class DatasetGenerationError(RuntimeError):
    """Raised when telemetry dataset generation fails."""


@dataclass(frozen=True)
class DatasetRecord:
    """Represent one labeled telemetry observation."""

    timestamp: str
    simulation_time: float

    joint_index: int

    commanded_position: float
    position: float
    position_error: float

    velocity: float
    acceleration: float

    force: float
    motor_current: float
    temperature: float

    fault_type: str
    fault_joint_index: int
    fault_severity: float


class RobotDatasetGenerator:
    """
    Generate diagnostic telemetry using the same telemetry stack
    used by runtime diagnosis.
    """

    def __init__(
        self,
        simulator: RobotSimulator,
        output_file: Path = DEFAULT_OUTPUT_FILE,
        scenario_duration: float = DEFAULT_SCENARIO_DURATION,
        sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
    ) -> None:

        if scenario_duration <= 0:
            raise ValueError(
                "scenario_duration must be greater than zero."
            )

        if sample_interval <= 0:
            raise ValueError(
                "sample_interval must be greater than zero."
            )

        if sample_interval < PHYSICS_TIME_STEP:
            raise ValueError(
                "sample_interval must be greater than "
                "the PyBullet physics time step."
            )

        self.simulator = simulator
        self.output_file = Path(output_file)

        self.scenario_duration = float(
            scenario_duration
        )

        self.sample_interval = float(
            sample_interval
        )

        self.fault_injector = FaultInjector(
            simulator
        )

        self.adapter = PyBulletRobotAdapter(
            simulator=simulator,
            fault_injector=self.fault_injector,
        )

        self._previous_velocity: dict[int, float] = {}
        self._commanded_positions: dict[int, float] = {}

    # ============================================================
    # PUBLIC API
    # ============================================================

    def generate(self) -> list[DatasetRecord]:
        """Generate telemetry for all controlled fault scenarios."""

        records: list[DatasetRecord] = []

        for fault_type in FAULT_TYPES:

            LOGGER.info(
                "Generating fault family: %s",
                fault_type.value,
            )

            for fault_joint_index in FAULT_JOINTS:

                LOGGER.info(
                    "Generating scenario | fault=%s | joint=%d",
                    fault_type.value,
                    fault_joint_index,
                )

                self._reset_tracking()
                self._reset_robot_state()

                scenario_records = (
                    self._generate_scenario(
                        fault_type=fault_type,
                        fault_joint_index=fault_joint_index,
                    )
                )

                records.extend(
                    scenario_records
                )

                LOGGER.info(
                    "Scenario complete | fault=%s | joint=%d | records=%d",
                    fault_type.value,
                    fault_joint_index,
                    len(scenario_records),
                )

        LOGGER.info(
            "Dataset generation completed | records=%d",
            len(records),
        )

        return records

    def save(
        self,
        records: list[DatasetRecord],
    ) -> None:
        """Persist dataset as CSV."""

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = [
            "timestamp",
            "simulation_time",
            "joint_index",
            "commanded_position",
            "position",
            "position_error",
            "velocity",
            "acceleration",
            "force",
            "motor_current",
            "temperature",
            "fault_type",
            "fault_joint_index",
            "fault_severity",
        ]

        with self.output_file.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for record in records:

                writer.writerow(
                    {
                        "timestamp": record.timestamp,
                        "simulation_time": (
                            record.simulation_time
                        ),
                        "joint_index": (
                            record.joint_index
                        ),
                        "commanded_position": (
                            record.commanded_position
                        ),
                        "position": record.position,
                        "position_error": (
                            record.position_error
                        ),
                        "velocity": record.velocity,
                        "acceleration": (
                            record.acceleration
                        ),
                        "force": record.force,
                        "motor_current": (
                            record.motor_current
                        ),
                        "temperature": (
                            record.temperature
                        ),
                        "fault_type": (
                            record.fault_type
                        ),
                        "fault_joint_index": (
                            record.fault_joint_index
                        ),
                        "fault_severity": (
                            record.fault_severity
                        ),
                    }
                )

        LOGGER.info(
            "Dataset saved | file=%s | records=%d",
            self.output_file,
            len(records),
        )

    # ============================================================
    # SCENARIO GENERATION
    # ============================================================

    def _generate_scenario(
        self,
        fault_type: FaultType,
        fault_joint_index: int,
    ) -> list[DatasetRecord]:
        """Generate one controlled scenario."""

        scenario = FaultScenario(
            fault_type=fault_type,
            joint_index=fault_joint_index,
            severity=(
                0.0
                if fault_type == FaultType.NORMAL
                else DEFAULT_FAULT_SEVERITY
            ),
        )

        records: list[DatasetRecord] = []

        simulation_time = 0.0
        next_sample_time = 0.0

        while simulation_time < self.scenario_duration:

            self._apply_movement(
                simulation_time=simulation_time
            )

            # The fault injector is applied after the nominal
            # movement command, exactly like runtime control.
            self.fault_injector.apply(
                scenario
            )

            self.simulator.step()

            simulation_time += PHYSICS_TIME_STEP

            if simulation_time + 1e-9 >= next_sample_time:

                records.extend(
                    self._collect_records(
                        scenario=scenario,
                        simulation_time=simulation_time,
                    )
                )

                next_sample_time += (
                    self.sample_interval
                )

        return records

    # ============================================================
    # TELEMETRY COLLECTION
    # ============================================================

    def _collect_records(
        self,
        scenario: FaultScenario,
        simulation_time: float,
    ) -> list[DatasetRecord]:
        """
        Collect telemetry through PyBulletRobotAdapter.

        IMPORTANT:
        motor_current and temperature come from the same
        SensorModel-backed adapter used during runtime.
        """

        timestamp = (
            datetime.now(timezone.utc)
            .isoformat()
        )

        records: list[DatasetRecord] = []

        for joint_index in FAULT_JOINTS:

            telemetry = (
                self.adapter.get_joint_telemetry(
                    joint_index
                )
            )

            commanded_position = (
                self._get_commanded_position(
                    scenario=scenario,
                    joint_index=joint_index,
                )
            )

            position = float(
                telemetry.position
            )

            velocity = float(
                telemetry.velocity
            )

            force = float(
                telemetry.force
            )

            position_error = (
                commanded_position - position
            )

            previous_velocity = (
                self._previous_velocity.get(
                    joint_index,
                    velocity,
                )
            )

            acceleration = (
                velocity - previous_velocity
            ) / self.sample_interval

            record_fault_type = (
                self._resolve_record_fault_type(
                    scenario=scenario,
                    joint_index=joint_index,
                )
            )

            if (
                record_fault_type
                != FaultType.NORMAL.value
            ):
                record_fault_joint = (
                    scenario.joint_index
                )

                record_severity = (
                    scenario.severity
                )

            else:
                record_fault_joint = -1
                record_severity = 0.0

            records.append(
                DatasetRecord(
                    timestamp=timestamp,
                    simulation_time=round(
                        simulation_time,
                        6,
                    ),
                    joint_index=joint_index,
                    commanded_position=round(
                        commanded_position,
                        6,
                    ),
                    position=round(
                        position,
                        6,
                    ),
                    position_error=round(
                        position_error,
                        6,
                    ),
                    velocity=round(
                        velocity,
                        6,
                    ),
                    acceleration=round(
                        acceleration,
                        6,
                    ),
                    force=round(
                        force,
                        6,
                    ),
                    motor_current=round(
                        float(
                            telemetry.motor_current
                        ),
                        6,
                    ),
                    temperature=round(
                        float(
                            telemetry.temperature
                        ),
                        6,
                    ),
                    fault_type=(
                        record_fault_type
                    ),
                    fault_joint_index=(
                        record_fault_joint
                    ),
                    fault_severity=(
                        record_severity
                    ),
                )
            )

            self._previous_velocity[
                joint_index
            ] = velocity

        return records

    # ============================================================
    # COMMAND POSITION
    # ============================================================

    def _apply_movement(
        self,
        simulation_time: float,
    ) -> None:
        """
        Apply nominal movement to every joint.

        The fault injector subsequently overrides the affected
        joint with its fault-specific actuator command.
        """

        robot_id = self.simulator.robot_id

        phase = (
            simulation_time * 0.8
        )

        self._commanded_positions.clear()

        for joint_index, base_target in enumerate(
            JOINT_TARGETS
        ):

            target_position = (
                base_target
                * math.sin(
                    phase + joint_index
                )
            )

            self._commanded_positions[
                joint_index
            ] = target_position

            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_position,
                force=80.0,
                physicsClientId=(
                    self.simulator.client_id
                ),
            )

    def _get_commanded_position(
        self,
        scenario: FaultScenario,
        joint_index: int,
    ) -> float:
        """
        Return the effective commanded position.

        For the affected joint, use the fault injector's
        effective command. For healthy joints, use the
        nominal movement command.
        """

        if (
            joint_index
            == scenario.joint_index
        ):

            fault_telemetry = (
                self.fault_injector
                .get_fault_telemetry(
                    joint_index
                )
            )

            return float(
                fault_telemetry.target_position
            )

        return float(
            self._commanded_positions.get(
                joint_index,
                0.0,
            )
        )

    # ============================================================
    # LABELING
    # ============================================================

    @staticmethod
    def _resolve_record_fault_type(
        scenario: FaultScenario,
        joint_index: int,
    ) -> str:
        """Label only the affected joint."""

        if (
            scenario.fault_type
            == FaultType.NORMAL
        ):
            return FaultType.NORMAL.value

        if (
            joint_index
            == scenario.joint_index
        ):
            return scenario.fault_type.value

        return FaultType.NORMAL.value

    # ============================================================
    # STATE MANAGEMENT
    # ============================================================

    def _reset_robot_state(self) -> None:
        """
        Reset simulated joint states between scenarios.

        This prevents the previous fault episode from becoming
        hidden state for the next experiment.
        """

        robot_id = self.simulator.robot_id

        for joint_index in FAULT_JOINTS:

            p.resetJointState(
                bodyUniqueId=robot_id,
                jointIndex=joint_index,
                targetValue=0.0,
                targetVelocity=0.0,
                physicsClientId=(
                    self.simulator.client_id
                ),
            )

        self.fault_injector.clear_all()

    def _reset_tracking(self) -> None:
        """Reset derived telemetry state."""

        self._previous_velocity.clear()
        self._commanded_positions.clear()

    # ============================================================
    # MAIN
    # ============================================================


def main() -> None:
    """Generate and save the RoboTwin-AI diagnostics dataset."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    simulator = RobotSimulator()

    generator = RobotDatasetGenerator(
        simulator=simulator,
        output_file=DEFAULT_OUTPUT_FILE,
        scenario_duration=8.0,
        sample_interval=0.1,
    )

    try:
        # IMPORTANT:
        # Connect through the adapter so that the adapter's
        # internal connection state is initialized.
        generator.adapter.connect()

        records = generator.generate()

        if not records:
            raise DatasetGenerationError(
                "Dataset generation produced zero records."
            )

        generator.save(records)

    except Exception as exc:
        LOGGER.exception(
            "Dataset generation failed."
        )

        raise DatasetGenerationError(
            "Unable to generate the RoboTwin-AI "
            "diagnostics dataset."
        ) from exc

    finally:
        try:
            generator.adapter.disconnect()
        except Exception:
            LOGGER.exception(
                "Failed to disconnect telemetry adapter cleanly."
            )


if __name__ == "__main__":
    main()