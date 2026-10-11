"""Generate realistic multi-episode robot diagnostic telemetry."""

from __future__ import annotations

import csv
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import pybullet as p

from robot_twin.diagnostics.fault_injector import (
    FaultInjector,
    FaultScenario,
    FaultType,
)
from robot_twin.simulation.robot_simulator import RobotSimulator


LOGGER = logging.getLogger(__name__)

OUTPUT_FILE = Path("data") / "robot_diagnostics_dataset_v3.csv"

PHYSICS_FREQUENCY = 240
TELEMETRY_FREQUENCY = 10

PHYSICS_TIME_STEP = 1.0 / PHYSICS_FREQUENCY
TELEMETRY_INTERVAL = 1.0 / TELEMETRY_FREQUENCY

EPISODE_DURATION = 10.0
EPISODES_PER_FAULT = 8

FAULT_JOINT_INDEX = 2

JOINT_TARGETS = (
    0.4,
    -0.5,
    0.6,
    -0.4,
    0.5,
    -0.3,
    0.2,
)

RANDOM_SEED = 42

POSITION_NOISE_STD = 0.002
VELOCITY_NOISE_STD = 0.01
FORCE_NOISE_STD = 0.15
CURRENT_NOISE_STD = 0.03
TEMPERATURE_NOISE_STD = 0.2


@dataclass(frozen=True)
class DatasetRecord:
    """Represent one robot telemetry observation."""

    episode_id: int
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


class RobotDatasetGeneratorV3:
    """Generate multi-episode diagnostic telemetry."""

    def __init__(
        self,
        simulator: RobotSimulator,
        episode_duration: float = EPISODE_DURATION,
        episodes_per_fault: int = EPISODES_PER_FAULT,
    ) -> None:
        if episode_duration <= 0:
            raise ValueError(
                "episode_duration must be greater than zero."
            )

        if episodes_per_fault <= 0:
            raise ValueError(
                "episodes_per_fault must be greater than zero."
            )

        self.simulator = simulator
        self.episode_duration = episode_duration
        self.episodes_per_fault = episodes_per_fault
        self.fault_injector = FaultInjector(simulator)

        self._previous_velocity: dict[int, float] = {}
        self._random = random.Random(RANDOM_SEED)

    def generate(self) -> list[DatasetRecord]:
        """Generate telemetry for multiple fault episodes."""
        records: list[DatasetRecord] = []

        scenarios = (
            FaultType.NORMAL,
            FaultType.MOTOR_RESISTANCE,
            FaultType.LOW_MOTOR_POWER,
            FaultType.POSITION_DRIFT,
        )

        episode_id = 1

        for fault_type in scenarios:
            for episode_number in range(self.episodes_per_fault):
                severity = self._generate_severity(fault_type)

                LOGGER.info(
                    "Episode %s | fault=%s | severity=%.2f",
                    episode_id,
                    fault_type.value,
                    severity,
                )

                episode_records = self._generate_episode(
                    episode_id=episode_id,
                    fault_type=fault_type,
                    severity=severity,
                )

                records.extend(episode_records)

                episode_id += 1

        LOGGER.info(
            "Generated %s total telemetry records.",
            len(records),
        )

        return records

    def save(
        self,
        records: list[DatasetRecord],
        output_file: Path = OUTPUT_FILE,
    ) -> None:
        """Save telemetry records to CSV."""
        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = list(
            DatasetRecord.__annotations__.keys()
        )

        with output_file.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for record in records:
                writer.writerow(record.__dict__)

        LOGGER.info(
            "Dataset saved: %s",
            output_file,
        )

    def _generate_episode(
        self,
        episode_id: int,
        fault_type: FaultType,
        severity: float,
    ) -> list[DatasetRecord]:
        """Generate one independent simulation episode."""
        records: list[DatasetRecord] = []

        scenario = FaultScenario(
            fault_type=fault_type,
            joint_index=FAULT_JOINT_INDEX,
            severity=severity,
        )

        self._previous_velocity.clear()

        simulation_time = 0.0
        next_sample_time = 0.0

        physics_steps = int(
            self.episode_duration
            * PHYSICS_FREQUENCY
        )

        for _ in range(physics_steps):
            self._apply_movement(simulation_time)

            self.fault_injector.apply(scenario)

            self.simulator.step()

            if simulation_time >= next_sample_time:
                records.extend(
                    self._collect_records(
                        episode_id=episode_id,
                        scenario=scenario,
                        simulation_time=simulation_time,
                    )
                )

                next_sample_time += TELEMETRY_INTERVAL

            simulation_time += PHYSICS_TIME_STEP

        return records

    def _apply_movement(
        self,
        simulation_time: float,
    ) -> None:
        """Apply smooth movement to all robot joints."""
        robot_id = self.simulator.robot_id
        client_id = self.simulator.client_id

        for joint_index, base_target in enumerate(
            JOINT_TARGETS
        ):
            target = base_target + (
                0.15
                * math.sin(
                    simulation_time * 2.0
                    + joint_index
                )
            )

            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=80.0,
                physicsClientId=client_id,
            )

    def _collect_records(
        self,
        episode_id: int,
        scenario: FaultScenario,
        simulation_time: float,
    ) -> list[DatasetRecord]:
        """Collect noisy telemetry from all revolute joints."""
        records: list[DatasetRecord] = []

        for joint_state in self.simulator.get_joint_states():
            joint_index = int(
                joint_state["joint_index"]
            )

            raw_position = float(
                joint_state["position"]
            )

            raw_velocity = float(
                joint_state["velocity"]
            )

            raw_force = float(
                joint_state["force"]
            )

            commanded_position = (
                self._commanded_position(
                    joint_index,
                    simulation_time,
                )
            )

            position = (
                raw_position
                + self._random.gauss(
                    0.0,
                    POSITION_NOISE_STD,
                )
            )

            velocity = (
                raw_velocity
                + self._random.gauss(
                    0.0,
                    VELOCITY_NOISE_STD,
                )
            )

            force = (
                raw_force
                + self._random.gauss(
                    0.0,
                    FORCE_NOISE_STD,
                )
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
            ) / TELEMETRY_INTERVAL

            self._previous_velocity[
                joint_index
            ] = velocity

            motor_current = (
                self._estimate_motor_current(
                    force=force,
                    scenario=scenario,
                    joint_index=joint_index,
                )
            )

            motor_current += self._random.gauss(
                0.0,
                CURRENT_NOISE_STD,
            )

            temperature = (
                self._estimate_temperature(
                    motor_current=motor_current,
                    scenario=scenario,
                    joint_index=joint_index,
                )
            )

            temperature += self._random.gauss(
                0.0,
                TEMPERATURE_NOISE_STD,
            )

            records.append(
                DatasetRecord(
                    episode_id=episode_id,
                    simulation_time=round(
                        simulation_time,
                        4,
                    ),
                    joint_index=joint_index,
                    commanded_position=round(
                        commanded_position,
                        6,
                    ),
                    position=round(position, 6),
                    position_error=round(
                        position_error,
                        6,
                    ),
                    velocity=round(velocity, 6),
                    acceleration=round(
                        acceleration,
                        6,
                    ),
                    force=round(force, 6),
                    motor_current=round(
                        motor_current,
                        6,
                    ),
                    temperature=round(
                        temperature,
                        6,
                    ),
                    fault_type=scenario.fault_type.value,
                    fault_joint_index=(
                        scenario.joint_index
                    ),
                    fault_severity=round(
                        scenario.severity,
                        4,
                    ),
                )
            )

        return records

    def _generate_severity(
        self,
        fault_type: FaultType,
    ) -> float:
        """Generate a random fault severity."""
        if fault_type == FaultType.NORMAL:
            return 0.0

        return round(
            self._random.uniform(
                0.30,
                0.90,
            ),
            2,
        )

    @staticmethod
    def _commanded_position(
        joint_index: int,
        simulation_time: float,
    ) -> float:
        """Calculate commanded joint position."""
        return JOINT_TARGETS[joint_index] + (
            0.15
            * math.sin(
                simulation_time * 2.0
                + joint_index
            )
        )

    @staticmethod
    def _estimate_motor_current(
        force: float,
        scenario: FaultScenario,
        joint_index: int,
    ) -> float:
        """Estimate synthetic motor current."""
        current = abs(force) * 0.08 + 0.5

        if (
            scenario.fault_type
            == FaultType.MOTOR_RESISTANCE
            and joint_index == scenario.joint_index
        ):
            current *= (
                1.0
                + scenario.severity * 2.0
            )

        elif (
            scenario.fault_type
            == FaultType.LOW_MOTOR_POWER
            and joint_index == scenario.joint_index
        ):
            current *= (
                0.6
                - scenario.severity * 0.15
            )

        return current

    @staticmethod
    def _estimate_temperature(
        motor_current: float,
        scenario: FaultScenario,
        joint_index: int,
    ) -> float:
        """Estimate synthetic motor temperature."""
        temperature = (
            35.0
            + motor_current * 2.5
        )

        if (
            scenario.fault_type
            == FaultType.MOTOR_RESISTANCE
            and joint_index == scenario.joint_index
        ):
            temperature += (
                20.0
                * scenario.severity
            )

        return temperature


def main() -> None:
    """Generate the RoboTwin-AI V3 dataset."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    simulator = RobotSimulator(
        config=None,
    )

    try:
        simulator.connect()
        simulator.load_world()
        simulator.load_robot()

        generator = RobotDatasetGeneratorV3(
            simulator=simulator,
        )

        records = generator.generate()
        generator.save(records)

    except Exception:
        LOGGER.exception(
            "V3 dataset generation failed."
        )
        raise

    finally:
        simulator.close()


if __name__ == "__main__":
    main()
