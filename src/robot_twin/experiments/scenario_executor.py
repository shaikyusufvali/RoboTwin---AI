"""
Controlled experiment execution for RoboTwin-AI.

This module provides a research-oriented experiment harness for the
RoboTwin-AI digital twin.

Design goals
------------
1. Reproducible scenario execution.
2. Explicit simulation ground truth.
3. Controlled fault injection.
4. Controlled visual obstacle injection.
5. Telemetry diagnosis.
6. Vision observation.
7. Decision-level multimodal fusion.
8. Latency measurement.
9. Defensive resource cleanup.
10. Machine-readable experiment results.

Scientific boundary
-------------------
This module does NOT claim that camera observations alone can identify
an internal actuator fault.

Vision is treated as an independent source of measurable evidence:

    - obstacle presence
    - obstacle distance
    - depth validity
    - visual reliability

Internal actuator fault diagnosis remains telemetry-driven.

The implementation uses original RoboTwin-AI orchestration code and
publicly documented digital-twin engineering patterns.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pybullet as p

from robot_twin.diagnostics.diagnostic_brain import DiagnosticBrain
from robot_twin.diagnostics.fault_injector import (
    FaultInjector,
    FaultScenario,
    FaultType as DiagnosticFaultType,
)
from robot_twin.diagnostics.multimodal_diagnostic_brain import (
    MultimodalDiagnosticBrain,
)
from robot_twin.experiments.experiment_runner import (
    ExperimentScenario,
    build_initial_scenarios,
)
from robot_twin.simulation.robot_simulator import RobotSimulator
from robot_twin.telemetry.pybullet_adapter import PyBulletRobotAdapter
from robot_twin.vision.camera import (
    CameraConfig,
    RobotCamera,
)
from robot_twin.vision.obstacle_detector import ObstacleDetector
from robot_twin.vision.vision_observer import VisionObserver


LOGGER = logging.getLogger(__name__)


# ============================================================================
# Experiment defaults
# ============================================================================

DEFAULT_JOINT_INDEX = 2
DEFAULT_FAULT_SEVERITY = 0.70
DEFAULT_SETTLING_STEPS = 240

DEFAULT_OUTPUT_DIRECTORY = Path(
    "data/experiments"
)

DEFAULT_RESULT_FILENAME = (
    "scenario_results.json"
)


# ============================================================================
# Controlled obstacle configuration
# ============================================================================

OBSTACLE_POSITION = (
    1.90,
    -1.90,
    1.20,
)

OBSTACLE_HALF_EXTENTS = (
    0.30,
    0.30,
    0.55,
)

OBSTACLE_MASS = 0.0


# ============================================================================
# Result schema
# ============================================================================


@dataclass(frozen=True)
class ScenarioExecutionResult:
    """Machine-readable result from one controlled experiment."""

    scenario_id: str
    scenario_name: str
    seed: int

    ground_truth_fault: str
    ground_truth_obstacle: bool
    requested_vision_quality: str

    joint_index: int
    fault_severity: float

    telemetry_status: str
    telemetry_fault: str
    telemetry_confidence: float
    telemetry_anomaly_score: float

    vision_available: bool
    vision_score: float
    vision_reliability: float

    obstacle_detected: bool
    obstacle_distance: float | None

    multimodal_status: str
    multimodal_fault: str
    multimodal_confidence: float
    fusion_mode: str

    telemetry_latency_seconds: float
    vision_latency_seconds: float
    multimodal_latency_seconds: float
    total_latency_seconds: float

    vision_only_fault_prediction: str
    vision_only_evaluable: bool

    execution_status: str
    execution_message: str

    timestamp_utc: str


# ============================================================================
# Scenario executor
# ============================================================================


class ScenarioExecutor:
    """
    Execute controlled RoboTwin-AI diagnostic experiments.

    Architecture
    ------------
        ExperimentScenario
                |
                v
        RobotSimulator
                |
                +---- FaultInjector
                |
                +---- PyBulletRobotAdapter
                |
                +---- DiagnosticBrain
                |
                +---- RobotCamera
                |
                +---- ObstacleDetector
                |
                +---- VisionObserver
                |
                +---- MultimodalDiagnosticBrain
                |
                v
        ScenarioExecutionResult

    The same FaultInjector instance is shared between the simulator
    orchestration and telemetry adapter so the simulation state and
    telemetry state cannot diverge because of separate fault objects.
    """

    def __init__(
        self,
        *,
        joint_index: int = DEFAULT_JOINT_INDEX,
        fault_severity: float = DEFAULT_FAULT_SEVERITY,
        settling_steps: int = DEFAULT_SETTLING_STEPS,
        output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    ) -> None:

        self.joint_index = self._validate_joint_index(
            joint_index
        )

        self.fault_severity = self._validate_severity(
            fault_severity
        )

        self.settling_steps = self._validate_steps(
            settling_steps
        )

        self.output_directory = Path(
            output_directory
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================================
    # Validation
    # ========================================================================

    @staticmethod
    def _validate_joint_index(
        value: int,
    ) -> int:
        """Validate robot joint index."""

        if not isinstance(value, int):
            raise TypeError(
                "joint_index must be an integer."
            )

        if value < 0:
            raise ValueError(
                "joint_index must be non-negative."
            )

        return value

    @staticmethod
    def _validate_severity(
        value: float,
    ) -> float:
        """Validate fault severity."""

        try:
            value = float(value)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise TypeError(
                "fault_severity must be numeric."
            ) from exc

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "fault_severity must be between 0.0 and 1.0."
            )

        return value

    @staticmethod
    def _validate_steps(
        value: int,
    ) -> int:
        """Validate simulation settling steps."""

        if not isinstance(value, int):
            raise TypeError(
                "settling_steps must be an integer."
            )

        if value <= 0:
            raise ValueError(
                "settling_steps must be greater than zero."
            )

        return value

    # ========================================================================
    # Reproducibility
    # ========================================================================

    @staticmethod
    def _seed_random_generators(
        seed: int,
    ) -> None:
        """
        Seed Python-level randomness.

        This controls randomness owned by the experiment framework.
        It does not claim bit-for-bit PyBullet determinism.
        """

        if not isinstance(seed, int):
            raise TypeError(
                "Scenario seed must be an integer."
            )

        random.seed(seed)

    # ========================================================================
    # Fault mapping
    # ========================================================================

    @staticmethod
    def _map_fault_type(
        fault_type: str,
    ) -> DiagnosticFaultType:
        """
        Convert experiment-level fault labels into
        FaultInjector fault types.
        """

        mapping = {
            "normal": DiagnosticFaultType.NORMAL,
            "motor_resistance": DiagnosticFaultType.MOTOR_RESISTANCE,
            "low_motor_power": DiagnosticFaultType.LOW_MOTOR_POWER,
            "position_drift": DiagnosticFaultType.POSITION_DRIFT,
        }

        try:
            return mapping[fault_type]
        except KeyError as exc:
            raise ValueError(
                "Unsupported experiment fault type: "
                f"{fault_type}"
            ) from exc

    def _build_fault_scenario(
        self,
        scenario: ExperimentScenario,
    ) -> FaultScenario:
        """Build the fault-injection configuration."""

        if hasattr(
            scenario.fault_type,
            "value",
        ):
            fault_type = str(
                scenario.fault_type.value
            )
        else:
            fault_type = str(
                scenario.fault_type
            )

        return FaultScenario(
            fault_type=self._map_fault_type(
                fault_type
            ),
            joint_index=self.joint_index,
            severity=self.fault_severity,
        )

    # ========================================================================
    # Simulation settling
    # ========================================================================

    def _settle_simulation(
        self,
        *,
        simulator: RobotSimulator,
        injector: FaultInjector,
        fault_scenario: FaultScenario,
    ) -> None:
        """
        Apply the selected fault during every simulation step.

        Repeated application is intentional because the fault injector
        controls the commanded joint behavior.
        """

        for _ in range(self.settling_steps):
            injector.apply(
                fault_scenario
            )

            simulator.step()

    # ========================================================================
    # Numeric safety
    # ========================================================================

    @staticmethod
    def _safe_float(
        value: object,
    ) -> float:
        """Convert a value into a finite float."""

        try:
            converted = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        if converted != converted:
            return 0.0

        if converted in (
            float("inf"),
            float("-inf"),
        ):
            return 0.0

        return converted

    # ========================================================================
    # Scenario labels
    # ========================================================================

    @staticmethod
    def _fault_label(
        scenario: ExperimentScenario,
    ) -> str:
        """Return canonical fault label."""

        if hasattr(
            scenario.fault_type,
            "value",
        ):
            return str(
                scenario.fault_type.value
            )

        return str(
            scenario.fault_type
        )

    @staticmethod
    def _vision_quality_label(
        scenario: ExperimentScenario,
    ) -> str:
        """Return requested vision condition."""

        return str(
            scenario.vision_quality
        )

    # ========================================================================
    # Obstacle injection
    # ========================================================================

    @staticmethod
    def _create_obstacle(
        simulator: RobotSimulator,
    ) -> int:
        """Create a deterministic static visual obstacle."""

        client_id = simulator.client_id

        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=OBSTACLE_HALF_EXTENTS,
            physicsClientId=client_id,
        )

        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=OBSTACLE_HALF_EXTENTS,
            rgbaColor=(
                0.75,
                0.25,
                0.10,
                1.0,
            ),
            physicsClientId=client_id,
        )

        obstacle_id = p.createMultiBody(
            baseMass=OBSTACLE_MASS,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=OBSTACLE_POSITION,
            physicsClientId=client_id,
        )

        LOGGER.info(
            "Controlled obstacle created. "
            "body_id=%s position=%s",
            obstacle_id,
            OBSTACLE_POSITION,
        )

        return int(obstacle_id)

    @staticmethod
    def _remove_obstacle(
        simulator: RobotSimulator,
        obstacle_id: int | None,
    ) -> None:
        """Remove the controlled obstacle safely."""

        if obstacle_id is None:
            return

        try:
            p.removeBody(
                obstacle_id,
                physicsClientId=simulator.client_id,
            )

            LOGGER.info(
                "Controlled obstacle removed. "
                "body_id=%s",
                obstacle_id,
            )

        except Exception:
            LOGGER.exception(
                "Failed to remove controlled "
                "obstacle body_id=%s.",
                obstacle_id,
            )

    # ========================================================================
    # Camera construction
    # ========================================================================

    @staticmethod
    def _build_camera(
        *,
        simulator: RobotSimulator,
        scenario: ExperimentScenario,
    ) -> RobotCamera:
        """
        Build camera according to the experiment condition.

        normal -> 640x480
        poor   -> 320x240
        """

        quality = str(
            scenario.vision_quality
        ).strip().lower()

        if quality == "poor":

            camera_config = CameraConfig(
                width=320,
                height=240,
            )

            LOGGER.info(
                "Scenario %s using reduced-resolution "
                "vision condition: 320x240.",
                scenario.scenario_id,
            )

        else:

            camera_config = CameraConfig(
                width=640,
                height=480,
            )

            LOGGER.info(
                "Scenario %s using normal "
                "vision condition: 640x480.",
                scenario.scenario_id,
            )

        return RobotCamera(
            simulator=simulator,
            config=camera_config,
        )

    # ========================================================================
    # One scenario
    # ========================================================================

    def execute(
        self,
        scenario: ExperimentScenario,
    ) -> ScenarioExecutionResult:
        """
        Execute one controlled experiment scenario.

        Any scenario-level failure is converted into a structured
        ScenarioExecutionResult so the complete batch can continue.
        """

        if not isinstance(
            scenario,
            ExperimentScenario,
        ):
            raise TypeError(
                "scenario must be an "
                "ExperimentScenario instance."
            )

        self._seed_random_generators(
            scenario.seed
        )

        start_total = time.perf_counter()

        simulator = RobotSimulator()

        injector: FaultInjector | None = None
        adapter: PyBulletRobotAdapter | None = None
        obstacle_id: int | None = None

        try:

            LOGGER.info(
                "Starting scenario %s: %s",
                scenario.scenario_id,
                scenario.name,
            )

            # --------------------------------------------------------------
            # Shared fault state
            # --------------------------------------------------------------

            injector = FaultInjector(
                simulator
            )

            adapter = PyBulletRobotAdapter(
                simulator=simulator,
                fault_injector=injector,
            )

            # --------------------------------------------------------------
            # Connect simulation
            # --------------------------------------------------------------

            adapter.connect()

            # --------------------------------------------------------------
            # Controlled environmental condition
            # --------------------------------------------------------------

            if bool(
                scenario.obstacle_present
            ):
                obstacle_id = self._create_obstacle(
                    simulator
                )

            # --------------------------------------------------------------
            # Diagnostic stack
            # --------------------------------------------------------------

            diagnostic_brain = DiagnosticBrain()

            # IMPORTANT:
            # Use the same DiagnosticBrain instance inside the multimodal
            # layer. This keeps the diagnostic stack explicitly unified.
            multimodal_brain = MultimodalDiagnosticBrain(
                diagnostic_brain=diagnostic_brain
            )

            camera = self._build_camera(
                simulator=simulator,
                scenario=scenario,
            )

            obstacle_detector = ObstacleDetector()

            vision_observer = VisionObserver(
                camera=camera,
                obstacle_detector=obstacle_detector,
            )

            # --------------------------------------------------------------
            # Fault scenario
            # --------------------------------------------------------------

            fault_scenario = self._build_fault_scenario(
                scenario
            )

            # --------------------------------------------------------------
            # Clear previous fault state
            # --------------------------------------------------------------

            injector.clear(
                joint_index=self.joint_index
            )

            # --------------------------------------------------------------
            # Apply fault and settle physics
            # --------------------------------------------------------------

            self._settle_simulation(
                simulator=simulator,
                injector=injector,
                fault_scenario=fault_scenario,
            )

            # --------------------------------------------------------------
            # Collect telemetry
            # --------------------------------------------------------------

            telemetry = adapter.get_joint_telemetry(
                joint_index=self.joint_index
            )

            target_position = injector.get_target_position(
                joint_index=self.joint_index
            )

            # --------------------------------------------------------------
            # Telemetry diagnosis
            # --------------------------------------------------------------

            telemetry_start = time.perf_counter()

            telemetry_result = diagnostic_brain.diagnose(
                telemetry=telemetry,
                target_position=target_position,
            )

            telemetry_latency = (
                time.perf_counter()
                - telemetry_start
            )

            # --------------------------------------------------------------
            # Vision observation
            # --------------------------------------------------------------

            vision_start = time.perf_counter()

            vision_observation = vision_observer.observe()

            vision_latency = (
                time.perf_counter()
                - vision_start
            )

            # --------------------------------------------------------------
            # Multimodal diagnosis
            # --------------------------------------------------------------

            multimodal_start = time.perf_counter()

            multimodal_result = multimodal_brain.diagnose(
                diagnostic_result=telemetry_result,
                vision_observation=vision_observation,
            )

            multimodal_latency = (
                time.perf_counter()
                - multimodal_start
            )

            # --------------------------------------------------------------
            # Total latency
            # --------------------------------------------------------------

            total_latency = (
                time.perf_counter()
                - start_total
            )

            # --------------------------------------------------------------
            # Ground truth
            # --------------------------------------------------------------

            ground_truth_fault = self._fault_label(
                scenario
            )

            ground_truth_obstacle = bool(
                scenario.obstacle_present
            )

            # --------------------------------------------------------------
            # Obstacle distance
            # --------------------------------------------------------------

            obstacle_distance = (
                vision_observation.obstacle_distance
            )

            if obstacle_distance is not None:
                obstacle_distance = self._safe_float(
                    obstacle_distance
                )

            # --------------------------------------------------------------
            # Successful result
            # --------------------------------------------------------------

            result = ScenarioExecutionResult(
                scenario_id=scenario.scenario_id,
                scenario_name=scenario.name,
                seed=scenario.seed,

                ground_truth_fault=ground_truth_fault,
                ground_truth_obstacle=ground_truth_obstacle,
                requested_vision_quality=(
                    self._vision_quality_label(
                        scenario
                    )
                ),

                joint_index=self.joint_index,
                fault_severity=self.fault_severity,

                telemetry_status=str(
                    telemetry_result.status
                ),
                telemetry_fault=str(
                    telemetry_result.fault_type
                ),
                telemetry_confidence=self._safe_float(
                    telemetry_result.confidence
                ),
                telemetry_anomaly_score=self._safe_float(
                    telemetry_result.anomaly_score
                ),

                vision_available=True,
                vision_score=self._safe_float(
                    multimodal_result.vision_score
                ),
                vision_reliability=self._safe_float(
                    multimodal_result.vision_reliability
                ),

                obstacle_detected=bool(
                    vision_observation.obstacle_detected
                ),
                obstacle_distance=obstacle_distance,

                multimodal_status=str(
                    multimodal_result.status
                ),
                multimodal_fault=str(
                    multimodal_result.fault_type
                ),
                multimodal_confidence=self._safe_float(
                    multimodal_result.combined_score
                ),
                fusion_mode=str(
                    multimodal_result.fusion_mode
                ),

                telemetry_latency_seconds=self._safe_float(
                    telemetry_latency
                ),
                vision_latency_seconds=self._safe_float(
                    vision_latency
                ),
                multimodal_latency_seconds=self._safe_float(
                    multimodal_latency
                ),
                total_latency_seconds=self._safe_float(
                    total_latency
                ),

                vision_only_fault_prediction=(
                    "NOT_IMPLEMENTED"
                ),
                vision_only_evaluable=False,

                execution_status="SUCCESS",

                execution_message=(
                    "Scenario executed successfully. "
                    "Vision provides measurable environmental "
                    "evidence; internal fault classification "
                    "remains telemetry-driven."
                ),

                timestamp_utc=datetime.now(
                    timezone.utc
                ).isoformat(),
            )

            LOGGER.info(
                "Scenario %s completed | "
                "truth=%s | telemetry=%s | "
                "vision_obstacle=%s | "
                "multimodal=%s | confidence=%.4f",
                scenario.scenario_id,
                ground_truth_fault,
                telemetry_result.fault_type,
                vision_observation.obstacle_detected,
                multimodal_result.fault_type,
                result.multimodal_confidence,
            )

            return result

        except Exception as exc:

            total_latency = (
                time.perf_counter()
                - start_total
            )

            ground_truth_fault = self._fault_label(
                scenario
            )

            LOGGER.exception(
                "Scenario %s failed.",
                scenario.scenario_id,
            )

            return ScenarioExecutionResult(
                scenario_id=scenario.scenario_id,
                scenario_name=scenario.name,
                seed=scenario.seed,

                ground_truth_fault=ground_truth_fault,
                ground_truth_obstacle=bool(
                    scenario.obstacle_present
                ),
                requested_vision_quality=(
                    self._vision_quality_label(
                        scenario
                    )
                ),

                joint_index=self.joint_index,
                fault_severity=self.fault_severity,

                telemetry_status="ERROR",
                telemetry_fault="ERROR",
                telemetry_confidence=0.0,
                telemetry_anomaly_score=0.0,

                vision_available=False,
                vision_score=0.0,
                vision_reliability=0.0,

                obstacle_detected=False,
                obstacle_distance=None,

                multimodal_status="ERROR",
                multimodal_fault="ERROR",
                multimodal_confidence=0.0,
                fusion_mode="ERROR",

                telemetry_latency_seconds=0.0,
                vision_latency_seconds=0.0,
                multimodal_latency_seconds=0.0,
                total_latency_seconds=self._safe_float(
                    total_latency
                ),

                vision_only_fault_prediction=(
                    "NOT_IMPLEMENTED"
                ),
                vision_only_evaluable=False,

                execution_status="FAILED",
                execution_message=str(exc),

                timestamp_utc=datetime.now(
                    timezone.utc
                ).isoformat(),
            )

        finally:

            # --------------------------------------------------------------
            # Remove obstacle
            # --------------------------------------------------------------

            if obstacle_id is not None:

                try:
                    self._remove_obstacle(
                        simulator=simulator,
                        obstacle_id=obstacle_id,
                    )

                except Exception:
                    LOGGER.exception(
                        "Obstacle cleanup failed "
                        "for scenario %s.",
                        scenario.scenario_id,
                    )

            # --------------------------------------------------------------
            # Clear fault
            # --------------------------------------------------------------

            if injector is not None:

                try:
                    injector.clear(
                        joint_index=self.joint_index
                    )

                except Exception:
                    LOGGER.exception(
                        "Fault cleanup failed "
                        "for scenario %s.",
                        scenario.scenario_id,
                    )

            # --------------------------------------------------------------
            # Disconnect
            # --------------------------------------------------------------

            if adapter is not None:

                try:
                    adapter.disconnect()

                except Exception:
                    LOGGER.exception(
                        "Simulator cleanup failed "
                        "for scenario %s.",
                        scenario.scenario_id,
                    )

    # ========================================================================
    # Batch execution
    # ========================================================================

    def execute_all(
        self,
        scenarios: Iterable[
            ExperimentScenario
        ],
    ) -> list[
        ScenarioExecutionResult
    ]:
        """Execute scenarios sequentially."""

        if scenarios is None:
            raise ValueError(
                "scenarios cannot be None."
            )

        scenarios = list(
            scenarios
        )

        if not scenarios:
            raise ValueError(
                "scenarios must contain at least "
                "one scenario."
            )

        results: list[
            ScenarioExecutionResult
        ] = []

        total_scenarios = len(
            scenarios
        )

        for index, scenario in enumerate(
            scenarios,
            start=1,
        ):

            if not isinstance(
                scenario,
                ExperimentScenario,
            ):
                raise TypeError(
                    "Every item in scenarios must be "
                    "an ExperimentScenario instance."
                )

            LOGGER.info(
                "Running scenario %d/%d: %s",
                index,
                total_scenarios,
                scenario.scenario_id,
            )

            result = self.execute(
                scenario
            )

            results.append(
                result
            )

        return results

    # ========================================================================
    # Result persistence
    # ========================================================================

    def save_results(
        self,
        results: list[
            ScenarioExecutionResult
        ],
        filename: str = DEFAULT_RESULT_FILENAME,
    ) -> Path:
        """Save experiment results as structured JSON."""

        if not results:
            raise ValueError(
                "results must contain at least "
                "one result."
            )

        if not isinstance(
            filename,
            str,
        ):
            raise TypeError(
                "filename must be a string."
            )

        if not filename.endswith(
            ".json"
        ):
            raise ValueError(
                "Experiment result filename "
                "must end with .json."
            )

        output_path = (
            self.output_directory
            / filename
        )

        payload = {
            "experiment": (
                "RoboTwin-AI controlled "
                "multimodal diagnostic experiments"
            ),

            "schema_version": "2.0",

            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),

            "configuration": {
                "joint_index": self.joint_index,
                "fault_severity": self.fault_severity,
                "settling_steps": self.settling_steps,
            },

            "controlled_environment": {
                "obstacle_injection": {
                    "implemented": True,
                    "position": OBSTACLE_POSITION,
                    "half_extents": OBSTACLE_HALF_EXTENTS,
                    "mass": OBSTACLE_MASS,
                },

                "vision_conditions": {
                    "normal_resolution": "640x480",
                    "reduced_resolution": "320x240",
                },
            },

            "research_notes": {
                "ground_truth_source": (
                    "controlled simulation scenario"
                ),

                "telemetry_diagnosis": (
                    "DiagnosticBrain"
                ),

                "vision_role": (
                    "measurable environmental evidence"
                ),

                "multimodal_fusion": (
                    "MultimodalDiagnosticBrain"
                ),

                "vision_only_internal_fault_classification": (
                    "not_implemented"
                ),

                "scenario_obstacle_injection": (
                    "implemented"
                ),

                "vision_quality_condition": (
                    "reduced-resolution condition implemented"
                ),

                "python_random_seeded": True,

                "bitwise_physics_determinism_claimed": False,

                "proprietary_robotics_company_code_used": False,
            },

            "results": [
                asdict(result)
                for result in results
            ],
        }

        output_path.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        LOGGER.info(
            "Experiment results saved to %s",
            output_path,
        )

        return output_path


# ============================================================================
# Result summary
# ============================================================================


def _print_results(
    results: list[
        ScenarioExecutionResult
    ],
) -> None:
    """Print a compact research-oriented result table."""

    print()
    print(
        "EXPERIMENT RESULTS"
    )
    print(
        "=" * 112
    )

    print(
        f"{'ID':<5}"
        f"{'TRUTH':<22}"
        f"{'TEL':<22}"
        f"{'OBS':<8}"
        f"{'MM':<22}"
        f"{'CONF':<10}"
        f"{'STATUS':<10}"
    )

    print(
        "-" * 112
    )

    for result in results:

        truth = result.ground_truth_fault

        telemetry = result.telemetry_fault

        obstacle = (
            "YES"
            if result.obstacle_detected
            else "NO"
        )

        multimodal = result.multimodal_fault

        confidence = (
            f"{result.multimodal_confidence:.4f}"
        )

        status = result.execution_status

        print(
            f"{result.scenario_id:<5}"
            f"{truth:<22}"
            f"{telemetry:<22}"
            f"{obstacle:<8}"
            f"{multimodal:<22}"
            f"{confidence:<10}"
            f"{status:<10}"
        )

    print(
        "-" * 112
    )

    # ------------------------------------------------------------------------
    # Obstacle evaluation
    # ------------------------------------------------------------------------

    obstacle_cases = [
        result
        for result in results
        if result.ground_truth_obstacle
    ]

    if obstacle_cases:

        detected_count = sum(
            1
            for result in obstacle_cases
            if result.obstacle_detected
        )

        print()
        print(
            "OBSTACLE DETECTION CHECK"
        )

        print(
            f"Ground-truth obstacle scenarios: "
            f"{len(obstacle_cases)}"
        )

        print(
            f"Detected obstacles: "
            f"{detected_count}"
        )

        if detected_count == len(
            obstacle_cases
        ):
            print(
                "Controlled obstacle evidence: "
                "DETECTED"
            )
        else:
            print(
                "Controlled obstacle evidence: "
                "NOT DETECTED IN EVERY CASE"
            )

    # ------------------------------------------------------------------------
    # Execution summary
    # ------------------------------------------------------------------------

    successful = sum(
        1
        for result in results
        if result.execution_status == "SUCCESS"
    )

    failed = (
        len(results)
        - successful
    )

    print()
    print(
        "EXECUTION SUMMARY"
    )

    print(
        f"Successful scenarios: {successful}"
    )

    print(
        f"Failed scenarios:     {failed}"
    )

    print(
        "=" * 112
    )


# ============================================================================
# Main entry point
# ============================================================================


def main() -> None:
    """Execute the configured RoboTwin-AI experiment suite."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    scenarios = build_initial_scenarios()

    executor = ScenarioExecutor(
        joint_index=DEFAULT_JOINT_INDEX,
        fault_severity=DEFAULT_FAULT_SEVERITY,
        settling_steps=DEFAULT_SETTLING_STEPS,
    )

    print()
    print(
        "ROBOTWIN-AI CONTROLLED "
        "MULTIMODAL EXPERIMENT EXECUTION"
    )
    print(
        "=" * 112
    )

    results = executor.execute_all(
        scenarios
    )

    output_path = executor.save_results(
        results
    )

    _print_results(
        results
    )

    print()
    print(
        f"Results saved to: "
        f"{output_path}"
    )

    print(
        "=" * 112
    )


if __name__ == "__main__":
    main()