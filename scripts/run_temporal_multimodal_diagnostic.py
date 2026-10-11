"""
Temporal + multimodal diagnostic runtime for RoboTwin-AI.

Phase 15
--------

Temporal Fault Diagnosis

This runtime evaluates whether the existing RoboTwin-AI diagnostic
architecture can distinguish between:

    - a single diagnostic observation
    - a developing fault
    - a persistent fault
    - a stable fault
    - a healthy state

The temporal layer does NOT replace the existing DiagnosticBrain or
MultimodalDiagnosticBrain.

Architecture
------------

    RobotSimulator
          |
          v
    FaultInjector
          |
          v
    PyBulletRobotAdapter
          |
          v
       Telemetry
          |
          +----------------------+
          |                      |
          v                      v
    DiagnosticBrain       VisionObserver
          |                      |
          v                      v
 TemporalDiagnosticEngine   VisionObservation
          |                      |
          +----------+-----------+
                     |
                     v
          MultimodalDiagnosticBrain
                     |
                     v
             Final Diagnostic State

Scientific boundary
-------------------

Telemetry remains the authority for internal actuator-fault diagnosis.

Vision is treated as contextual/environmental evidence and is not allowed
to independently manufacture an internal actuator fault.

The temporal layer evaluates repeated diagnostic observations over time.
It does not invent a new fault classifier.

This module is intentionally implemented as a separate experiment runner
so the previously validated diagnostic components remain unchanged.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from robot_twin.diagnostics.diagnostic_brain import DiagnosticBrain
from robot_twin.diagnostics.fault_injector import (
    FaultInjector,
    FaultScenario,
)
from robot_twin.diagnostics.multimodal_diagnostic_brain import (
    MultimodalDiagnosticBrain,
)
from robot_twin.diagnostics.temporal_diagnostic_engine import (
    TemporalDiagnosticEngine,
)
from robot_twin.experiments.experiment_runner import (
    ExperimentScenario,
    build_initial_scenarios,
)
from robot_twin.simulation.robot_simulator import RobotSimulator
from robot_twin.telemetry.pybullet_adapter import PyBulletRobotAdapter
from robot_twin.vision.camera import CameraConfig, RobotCamera
from robot_twin.vision.obstacle_detector import ObstacleDetector
from robot_twin.vision.vision_observer import VisionObserver


LOGGER = logging.getLogger(__name__)


# ============================================================================
# Experiment configuration
# ============================================================================

DEFAULT_JOINT_INDEX = 2

DEFAULT_HISTORY_SIZE = 10

DEFAULT_SETTLING_STEPS = 240

DEFAULT_TEMPORAL_SAMPLES = 6

DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.10

DEFAULT_SCENARIO_ID = "S05"

DEFAULT_OUTPUT_DIRECTORY = Path(
    "data/experiments/temporal"
)


# ============================================================================
# Result schema
# ============================================================================


@dataclass(frozen=True)
class TemporalSampleResult:
    """
    Machine-readable result for one temporal observation.
    """

    sample_index: int

    timestamp_utc: str

    ground_truth_fault: str

    joint_index: int

    fault_severity: float

    telemetry_fault: str

    telemetry_confidence: float

    multimodal_fault: str

    multimodal_confidence: float

    obstacle_detected: bool

    temporal_fault: str

    temporal_confidence: float

    temporal_state: str

    persistence_score: float

    confidence_trend: str

    developing: bool

    persistent: bool

    stable: bool


# ============================================================================
# Temporal experiment runner
# ============================================================================


class TemporalMultimodalExperiment:
    """
    Execute a controlled temporal diagnostic experiment.

    The runner intentionally reuses the existing RoboTwin-AI components
    instead of implementing replacement diagnostic logic.
    """

    def __init__(
        self,
        *,
        joint_index: int = DEFAULT_JOINT_INDEX,
        history_size: int = DEFAULT_HISTORY_SIZE,
        settling_steps: int = DEFAULT_SETTLING_STEPS,
        temporal_samples: int = DEFAULT_TEMPORAL_SAMPLES,
        sample_interval_seconds: float = (
            DEFAULT_SAMPLE_INTERVAL_SECONDS
        ),
        output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    ) -> None:

        self.joint_index = self._validate_joint_index(
            joint_index
        )

        self.history_size = self._validate_history_size(
            history_size
        )

        self.settling_steps = self._validate_positive_int(
            settling_steps,
            "settling_steps",
        )

        self.temporal_samples = self._validate_positive_int(
            temporal_samples,
            "temporal_samples",
        )

        self.sample_interval_seconds = (
            self._validate_positive_float(
                sample_interval_seconds,
                "sample_interval_seconds",
            )
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
        """
        Validate the robot joint index.
        """

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
    def _validate_history_size(
        value: int,
    ) -> int:
        """
        Validate temporal history length.
        """

        if not isinstance(value, int):
            raise TypeError(
                "history_size must be an integer."
            )

        if value < 2:
            raise ValueError(
                "history_size must be at least 2."
            )

        return value

    @staticmethod
    def _validate_positive_int(
        value: int,
        name: str,
    ) -> int:
        """
        Validate a positive integer configuration value.
        """

        if not isinstance(value, int):
            raise TypeError(
                f"{name} must be an integer."
            )

        if value <= 0:
            raise ValueError(
                f"{name} must be greater than zero."
            )

        return value

    @staticmethod
    def _validate_positive_float(
        value: float,
        name: str,
    ) -> float:
        """
        Validate a positive floating-point configuration value.
        """

        try:
            value = float(value)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise TypeError(
                f"{name} must be numeric."
            ) from exc

        if value <= 0.0:
            raise ValueError(
                f"{name} must be greater than zero."
            )

        return value

    # ========================================================================
    # Scenario selection
    # ========================================================================

    @staticmethod
    def _select_scenario(
        scenarios: tuple[ExperimentScenario, ...],
        scenario_id: str,
    ) -> ExperimentScenario:
        """
        Select one experiment scenario by identifier.
        """

        for scenario in scenarios:

            if scenario.scenario_id == scenario_id:
                return scenario

        available = [
            scenario.scenario_id
            for scenario in scenarios
        ]

        raise ValueError(
            f"Scenario '{scenario_id}' was not found. "
            f"Available scenarios: {available}"
        )

    # ========================================================================
    # Fault scenario
    # ========================================================================

    @staticmethod
    def _build_fault_scenario(
        scenario: ExperimentScenario,
        injector: FaultInjector,
        joint_index: int,
        severity: float,
    ) -> FaultScenario:
        """
        Build the existing FaultInjector configuration.

        The experiment-level scenario owns the canonical fault label,
        while FaultScenario owns the actual simulation configuration.

        The FaultInjector itself remains unchanged and continues to
        consume FaultScenario through its existing apply() API.
        """

        return FaultScenario(
            fault_type=scenario.fault_type,
            joint_index=joint_index,
            severity=severity,
        )

    # ========================================================================
    # Simulation settling
    # ========================================================================

    def _settle(
        self,
        *,
        simulator: RobotSimulator,
        injector: FaultInjector,
        fault_scenario: FaultScenario,
    ) -> None:
        """
        Apply the selected fault repeatedly while physics settles.
        """

        LOGGER.info(
            "Settling simulation for %d steps.",
            self.settling_steps,
        )

        for _ in range(self.settling_steps):

            injector.apply(
                fault_scenario
            )

            simulator.step()

    # ========================================================================
    # Camera
    # ========================================================================

    @staticmethod
    def _build_camera(
        *,
        simulator: RobotSimulator,
        scenario: ExperimentScenario,
    ) -> RobotCamera:
        """
        Build the camera using the same vision-quality policy as the
        validated ScenarioExecutor.
        """

        quality = str(
            scenario.vision_quality
        ).strip().lower()

        if quality == "poor":

            config = CameraConfig(
                width=320,
                height=240,
            )

            LOGGER.info(
                "Temporal scenario %s using poor vision: 320x240.",
                scenario.scenario_id,
            )

        else:

            config = CameraConfig(
                width=640,
                height=480,
            )

            LOGGER.info(
                "Temporal scenario %s using normal vision: 640x480.",
                scenario.scenario_id,
            )

        return RobotCamera(
            simulator=simulator,
            config=config,
        )

    # ========================================================================
    # Safe conversion
    # ========================================================================

    @staticmethod
    def _safe_float(
        value: object,
    ) -> float:
        """
        Convert a value to a finite float.
        """

        try:
            result = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        if result != result:
            return 0.0

        if result in (
            float("inf"),
            float("-inf"),
        ):
            return 0.0

        return result

    # ========================================================================
    # Sample execution
    # ========================================================================

    def run(
        self,
        scenario: ExperimentScenario,
    ) -> list[TemporalSampleResult]:
        """
        Execute the complete temporal experiment.
        """

        simulator = RobotSimulator()

        injector: FaultInjector | None = None

        adapter: PyBulletRobotAdapter | None = None

        temporal_results: list[
            TemporalSampleResult
        ] = []

        try:

            LOGGER.info(
                "Starting temporal experiment: %s",
                scenario.scenario_id,
            )

            LOGGER.info(
                "Ground truth fault: %s",
                scenario.fault_type,
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
            # Connect simulator
            # --------------------------------------------------------------

            adapter.connect()

            # --------------------------------------------------------------
            # Diagnostic stack
            # --------------------------------------------------------------

            diagnostic_brain = DiagnosticBrain()

            multimodal_brain = MultimodalDiagnosticBrain(
                diagnostic_brain=diagnostic_brain
            )

            temporal_engine = TemporalDiagnosticEngine(
                diagnostic_brain=diagnostic_brain,
                history_size=self.history_size,
            )

            # --------------------------------------------------------------
            # Vision stack
            # --------------------------------------------------------------

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
            # Fault configuration
            # --------------------------------------------------------------

            fault_scenario = self._build_fault_scenario(
                scenario=scenario,
                injector=injector,
                joint_index=self.joint_index,
                severity=float(
                    scenario.fault_severity
                ),
            )

            injector.clear(
                joint_index=self.joint_index
            )

            # --------------------------------------------------------------
            # Physics settling
            # --------------------------------------------------------------

            self._settle(
                simulator=simulator,
                injector=injector,
                fault_scenario=fault_scenario,
            )

            # --------------------------------------------------------------
            # Target position
            # --------------------------------------------------------------

            target_position = injector.get_target_position(
                joint_index=self.joint_index
            )

            LOGGER.info(
                "Target position: %.6f",
                target_position,
            )

            # --------------------------------------------------------------
            # Temporal observation loop
            # --------------------------------------------------------------

            for sample_index in range(
                1,
                self.temporal_samples + 1,
            ):

                loop_start = time.perf_counter()

                # ----------------------------------------------------------
                # Keep the fault active
                # ----------------------------------------------------------

                injector.apply(
                    fault_scenario
                )

                simulator.step()

                # ----------------------------------------------------------
                # Live telemetry
                # ----------------------------------------------------------

                telemetry = adapter.get_joint_telemetry(
                    joint_index=self.joint_index
                )

                # ----------------------------------------------------------
                # DEBUG: inspect raw telemetry for S01 false-positive diagnosis
                # ----------------------------------------------------------

                print(
                    "[DEBUG TELEMETRY]",
                    {
                        "position": telemetry.position,
                        "velocity": telemetry.velocity,
                        "force": telemetry.force,
                        "motor_current": telemetry.motor_current,
                        "temperature": telemetry.temperature,
                        "target_position": target_position,
                        "position_error": (
                            telemetry.position - target_position
                        ),
                    },
                )

                # ----------------------------------------------------------
                # Temporal diagnosis
                # ----------------------------------------------------------

                temporal_result = temporal_engine.update(
                    telemetry=telemetry,
                    target_position=target_position,
                )

                # ----------------------------------------------------------
                # Independent telemetry diagnosis
                #
                # This is intentionally kept visible so we can compare
                # single-snapshot diagnosis against temporal diagnosis.
                # ----------------------------------------------------------

                telemetry_result = diagnostic_brain.diagnose(
                    telemetry=telemetry,
                    target_position=target_position,
                )

                # ----------------------------------------------------------
                # Vision observation
                # ----------------------------------------------------------

                vision_observation = (
                    vision_observer.observe()
                )

                # ----------------------------------------------------------
                # Multimodal diagnosis
                # ----------------------------------------------------------

                multimodal_result = multimodal_brain.diagnose(
                    diagnostic_result=telemetry_result,
                    vision_observation=vision_observation,
                )

                # ----------------------------------------------------------
                # Result extraction
                # ----------------------------------------------------------

                result = TemporalSampleResult(
                    sample_index=sample_index,

                    timestamp_utc=(
                        datetime.now(
                            timezone.utc
                        ).isoformat()
                    ),

                    ground_truth_fault=str(
                        scenario.fault_type.value
                        if hasattr(
                            scenario.fault_type,
                            "value",
                        )
                        else scenario.fault_type
                    ),

                    joint_index=self.joint_index,

                    fault_severity=self._safe_float(
                        scenario.fault_severity
                    ),

                    telemetry_fault=str(
                        telemetry_result.fault_type
                    ),

                    telemetry_confidence=self._safe_float(
                        telemetry_result.confidence
                    ),

                    multimodal_fault=str(
                        multimodal_result.fault_type
                    ),

                    multimodal_confidence=self._safe_float(
                        multimodal_result.combined_score
                    ),

                    obstacle_detected=bool(
                        vision_observation.obstacle_detected
                    ),

                    temporal_fault=str(
                        temporal_result.fault_type
                    ),

                    temporal_confidence=self._safe_float(
                        temporal_result.confidence
                    ),

                    temporal_state=str(
                        temporal_result.state
                    ),

                    persistence_score=self._safe_float(
                        temporal_result.persistence_score
                    ),

                    confidence_trend=str(
                        temporal_result.trend
                    ),

                    developing=bool(
                        temporal_result.developing
                    ),

                    persistent=bool(
                        temporal_result.persistent
                    ),

                    stable=bool(
                        temporal_result.stable
                    ),
                )

                temporal_results.append(
                    result
                )

                # ----------------------------------------------------------
                # Research-readable console output
                # ----------------------------------------------------------

                print()

                print(
                    f"[TEMPORAL SAMPLE {sample_index:02d}/"
                    f"{self.temporal_samples:02d}]"
                )

                print(
                    "-" * 72
                )

                print(
                    f"Ground truth       : "
                    f"{result.ground_truth_fault}"
                )

                print(
                    f"Telemetry fault    : "
                    f"{result.telemetry_fault}"
                )

                print(
                    f"Telemetry conf.    : "
                    f"{result.telemetry_confidence:.4f}"
                )

                print(
                    f"Multimodal fault   : "
                    f"{result.multimodal_fault}"
                )

                print(
                    f"Multimodal conf.   : "
                    f"{result.multimodal_confidence:.4f}"
                )

                print(
                    f"Temporal fault     : "
                    f"{result.temporal_fault}"
                )

                print(
                    f"Temporal conf.     : "
                    f"{result.temporal_confidence:.4f}"
                )

                print(
                    f"Temporal state     : "
                    f"{result.temporal_state}"
                )

                print(
                    f"Persistence        : "
                    f"{result.persistence_score:.4f}"
                )

                print(
                    f"Confidence trend   : "
                    f"{result.confidence_trend}"
                )

                print(
                    f"Developing         : "
                    f"{result.developing}"
                )

                print(
                    f"Persistent         : "
                    f"{result.persistent}"
                )

                print(
                    f"Stable             : "
                    f"{result.stable}"
                )

                print(
                    f"Obstacle           : "
                    f"{result.obstacle_detected}"
                )

                print(
                    "-" * 72
                )

                # ----------------------------------------------------------
                # Sampling interval
                # ----------------------------------------------------------

                elapsed = (
                    time.perf_counter()
                    - loop_start
                )

                remaining = (
                    self.sample_interval_seconds
                    - elapsed
                )

                if remaining > 0.0:

                    time.sleep(
                        remaining
                    )

            return temporal_results

        finally:

            # --------------------------------------------------------------
            # Fault cleanup
            # --------------------------------------------------------------

            if injector is not None:

                try:

                    injector.clear(
                        joint_index=self.joint_index
                    )

                except Exception:

                    LOGGER.exception(
                        "Fault cleanup failed."
                    )

            # --------------------------------------------------------------
            # Simulator cleanup
            # --------------------------------------------------------------

            if adapter is not None:

                try:

                    adapter.disconnect()

                except Exception:

                    LOGGER.exception(
                        "Simulator cleanup failed."
                    )

    # ========================================================================
    # Final summary
    # ========================================================================

    @staticmethod
    def print_final_summary(
        results: list[TemporalSampleResult],
    ) -> None:
        """
        Print the final temporal diagnosis summary.
        """

        if not results:

            raise ValueError(
                "No temporal results were produced."
            )

        final = results[-1]

        print()
        print()

        print(
            "ROBOTWIN-AI TEMPORAL MULTIMODAL DIAGNOSIS"
        )

        print(
            "=" * 72
        )

        print(
            f"Ground Truth       : "
            f"{final.ground_truth_fault}"
        )

        print(
            f"Final Telemetry    : "
            f"{final.telemetry_fault}"
        )

        print(
            f"Final Multimodal   : "
            f"{final.multimodal_fault}"
        )

        print(
            f"Final Temporal     : "
            f"{final.temporal_fault}"
        )

        print(
            f"Final State        : "
            f"{final.temporal_state}"
        )

        print(
            f"Final Confidence   : "
            f"{final.temporal_confidence:.4f}"
        )

        print(
            f"Persistence Score  : "
            f"{final.persistence_score:.4f}"
        )

        print(
            f"Confidence Trend   : "
            f"{final.confidence_trend}"
        )

        print(
            f"Developing         : "
            f"{final.developing}"
        )

        print(
            f"Persistent         : "
            f"{final.persistent}"
        )

        print(
            f"Stable             : "
            f"{final.stable}"
        )

        print(
            "=" * 72
        )

        # --------------------------------------------------------------
        # Scientific interpretation
        # --------------------------------------------------------------

        print()

        if final.persistent:

            print(
                "TEMPORAL CONCLUSION : PERSISTENT FAULT"
            )

            print(
                "Repeated observations consistently "
                "support the diagnosed fault."
            )

        elif final.developing:

            print(
                "TEMPORAL CONCLUSION : DEVELOPING FAULT"
            )

            print(
                "The fault is beginning to persist across "
                "multiple observations."
            )

        elif final.stable:

            print(
                "TEMPORAL CONCLUSION : STABLE FAULT"
            )

            print(
                "The diagnosed fault and confidence remain "
                "stable over time."
            )

        else:

            print(
                "TEMPORAL CONCLUSION : SINGLE / INSUFFICIENT"
            )

            print(
                "More temporal observations are required "
                "for a stronger persistence decision."
            )

        print()

        print(
            "Observation count  : "
            f"{len(results)}"
        )

        print(
            "=" * 72
        )

    # ========================================================================
    # JSON persistence
    # ========================================================================

    def save_results(
        self,
        results: list[TemporalSampleResult],
        scenario: ExperimentScenario,
    ) -> Path:
        """
        Persist temporal experiment results as JSON.
        """

        import json
        from dataclasses import asdict

        if not results:

            raise ValueError(
                "results must contain at least one item."
            )

        output_path = (
            self.output_directory
            /
            (
                f"temporal_"
                f"{scenario.scenario_id.lower()}.json"
            )
        )

        payload = {

            "experiment": (
                "RoboTwin-AI temporal multimodal "
                "diagnostic experiment"
            ),

            "phase": (
                "Phase 15 - Temporal Fault Diagnosis"
            ),

            "created_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),

            "configuration": {

                "joint_index": self.joint_index,

                "history_size": self.history_size,

                "settling_steps": self.settling_steps,

                "temporal_samples": self.temporal_samples,

                "sample_interval_seconds": (
                    self.sample_interval_seconds
                ),
            },

            "scenario": {

                "scenario_id": scenario.scenario_id,

                "scenario_name": scenario.name,

                "fault_type": str(
                    scenario.fault_type.value
                    if hasattr(
                        scenario.fault_type,
                        "value",
                    )
                    else scenario.fault_type
                ),

                "fault_severity": (
                    scenario.fault_severity
                ),

                "obstacle_present": bool(
                    scenario.obstacle_present
                ),

                "vision_quality": str(
                    scenario.vision_quality
                ),
            },

            "results": [
                asdict(result)
                for result in results
            ],

            "scientific_boundary": {

                "internal_fault_authority": (
                    "telemetry"
                ),

                "vision_role": (
                    "environmental/contextual evidence"
                ),

                "temporal_role": (
                    "repeated-observation persistence "
                    "and confidence trend analysis"
                ),
            },
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
            "Temporal results saved to %s",
            output_path,
        )

        return output_path


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """
    Execute one controlled temporal diagnostic experiment.
    """

    # ------------------------------------------------------------------------
    # Command-line scenario selection
    #
    # Default remains S05, so running the script without arguments preserves
    # the previous behavior.
    #
    # Example:
    #
    #   python scripts/run_temporal_multimodal_diagnostic.py --scenario S06
    #
    # ------------------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=(
            "Run one RoboTwin-AI temporal "
            "multimodal diagnostic scenario."
        )
    )

    parser.add_argument(
        "--scenario",
        default=DEFAULT_SCENARIO_ID,
        help=(
            "Scenario ID to run, for example "
            "S01, S02, S03, S04, S05 or S06."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    print()

    print(
        "ROBOTWIN-AI PHASE 15"
    )

    print(
        "TEMPORAL MULTIMODAL DIAGNOSTIC EXPERIMENT"
    )

    print(
        "=" * 72
    )

    # ------------------------------------------------------------------------
    # Load existing controlled scenarios
    # ------------------------------------------------------------------------

    scenarios = build_initial_scenarios()

    scenario = TemporalMultimodalExperiment._select_scenario(
        scenarios=scenarios,
        scenario_id=args.scenario,
    )

    print(
        f"Scenario           : "
        f"{scenario.scenario_id}"
    )

    print(
        f"Fault              : "
        f"{scenario.fault_type}"
    )

    print(
        f"Severity            : "
        f"{scenario.fault_severity}"
    )

    print(
        f"Vision quality      : "
        f"{scenario.vision_quality}"
    )

    print(
        f"Temporal samples    : "
        f"{DEFAULT_TEMPORAL_SAMPLES}"
    )

    print(
        f"History size        : "
        f"{DEFAULT_HISTORY_SIZE}"
    )

    print(
        "=" * 72
    )

    # ------------------------------------------------------------------------
    # Create experiment
    # ------------------------------------------------------------------------

    experiment = TemporalMultimodalExperiment(
        joint_index=DEFAULT_JOINT_INDEX,
        history_size=DEFAULT_HISTORY_SIZE,
        settling_steps=DEFAULT_SETTLING_STEPS,
        temporal_samples=DEFAULT_TEMPORAL_SAMPLES,
        sample_interval_seconds=(
            DEFAULT_SAMPLE_INTERVAL_SECONDS
        ),
    )

    # ------------------------------------------------------------------------
    # Run experiment
    # ------------------------------------------------------------------------

    results = experiment.run(
        scenario
    )

    # ------------------------------------------------------------------------
    # Final diagnosis
    # ------------------------------------------------------------------------

    experiment.print_final_summary(
        results
    )

    # ------------------------------------------------------------------------
    # Save machine-readable results
    # ------------------------------------------------------------------------

    output_path = experiment.save_results(
        results=results,
        scenario=scenario,
    )

    print()

    print(
        f"Results saved to: {output_path}"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()