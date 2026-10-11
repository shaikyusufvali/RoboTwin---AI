"""
RoboTwin-AI
Randomized Severity Robustness Evaluation

Purpose
-------
Evaluate diagnostic robustness across repeated controlled experiments
while randomly varying fault severity.

Existing architecture reused:
    ExperimentScenario
    ScenarioExecutor
    DiagnosticBrain
    MultimodalDiagnosticBrain
    FaultInjector
    PyBullet simulation

Scientific scope
----------------
This experiment randomizes fault severity only.

It does NOT claim robustness against:
    - sensor noise
    - payload variation
    - friction variation
    - mass variation
    - actuator parameter variation
    - camera blur
    - camera occlusion
    - environmental randomness

Those are separate future robustness dimensions.
"""

from __future__ import annotations

import json
import logging
import random
import time

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from robot_twin.experiments.experiment_runner import (
    ExperimentScenario,
    build_initial_scenarios,
)

from robot_twin.experiments.scenario_executor import (
    ScenarioExecutor,
)


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_JOINT_INDEX = 2

DEFAULT_SETTLING_STEPS = 240

DEFAULT_REPETITIONS = 3

DEFAULT_MIN_SEVERITY = 0.30

DEFAULT_MAX_SEVERITY = 0.90

DEFAULT_BASE_SEED = 73001

DEFAULT_OUTPUT_FILE = (
    Path("data")
    / "experiments"
    / "robustness"
    / "randomized_robustness_results.json"
)


# ---------------------------------------------------------------------------
# Trial result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RobustnessTrial:
    """
    Result of one randomized robustness trial.
    """

    trial_id: str

    repetition: int

    scenario_id: str

    scenario_name: str

    seed: int

    fault_severity: float

    ground_truth_fault: str

    ground_truth_obstacle: bool

    requested_vision_quality: str

    telemetry_fault: str

    telemetry_confidence: float

    multimodal_fault: str

    multimodal_confidence: float

    fusion_mode: str

    obstacle_detected: bool

    execution_status: str

    execution_message: str

    total_latency_seconds: float

    telemetry_correct: bool

    multimodal_correct: bool

    obstacle_detection_correct: bool | None


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RobustnessSummary:
    """
    Aggregate metrics for the randomized robustness study.
    """

    total_trials: int

    successful_trials: int

    failed_trials: int

    telemetry_accuracy: float

    multimodal_accuracy: float

    obstacle_evaluable_trials: int

    obstacle_detection_accuracy: float

    minimum_severity: float

    maximum_severity: float

    mean_severity: float

    scenario_count: int

    repetition_count: int

    study_type: str

    interpretation: str


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RandomizedRobustnessEvaluator:
    """
    Repeated controlled robustness evaluation.

    The existing ScenarioExecutor remains the source of truth for:

        simulation
        fault injection
        telemetry
        vision
        diagnostic brain
        multimodal diagnosis
        obstacle detection

    This class only orchestrates repeated experiments and aggregates
    the resulting evidence.
    """

    def __init__(
        self,
        *,
        repetitions: int = DEFAULT_REPETITIONS,
        min_severity: float = DEFAULT_MIN_SEVERITY,
        max_severity: float = DEFAULT_MAX_SEVERITY,
        base_seed: int = DEFAULT_BASE_SEED,
        joint_index: int = DEFAULT_JOINT_INDEX,
        settling_steps: int = DEFAULT_SETTLING_STEPS,
        output_file: Path = DEFAULT_OUTPUT_FILE,
    ) -> None:

        if (
            not isinstance(
                repetitions,
                int,
            )
            or repetitions <= 0
        ):
            raise ValueError(
                "repetitions must be a positive integer."
            )

        self.repetitions = repetitions

        self.min_severity = (
            self._validate_severity(
                min_severity,
                "min_severity",
            )
        )

        self.max_severity = (
            self._validate_severity(
                max_severity,
                "max_severity",
            )
        )

        if (
            self.min_severity
            > self.max_severity
        ):
            raise ValueError(
                "min_severity cannot be greater "
                "than max_severity."
            )

        if not isinstance(
            base_seed,
            int,
        ):
            raise TypeError(
                "base_seed must be an integer."
            )

        if (
            not isinstance(
                joint_index,
                int,
            )
            or joint_index < 0
        ):
            raise ValueError(
                "joint_index must be a "
                "non-negative integer."
            )

        if (
            not isinstance(
                settling_steps,
                int,
            )
            or settling_steps <= 0
        ):
            raise ValueError(
                "settling_steps must be "
                "a positive integer."
            )

        self.base_seed = base_seed

        self.joint_index = joint_index

        self.settling_steps = settling_steps

        self.output_file = Path(
            output_file
        )

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    # -----------------------------------------------------------------------
    # Validation helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _validate_severity(
        value: float,
        name: str,
    ) -> float:
        """
        Validate severity in [0, 1].
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

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must be between "
                f"0.0 and 1.0."
            )

        return value

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> float:
        """
        Safely convert a value to finite float.
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

    @staticmethod
    def _value(
        result: Any,
        name: str,
        default: Any = None,
    ) -> Any:
        """
        Read a field from either a dict or an object.
        """

        if isinstance(
            result,
            dict,
        ):
            return result.get(
                name,
                default,
            )

        return getattr(
            result,
            name,
            default,
        )

    # -----------------------------------------------------------------------
    # Fault-label normalization
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalize_fault_label(
        value: Any,
    ) -> str:
        """
        Normalize fault labels before metric comparison.

        Why this is required
        --------------------
        ExperimentScenario may contain:

            motor_resistance

        while DiagnosticBrain returns:

            MOTOR_RESISTANCE

        These represent the same fault but are different raw strings.

        Metrics must compare canonical labels.
        """

        if value is None:
            return "UNKNOWN"

        text = str(
            value
        ).strip().upper()

        aliases = {
            "NORMAL": "NORMAL",
            "MOTOR_RESISTANCE": "MOTOR_RESISTANCE",
            "LOW_MOTOR_POWER": "LOW_MOTOR_POWER",
            "POSITION_DRIFT": "POSITION_DRIFT",
            "UNKNOWN": "UNKNOWN",
            "ERROR": "ERROR",
        }

        return aliases.get(
            text,
            text,
        )

    # -----------------------------------------------------------------------
    # Scenario helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _fault_label(
        scenario: ExperimentScenario,
    ) -> str:
        """
        Extract the scenario fault label.
        """

        value = scenario.fault_type

        if hasattr(
            value,
            "value",
        ):
            value = value.value

        return (
            RandomizedRobustnessEvaluator
            ._normalize_fault_label(
                value
            )
        )

    @staticmethod
    def _vision_quality_label(
        scenario: ExperimentScenario,
    ) -> str:
        """
        Return configured vision condition.
        """

        return str(
            scenario.vision_quality
        )

    # -----------------------------------------------------------------------
    # Seed generation
    # -----------------------------------------------------------------------

    @staticmethod
    def _trial_seed(
        base_seed: int,
        repetition: int,
        scenario_index: int,
    ) -> int:
        """
        Generate unique framework seed per trial.

        ScenarioExecutor consumes scenario.seed,
        therefore the generated seed is written into
        ExperimentScenario before execution.
        """

        return (
            base_seed
            + (
                repetition
                * 10_000
            )
            + scenario_index
        )

    # -----------------------------------------------------------------------
    # Severity sampling
    # -----------------------------------------------------------------------

    def _sample_severity(
        self,
        rng: random.Random,
    ) -> float:
        """
        Sample continuous severity from configured range.
        """

        return round(
            rng.uniform(
                self.min_severity,
                self.max_severity,
            ),
            4,
        )

    # -----------------------------------------------------------------------
    # Result conversion
    # -----------------------------------------------------------------------

    def _make_trial(
        self,
        *,
        trial_id: str,
        repetition: int,
        scenario: ExperimentScenario,
        seed: int,
        severity: float,
        result: Any,
    ) -> RobustnessTrial:
        """
        Convert ScenarioExecutor output into normalized
        robustness-trial representation.
        """

        raw_truth_fault = self._value(
            result,
            "ground_truth_fault",
            self._fault_label(
                scenario
            ),
        )

        raw_telemetry_fault = self._value(
            result,
            "telemetry_fault",
            "UNKNOWN",
        )

        raw_multimodal_fault = self._value(
            result,
            "multimodal_fault",
            "UNKNOWN",
        )

        # ---------------------------------------------------------------
        # IMPORTANT:
        # Normalize all three labels before comparison.
        # ---------------------------------------------------------------

        truth_fault = (
            self._normalize_fault_label(
                raw_truth_fault
            )
        )

        telemetry_fault = (
            self._normalize_fault_label(
                raw_telemetry_fault
            )
        )

        multimodal_fault = (
            self._normalize_fault_label(
                raw_multimodal_fault
            )
        )

        truth_obstacle = bool(
            self._value(
                result,
                "ground_truth_obstacle",
                scenario.obstacle_present,
            )
        )

        obstacle_detected = bool(
            self._value(
                result,
                "obstacle_detected",
                False,
            )
        )

        execution_status = str(
            self._value(
                result,
                "execution_status",
                "FAILED",
            )
        ).upper()

        successful = (
            execution_status
            == "SUCCESS"
        )

        return RobustnessTrial(
            trial_id=trial_id,

            repetition=repetition,

            scenario_id=(
                scenario.scenario_id
            ),

            scenario_name=(
                scenario.name
            ),

            seed=seed,

            fault_severity=severity,

            ground_truth_fault=truth_fault,

            ground_truth_obstacle=truth_obstacle,

            requested_vision_quality=(
                str(
                    self._value(
                        result,
                        "requested_vision_quality",
                        self._vision_quality_label(
                            scenario
                        ),
                    )
                )
            ),

            telemetry_fault=(
                telemetry_fault
            ),

            telemetry_confidence=(
                self._safe_float(
                    self._value(
                        result,
                        "telemetry_confidence",
                        0.0,
                    )
                )
            ),

            multimodal_fault=(
                multimodal_fault
            ),

            multimodal_confidence=(
                self._safe_float(
                    self._value(
                        result,
                        "multimodal_confidence",
                        0.0,
                    )
                )
            ),

            fusion_mode=str(
                self._value(
                    result,
                    "fusion_mode",
                    "UNKNOWN",
                )
            ),

            obstacle_detected=(
                obstacle_detected
            ),

            execution_status=(
                execution_status
            ),

            execution_message=str(
                self._value(
                    result,
                    "execution_message",
                    "",
                )
            ),

            total_latency_seconds=(
                self._safe_float(
                    self._value(
                        result,
                        "total_latency_seconds",
                        0.0,
                    )
                )
            ),

            # -----------------------------------------------------------
            # Correct normalized comparisons
            # -----------------------------------------------------------

            telemetry_correct=(
                successful
                and telemetry_fault
                == truth_fault
            ),

            multimodal_correct=(
                successful
                and multimodal_fault
                == truth_fault
            ),

            obstacle_detection_correct=(
                None
                if not successful
                else (
                    obstacle_detected
                    == truth_obstacle
                )
            ),
        )

    # -----------------------------------------------------------------------
    # Main experiment
    # -----------------------------------------------------------------------

    def run(
        self,
    ) -> dict[str, Any]:
        """
        Execute the complete randomized robustness study.
        """

        scenarios = list(
            build_initial_scenarios()
        )

        if not scenarios:
            raise RuntimeError(
                "build_initial_scenarios() "
                "returned no scenarios."
            )

        rng = random.Random(
            self.base_seed
        )

        trials: list[
            RobustnessTrial
        ] = []

        expected_trials = (
            len(scenarios)
            * self.repetitions
        )

        LOGGER.info(
            "Randomized robustness study "
            "started: %d scenarios x "
            "%d repetitions = %d trials.",
            len(scenarios),
            self.repetitions,
            expected_trials,
        )

        study_start = (
            time.perf_counter()
        )

        counter = 0

        for repetition in range(
            1,
            self.repetitions + 1,
        ):

            for (
                scenario_index,
                base_scenario,
            ) in enumerate(
                scenarios,
                start=1,
            ):

                counter += 1

                severity = (
                    self._sample_severity(
                        rng
                    )
                )

                seed = (
                    self._trial_seed(
                        self.base_seed,
                        repetition,
                        scenario_index,
                    )
                )

                trial_id = (
                    f"R{repetition:02d}-"
                    f"{base_scenario.scenario_id}"
                )

                LOGGER.info(
                    "Trial %s (%d/%d) | "
                    "severity=%.4f | seed=%d",
                    trial_id,
                    counter,
                    expected_trials,
                    severity,
                    seed,
                )

                # -------------------------------------------------------
                # Preserve existing scenario configuration.
                # Only seed is replaced.
                # -------------------------------------------------------

                scenario = replace(
                    base_scenario,
                    seed=seed,
                )

                try:

                    executor = (
                        ScenarioExecutor(
                            joint_index=(
                                self.joint_index
                            ),
                            fault_severity=(
                                severity
                            ),
                            settling_steps=(
                                self.settling_steps
                            ),
                            output_directory=(
                                Path(
                                    "data"
                                )
                                / "experiments"
                            ),
                        )
                    )

                    result = (
                        executor.execute(
                            scenario
                        )
                    )

                    trial = (
                        self._make_trial(
                            trial_id=trial_id,
                            repetition=repetition,
                            scenario=scenario,
                            seed=seed,
                            severity=severity,
                            result=result,
                        )
                    )

                except Exception as exc:

                    LOGGER.exception(
                        "Trial %s failed.",
                        trial_id,
                    )

                    trial = (
                        RobustnessTrial(
                            trial_id=trial_id,

                            repetition=repetition,

                            scenario_id=(
                                base_scenario.scenario_id
                            ),

                            scenario_name=(
                                base_scenario.name
                            ),

                            seed=seed,

                            fault_severity=severity,

                            ground_truth_fault=(
                                self._fault_label(
                                    base_scenario
                                )
                            ),

                            ground_truth_obstacle=bool(
                                base_scenario.obstacle_present
                            ),

                            requested_vision_quality=(
                                self._vision_quality_label(
                                    base_scenario
                                )
                            ),

                            telemetry_fault=(
                                "ERROR"
                            ),

                            telemetry_confidence=(
                                0.0
                            ),

                            multimodal_fault=(
                                "ERROR"
                            ),

                            multimodal_confidence=(
                                0.0
                            ),

                            fusion_mode=(
                                "ERROR"
                            ),

                            obstacle_detected=(
                                False
                            ),

                            execution_status=(
                                "FAILED"
                            ),

                            execution_message=(
                                str(exc)
                            ),

                            total_latency_seconds=(
                                0.0
                            ),

                            telemetry_correct=(
                                False
                            ),

                            multimodal_correct=(
                                False
                            ),

                            obstacle_detection_correct=(
                                None
                            ),
                        )
                    )

                trials.append(
                    trial
                )

        runtime_seconds = (
            time.perf_counter()
            - study_start
        )

        summary = (
            self._build_summary(
                trials
            )
        )

        payload = {
            "study": {
                "name": (
                    "RoboTwin-AI Randomized "
                    "Severity Robustness Evaluation"
                ),

                "study_type": (
                    "randomized_severity_robustness"
                ),

                "timestamp_utc": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),

                "base_seed": (
                    self.base_seed
                ),

                "scenario_count": (
                    len(scenarios)
                ),

                "repetition_count": (
                    self.repetitions
                ),

                "total_trials": (
                    len(trials)
                ),

                "severity_range": {
                    "minimum": (
                        self.min_severity
                    ),
                    "maximum": (
                        self.max_severity
                    ),
                },

                "joint_index": (
                    self.joint_index
                ),

                "settling_steps": (
                    self.settling_steps
                ),

                "runtime_seconds": round(
                    runtime_seconds,
                    6,
                ),

                "methodology": [
                    (
                        "Existing "
                        "ScenarioExecutor "
                        "is reused."
                    ),

                    (
                        "Fault severity is "
                        "the randomized "
                        "experimental factor."
                    ),

                    (
                        "Each trial receives "
                        "a unique framework "
                        "seed."
                    ),

                    (
                        "Ground truth comes "
                        "from "
                        "ExperimentScenario."
                    ),

                    (
                        "Telemetry diagnosis "
                        "comes from the "
                        "existing "
                        "DiagnosticBrain."
                    ),

                    (
                        "Multimodal diagnosis "
                        "comes from the "
                        "existing "
                        "MultimodalDiagnosticBrain."
                    ),

                    (
                        "Vision-only actuator "
                        "fault classification "
                        "is not fabricated."
                    ),

                    (
                        "Fault labels are "
                        "normalized before "
                        "metric comparison."
                    ),

                    (
                        "PyBullet bit-for-bit "
                        "determinism is not "
                        "claimed."
                    ),
                ],

                "limitations": [
                    (
                        "Sensor noise is "
                        "not randomized yet."
                    ),

                    (
                        "Actuator parameters "
                        "are not randomized yet."
                    ),

                    (
                        "Mass, friction, "
                        "payload, and "
                        "environment are "
                        "not randomized yet."
                    ),

                    (
                        "Camera blur and "
                        "occlusion are not "
                        "randomized in this "
                        "experiment."
                    ),

                    (
                        "This is a "
                        "severity-robustness "
                        "study, not the "
                        "final robustness "
                        "benchmark."
                    ),
                ],
            },

            "summary": asdict(
                summary
            ),

            "trials": [
                asdict(
                    trial
                )
                for trial in trials
            ],
        }

        self._save(
            payload
        )

        self._print_summary(
            summary
        )

        return payload

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_summary(
        trials: list[
            RobustnessTrial
        ],
    ) -> RobustnessSummary:
        """
        Calculate aggregate metrics.
        """

        total = len(
            trials
        )

        successful_trials = [
            trial
            for trial in trials
            if trial.execution_status
            == "SUCCESS"
        ]

        successful = len(
            successful_trials
        )

        failed = (
            total
            - successful
        )

        telemetry_correct_count = sum(
            1
            for trial in successful_trials
            if trial.telemetry_correct
        )

        multimodal_correct_count = sum(
            1
            for trial in successful_trials
            if trial.multimodal_correct
        )

        telemetry_accuracy = (
            telemetry_correct_count
            / successful
            if successful
            else 0.0
        )

        multimodal_accuracy = (
            multimodal_correct_count
            / successful
            if successful
            else 0.0
        )

        obstacle_trials = [
            trial
            for trial in successful_trials
            if (
                trial.obstacle_detection_correct
                is not None
            )
        ]

        obstacle_correct_count = sum(
            1
            for trial in obstacle_trials
            if trial.obstacle_detection_correct
        )

        obstacle_accuracy = (
            obstacle_correct_count
            / len(
                obstacle_trials
            )
            if obstacle_trials
            else 0.0
        )

        severities = [
            trial.fault_severity
            for trial in trials
        ]

        return RobustnessSummary(
            total_trials=total,

            successful_trials=successful,

            failed_trials=failed,

            telemetry_accuracy=round(
                telemetry_accuracy,
                4,
            ),

            multimodal_accuracy=round(
                multimodal_accuracy,
                4,
            ),

            obstacle_evaluable_trials=(
                len(
                    obstacle_trials
                )
            ),

            obstacle_detection_accuracy=round(
                obstacle_accuracy,
                4,
            ),

            minimum_severity=round(
                min(severities)
                if severities
                else 0.0,
                4,
            ),

            maximum_severity=round(
                max(severities)
                if severities
                else 0.0,
                4,
            ),

            mean_severity=round(
                mean(severities)
                if severities
                else 0.0,
                4,
            ),

            scenario_count=len(
                {
                    trial.scenario_id
                    for trial in trials
                }
            ),

            repetition_count=len(
                {
                    trial.repetition
                    for trial in trials
                }
            ),

            study_type=(
                "randomized_severity_robustness"
            ),

            interpretation=(
                "Controlled pilot robustness "
                "evidence across randomized "
                "fault severity. The experiment "
                "does not represent the final "
                "robustness benchmark because "
                "sensor noise, physical parameter "
                "variation, and broader operating "
                "condition randomization are not "
                "included."
            ),
        )

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def _save(
        self,
        payload: dict[str, Any],
    ) -> None:
        """
        Save complete experiment results.
        """

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.output_file.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
            )

        LOGGER.info(
            "Robustness results saved to: %s",
            self.output_file,
        )

    # -----------------------------------------------------------------------
    # Console report
    # -----------------------------------------------------------------------

    @staticmethod
    def _print_summary(
        summary: RobustnessSummary,
    ) -> None:
        """
        Print final study summary.
        """

        print()

        print(
            "ROBOTWIN-AI RANDOMIZED "
            "ROBUSTNESS EVALUATION"
        )

        print(
            "=" * 88
        )

        print(
            f"Total trials              : "
            f"{summary.total_trials}"
        )

        print(
            f"Successful trials         : "
            f"{summary.successful_trials}"
        )

        print(
            f"Failed trials             : "
            f"{summary.failed_trials}"
        )

        print(
            f"Telemetry accuracy        : "
            f"{summary.telemetry_accuracy:.4f}"
        )

        print(
            f"Multimodal accuracy       : "
            f"{summary.multimodal_accuracy:.4f}"
        )

        print(
            f"Obstacle accuracy         : "
            f"{summary.obstacle_detection_accuracy:.4f}"
        )

        print(
            f"Severity range            : "
            f"{summary.minimum_severity:.4f} - "
            f"{summary.maximum_severity:.4f}"
        )

        print(
            f"Mean severity             : "
            f"{summary.mean_severity:.4f}"
        )

        print(
            "=" * 88
        )

        print(
            "Study type                : "
            f"{summary.study_type}"
        )

        print(
            "Interpretation            : "
            "controlled severity robustness "
            "pilot"
        )

        print(
            "=" * 88
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run randomized robustness evaluation.
    """

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    evaluator = (
        RandomizedRobustnessEvaluator()
    )

    evaluator.run()


if __name__ == "__main__":
    main()