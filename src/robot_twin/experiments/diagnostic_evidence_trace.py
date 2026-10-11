"""
Targeted diagnostic evidence tracing for RoboTwin-AI.

Purpose
-------
Investigate known diagnostic misclassifications from the randomized
robustness study without changing the production diagnostic pipeline.

This module:

1. Loads the existing randomized robustness results.
2. Identifies genuine diagnostic failures.
3. Replays those exact trials using their recorded severity and seed.
4. Captures raw telemetry before/inside DiagnosticBrain.
5. Captures the diagnostic result and internal evidence.
6. Records signed position-error conventions.
7. Records vision and multimodal results.
8. Produces a structured JSON trace for scientific investigation.

Important
---------
This module is an experiment-analysis tool.

It does NOT modify:

    - DiagnosticBrain
    - FaultEvidenceEngine
    - FaultInjector
    - ScenarioExecutor
    - MultimodalDiagnosticBrain

It temporarily wraps DiagnosticBrain.diagnose() only while the
trace experiment is running and restores the original method afterward.

The production diagnostic behavior therefore remains unchanged.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any

from robot_twin.diagnostics.diagnostic_brain import DiagnosticBrain
from robot_twin.experiments.experiment_runner import (
    ExperimentScenario,
    build_initial_scenarios,
)
from robot_twin.experiments.scenario_executor import (
    ScenarioExecutionResult,
    ScenarioExecutor,
)


LOGGER = logging.getLogger(__name__)


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path.cwd()

ROBUSTNESS_RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "experiments"
    / "robustness"
    / "randomized_robustness_results.json"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "experiments"
    / "robustness"
)

OUTPUT_FILE = (
    OUTPUT_DIRECTORY
    / "diagnostic_evidence_trace.json"
)


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

JOINT_INDEX = 2
SETTLING_STEPS = 240

DEFAULT_FAULT_SEVERITY = 0.70

KNOWN_FAILURES = {
    "R02-S02",
    "R02-S05",
    "R02-S06",
    "R03-S06",
}

SUPPORTED_SCENARIO_IDS = {
    "S01",
    "S02",
    "S03",
    "S04",
    "S05",
    "S06",
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CapturedDiagnostic:
    """
    Raw diagnostic information captured from DiagnosticBrain.
    """

    telemetry: dict[str, Any]
    target_position: float | None
    diagnostic_result: dict[str, Any]


@dataclass
class TraceTrial:
    """
    Complete scientific trace for one targeted robustness trial.
    """

    trial_id: str
    scenario_id: str

    recorded_seed: int | None
    replay_seed: int | None

    recorded_severity: float
    replay_severity: float

    ground_truth_fault: str
    recorded_prediction: str

    replay_prediction: str
    replay_multimodal_prediction: str

    telemetry_correct: bool
    multimodal_correct: bool

    raw_telemetry: dict[str, Any]

    target_position: float | None

    signed_position_error_actual_minus_commanded: float | None
    signed_position_error_commanded_minus_actual: float | None

    diagnostic_result: dict[str, Any]

    diagnostic_evidence: Any

    vision_information: dict[str, Any]

    execution_result: dict[str, Any]

    replay_status: str
    replay_message: str

    created_at_utc: str


# ============================================================================
# GENERIC SERIALIZATION
# ============================================================================

def safe_float(
    value: Any,
    default: float | None = None,
) -> float | None:
    """
    Convert a value into a finite float.

    NaN and infinity are rejected because they should never be written
    into the scientific JSON output.
    """

    if value is None:
        return default

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(converted):
        return default

    return converted


def safe_int(
    value: Any,
    default: int | None = None,
) -> int | None:
    """
    Convert a value into an integer safely.
    """

    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def json_safe(
    value: Any,
) -> Any:
    """
    Convert arbitrary project objects into JSON-safe structures.

    This deliberately avoids depending on every internal dataclass
    field name in the production diagnostic stack.
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        if isinstance(value, float):
            if not math.isfinite(value):
                return None

        return value

    if isinstance(value, str):
        return value

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        try:
            return json_safe(asdict(value))
        except Exception:
            pass

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, set):
        return [
            json_safe(item)
            for item in sorted(
                value,
                key=lambda item: str(item),
            )
        ]

    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return {
                str(key): json_safe(item)
                for key, item in vars(value).items()
                if not str(key).startswith("__")
            }
        except Exception:
            pass

    return str(value)


def object_to_dict(
    value: Any,
) -> dict[str, Any]:
    """
    Convert a dataclass/object/dict into a dictionary.
    """

    if value is None:
        return {}

    if isinstance(value, dict):
        return json_safe(value)

    if is_dataclass(value):
        try:
            return json_safe(asdict(value))
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return json_safe(vars(value))
        except Exception:
            pass

    return {
        "value": json_safe(value)
    }


# ============================================================================
# LABEL NORMALIZATION
# ============================================================================

def normalize_fault_label(
    value: Any,
) -> str:
    """
    Normalize fault labels for scientific comparison.
    """

    if value is None:
        return "UNKNOWN"

    label = str(value).strip().upper()

    aliases = {
        "NORMAL": "NORMAL",
        "MOTOR_RESISTANCE": "MOTOR_RESISTANCE",
        "LOW_MOTOR_POWER": "LOW_MOTOR_POWER",
        "POSITION_DRIFT": "POSITION_DRIFT",
        "UNKNOWN": "UNKNOWN",
        "ERROR": "ERROR",
    }

    return aliases.get(
        label,
        label,
    )


# ============================================================================
# ROBUSTNESS JSON LOADING
# ============================================================================

def load_json(
    path: Path,
) -> Any:
    """
    Load a JSON file with explicit validation.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON file not found: {path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON file: {path}"
        ) from exc


def recursively_find_trial_records(
    value: Any,
) -> list[dict[str, Any]]:
    """
    Recursively locate dictionary objects that look like trial records.

    This keeps the trace tool tolerant of small schema differences between
    robustness-result versions.
    """

    records: list[dict[str, Any]] = []

    if isinstance(value, dict):

        keys = {
            str(key).lower()
            for key in value.keys()
        }

        trial_like = (
            (
                "trial_id" in keys
                or "trial" in keys
                or "trial_identifier" in keys
            )
            and (
                "severity" in keys
                or "fault_severity" in keys
            )
        )

        if trial_like:
            records.append(value)

        for child in value.values():
            records.extend(
                recursively_find_trial_records(child)
            )

    elif isinstance(value, list):

        for child in value:
            records.extend(
                recursively_find_trial_records(child)
            )

    return records


def first_value(
    record: dict[str, Any],
    keys: tuple[str, ...],
) -> Any:
    """
    Find the first available value using case-insensitive keys.
    """

    normalized = {
        str(key).lower(): value
        for key, value in record.items()
    }

    for key in keys:

        if key.lower() in normalized:
            return normalized[key.lower()]

    return None


def extract_scenario_id(
    record: dict[str, Any],
) -> str | None:
    """
    Extract scenario ID from a robustness record.
    """

    value = first_value(
        record,
        (
            "scenario_id",
            "scenario",
            "scenario_identifier",
        ),
    )

    if value is None:
        return None

    value = str(value).strip()

    if value in SUPPORTED_SCENARIO_IDS:
        return value

    for scenario_id in SUPPORTED_SCENARIO_IDS:
        if scenario_id.lower() in value.lower():
            return scenario_id

    return None


def extract_trial_id(
    record: dict[str, Any],
) -> str | None:
    """
    Extract the robustness trial identifier.
    """

    value = first_value(
        record,
        (
            "trial_id",
            "trial",
            "trial_identifier",
            "id",
        ),
    )

    if value is None:
        return None

    return str(value).strip()


def extract_severity(
    record: dict[str, Any],
) -> float | None:
    """
    Extract randomized fault severity.
    """

    value = first_value(
        record,
        (
            "severity",
            "fault_severity",
            "randomized_severity",
        ),
    )

    return safe_float(value)


def extract_seed(
    record: dict[str, Any],
) -> int | None:
    """
    Extract scenario/randomization seed.
    """

    value = first_value(
        record,
        (
            "seed",
            "scenario_seed",
            "random_seed",
        ),
    )

    return safe_int(value)


def extract_ground_truth(
    record: dict[str, Any],
) -> str:
    """
    Extract ground-truth fault label.
    """

    value = first_value(
        record,
        (
            "ground_truth_fault",
            "truth",
            "fault_type",
            "ground_truth",
        ),
    )

    return normalize_fault_label(value)


def extract_prediction(
    record: dict[str, Any],
) -> str:
    """
    Extract recorded multimodal prediction when available.
    """

    value = first_value(
        record,
        (
            "multimodal_fault",
            "prediction",
            "predicted_fault",
            "multimodal_prediction",
        ),
    )

    return normalize_fault_label(value)


def load_failed_trials() -> list[dict[str, Any]]:
    """
    Load genuine failures from the existing robustness study.

    If the JSON contains explicit correctness flags, they are used.

    Otherwise, the known failure IDs from the completed robustness study
    are used as a safe fallback.
    """

    payload = load_json(
        ROBUSTNESS_RESULTS_FILE
    )

    records = recursively_find_trial_records(
        payload
    )

    unique: dict[str, dict[str, Any]] = {}

    for record in records:

        trial_id = extract_trial_id(
            record
        )

        if trial_id is None:
            continue

        unique[trial_id] = record

    failed: list[dict[str, Any]] = []

    for trial_id, record in unique.items():

        normalized_id = trial_id.upper()

        if normalized_id in KNOWN_FAILURES:
            failed.append(record)
            continue

        ground_truth = extract_ground_truth(
            record
        )

        prediction = extract_prediction(
            record
        )

        telemetry_prediction = normalize_fault_label(
            first_value(
                record,
                (
                    "telemetry_fault",
                    "telemetry_prediction",
                ),
            )
        )

        explicit_correct = first_value(
            record,
            (
                "multimodal_correct",
                "correct",
                "is_correct",
            ),
        )

        if explicit_correct is False:
            failed.append(record)
            continue

        if (
            ground_truth != "UNKNOWN"
            and prediction != "UNKNOWN"
            and ground_truth != prediction
        ):
            failed.append(record)
            continue

        if (
            ground_truth != "UNKNOWN"
            and telemetry_prediction != "UNKNOWN"
            and ground_truth != telemetry_prediction
        ):
            failed.append(record)

    if not failed:
        LOGGER.warning(
            "No explicit failed trials were discovered. "
            "Using the four known failures from the completed "
            "robustness study."
        )

        for known_id in sorted(
            KNOWN_FAILURES
        ):
            for record in unique.values():

                trial_id = extract_trial_id(
                    record
                )

                if (
                    trial_id
                    and trial_id.upper() == known_id
                ):
                    failed.append(record)
                    break

    failed.sort(
        key=lambda record: (
            extract_trial_id(record)
            or ""
        )
    )

    return failed


# ============================================================================
# SCENARIO RESOLUTION
# ============================================================================

def resolve_base_scenario(
    scenario_id: str,
) -> ExperimentScenario:
    """
    Resolve the canonical experiment scenario from experiment_runner.py.
    """

    scenarios = build_initial_scenarios()

    for scenario in scenarios:

        if scenario.scenario_id == scenario_id:
            return scenario

    raise ValueError(
        f"Scenario {scenario_id!r} "
        "was not found in build_initial_scenarios()."
    )


# ============================================================================
# TELEMETRY EXTRACTION
# ============================================================================

def extract_telemetry_values(
    telemetry: Any,
) -> dict[str, Any]:
    """
    Extract all public telemetry fields.

    The exact JointTelemetry structure is intentionally discovered from
    the object rather than duplicated here.
    """

    data = object_to_dict(
        telemetry
    )

    cleaned: dict[str, Any] = {}

    for key, value in data.items():

        if str(key).startswith("_"):
            continue

        cleaned[str(key)] = json_safe(
            value
        )

    return cleaned


def get_telemetry_position(
    telemetry: dict[str, Any],
) -> float | None:
    """
    Get actual joint position.
    """

    value = telemetry.get(
        "position"
    )

    return safe_float(
        value
    )


def calculate_position_errors(
    telemetry: dict[str, Any],
    target_position: float | None,
) -> tuple[
    float | None,
    float | None,
]:
    """
    Calculate both signed position-error conventions.

    RoboTwin-AI currently uses:

        FaultEvidenceEngine:
            actual - commanded

        FaultClassifier training:
            commanded - actual

    Both are recorded explicitly so that sign confusion can be
    investigated scientifically.
    """

    actual_position = get_telemetry_position(
        telemetry
    )

    if (
        actual_position is None
        or target_position is None
    ):
        return (
            None,
            None,
        )

    actual_minus_commanded = (
        actual_position
        - target_position
    )

    commanded_minus_actual = (
        target_position
        - actual_position
    )

    return (
        actual_minus_commanded,
        commanded_minus_actual,
    )


# ============================================================================
# DIAGNOSTIC EVIDENCE EXTRACTION
# ============================================================================

def extract_diagnostic_evidence(
    diagnostic_result: Any,
) -> Any:
    """
    Extract the evidence section from DiagnosticResult.

    The production DiagnosticResult may evolve. This function therefore
    checks common representations instead of assuming one exact internal
    structure.
    """

    if diagnostic_result is None:
        return None

    if isinstance(
        diagnostic_result,
        dict,
    ):

        for key in (
            "evidence",
            "diagnostic_evidence",
            "fault_evidence",
        ):

            if key in diagnostic_result:
                return json_safe(
                    diagnostic_result[key]
                )

        return None

    for attribute in (
        "evidence",
        "diagnostic_evidence",
        "fault_evidence",
    ):

        if hasattr(
            diagnostic_result,
            attribute,
        ):

            try:
                return json_safe(
                    getattr(
                        diagnostic_result,
                        attribute,
                    )
                )
            except Exception:
                return None

    return None


def extract_result_fault(
    diagnostic_result: Any,
) -> str:
    """
    Extract predicted fault from DiagnosticResult.
    """

    if diagnostic_result is None:
        return "UNKNOWN"

    if isinstance(
        diagnostic_result,
        dict,
    ):

        value = first_value(
            diagnostic_result,
            (
                "fault_type",
                "predicted_fault",
                "prediction",
            ),
        )

        return normalize_fault_label(
            value
        )

    for attribute in (
        "fault_type",
        "predicted_fault",
        "prediction",
    ):

        if hasattr(
            diagnostic_result,
            attribute,
        ):

            try:
                return normalize_fault_label(
                    getattr(
                        diagnostic_result,
                        attribute,
                    )
                )
            except Exception:
                pass

    return "UNKNOWN"


def extract_result_fields(
    diagnostic_result: Any,
) -> dict[str, Any]:
    """
    Serialize the entire diagnostic result.

    This gives the investigation access to confidence, anomaly score,
    status, evidence, severity, and other fields without coupling the
    trace code to one exact DiagnosticResult implementation.
    """

    return object_to_dict(
        diagnostic_result
    )


# ============================================================================
# DIAGNOSTIC CAPTURE WRAPPER
# ============================================================================

class DiagnosticCapture:
    """
    Runtime capture object.

    DiagnosticBrain.diagnose() is wrapped temporarily during replay.
    """

    def __init__(self) -> None:

        self.latest: CapturedDiagnostic | None = None

    def wrapped_diagnose(
        self,
        telemetry: Any,
        target_position: float,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Call the real DiagnosticBrain.diagnose() while capturing
        its inputs and output.
        """

        result = (
            self._original_diagnose(
                self,
                telemetry,
                target_position,
                *args,
                **kwargs,
            )
        )

        telemetry_dict = (
            extract_telemetry_values(
                telemetry
            )
        )

        self.latest = CapturedDiagnostic(
            telemetry=telemetry_dict,
            target_position=safe_float(
                target_position
            ),
            diagnostic_result=(
                extract_result_fields(
                    result
                )
            ),
        )

        return result

    def _original_diagnose(
        self,
        brain: DiagnosticBrain,
        telemetry: Any,
        target_position: float,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Placeholder replaced when the wrapper is installed.
        """

        raise RuntimeError(
            "Original DiagnosticBrain.diagnose "
            "has not been attached."
        )


def install_diagnostic_capture(
    capture: DiagnosticCapture,
) -> Any:
    """
    Temporarily patch DiagnosticBrain.diagnose().
    """

    original_method = (
        DiagnosticBrain.diagnose
    )

    def traced_diagnose(
        self: DiagnosticBrain,
        telemetry: Any,
        target_position: float,
        *args: Any,
        **kwargs: Any,
    ) -> Any:

        result = original_method(
            self,
            telemetry,
            target_position,
            *args,
            **kwargs,
        )

        capture.latest = CapturedDiagnostic(
            telemetry=(
                extract_telemetry_values(
                    telemetry
                )
            ),
            target_position=(
                safe_float(
                    target_position
                )
            ),
            diagnostic_result=(
                extract_result_fields(
                    result
                )
            ),
        )

        return result

    DiagnosticBrain.diagnose = traced_diagnose

    return original_method


def restore_diagnostic_capture(
    original_method: Any,
) -> None:
    """
    Restore production DiagnosticBrain.diagnose().
    """

    DiagnosticBrain.diagnose = (
        original_method
    )


# ============================================================================
# EXACT TRIAL REPLAY
# ============================================================================

def replay_trial(
    record: dict[str, Any],
) -> TraceTrial:
    """
    Replay one robustness failure.

    The recorded seed is restored when available.

    This is important because the previous trace implementation
    accidentally dropped the original seed.
    """

    trial_id = (
        extract_trial_id(record)
        or "UNKNOWN_TRIAL"
    )

    scenario_id = (
        extract_scenario_id(record)
    )

    if scenario_id is None:

        raise ValueError(
            f"Could not resolve scenario ID "
            f"for trial {trial_id}."
        )

    severity = (
        extract_severity(record)
    )

    if severity is None:

        raise ValueError(
            f"Could not resolve severity "
            f"for trial {trial_id}."
        )

    recorded_seed = (
        extract_seed(record)
    )

    ground_truth = (
        extract_ground_truth(record)
    )

    recorded_prediction = (
        extract_prediction(record)
    )

    base_scenario = (
        resolve_base_scenario(
            scenario_id
        )
    )

    # ------------------------------------------------------------------
    # Preserve the exact robustness-study seed.
    #
    # dataclasses.replace() is intentionally used here.
    #
    # The previous implementation called replace() but forgot to import
    # it, producing:
    #
    #     NameError: name 'replace' is not defined
    #
    # This implementation imports it correctly and uses it for the
    # scientific purpose it was intended for.
    # ------------------------------------------------------------------

    if recorded_seed is not None:

        scenario = replace(
            base_scenario,
            seed=recorded_seed,
        )

    else:

        scenario = base_scenario

    replay_seed = (
        scenario.seed
    )

    # ------------------------------------------------------------------
    # Seed Python-level randomness.
    #
    # PyBullet bit-for-bit determinism is NOT claimed.
    # ------------------------------------------------------------------

    random.seed(
        replay_seed
    )

    capture = (
        DiagnosticCapture()
    )

    original_diagnose = (
        install_diagnostic_capture(
            capture
        )
    )

    executor = ScenarioExecutor(
        joint_index=JOINT_INDEX,
        fault_severity=severity,
        settling_steps=SETTLING_STEPS,
        output_directory=(
            PROJECT_ROOT
            / "data"
            / "experiments"
        ),
    )

    start_time = (
        time.perf_counter()
    )

    try:

        LOGGER.info(
            "Replaying %s | scenario=%s | "
            "severity=%.4f | seed=%s",
            trial_id,
            scenario_id,
            severity,
            replay_seed,
        )

        execution_result = (
            executor.execute(
                scenario
            )
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        execution_dict = (
            object_to_dict(
                execution_result
            )
        )

        if capture.latest is None:

            raise RuntimeError(
                "Diagnostic capture did not receive "
                "a DiagnosticBrain.diagnose() call."
            )

        raw_telemetry = (
            capture.latest.telemetry
        )

        target_position = (
            capture.latest.target_position
        )

        replay_diagnostic = (
            capture.latest.diagnostic_result
        )

        replay_prediction = (
            extract_result_fault(
                replay_diagnostic
            )
        )

        multimodal_prediction = (
            normalize_fault_label(
                execution_dict.get(
                    "multimodal_fault"
                )
            )
        )

        telemetry_correct = (
            normalize_fault_label(
                ground_truth
            )
            == normalize_fault_label(
                replay_prediction
            )
        )

        multimodal_correct = (
            normalize_fault_label(
                ground_truth
            )
            == multimodal_prediction
        )

        (
            actual_minus_commanded,
            commanded_minus_actual,
        ) = calculate_position_errors(
            raw_telemetry,
            target_position,
        )

        evidence = (
            extract_diagnostic_evidence(
                replay_diagnostic
            )
        )

        vision_information = {
            "vision_available": (
                execution_dict.get(
                    "vision_available"
                )
            ),
            "vision_score": (
                execution_dict.get(
                    "vision_score"
                )
            ),
            "vision_reliability": (
                execution_dict.get(
                    "vision_reliability"
                )
            ),
            "obstacle_detected": (
                execution_dict.get(
                    "obstacle_detected"
                )
            ),
            "obstacle_distance": (
                execution_dict.get(
                    "obstacle_distance"
                )
            ),
            "requested_vision_quality": (
                execution_dict.get(
                    "requested_vision_quality"
                )
            ),
        }

        return TraceTrial(
            trial_id=trial_id,
            scenario_id=scenario_id,
            recorded_seed=recorded_seed,
            replay_seed=replay_seed,
            recorded_severity=severity,
            replay_severity=safe_float(
                execution_dict.get(
                    "fault_severity"
                ),
                severity,
            ) or severity,
            ground_truth_fault=(
                ground_truth
            ),
            recorded_prediction=(
                recorded_prediction
            ),
            replay_prediction=(
                replay_prediction
            ),
            replay_multimodal_prediction=(
                multimodal_prediction
            ),
            telemetry_correct=(
                telemetry_correct
            ),
            multimodal_correct=(
                multimodal_correct
            ),
            raw_telemetry=(
                raw_telemetry
            ),
            target_position=(
                target_position
            ),
            signed_position_error_actual_minus_commanded=(
                actual_minus_commanded
            ),
            signed_position_error_commanded_minus_actual=(
                commanded_minus_actual
            ),
            diagnostic_result=(
                replay_diagnostic
            ),
            diagnostic_evidence=(
                evidence
            ),
            vision_information=(
                vision_information
            ),
            execution_result=(
                execution_dict
            ),
            replay_status="SUCCESS",
            replay_message=(
                "Targeted robustness trial "
                "replayed successfully."
            ),
            created_at_utc=(
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        )

    except Exception as exc:

        LOGGER.exception(
            "Replay failed for trial %s.",
            trial_id,
        )

        execution_dict = {}

        if "execution_result" in locals():

            execution_dict = (
                object_to_dict(
                    execution_result
                )
            )

        raw_telemetry = {}

        target_position = None

        diagnostic_result = {}

        replay_prediction = "UNKNOWN"

        multimodal_prediction = normalize_fault_label(
            execution_dict.get(
                "multimodal_fault"
            )
        )

        if capture.latest is not None:

            raw_telemetry = (
                capture.latest.telemetry
            )

            target_position = (
                capture.latest.target_position
            )

            diagnostic_result = (
                capture.latest.diagnostic_result
            )

            replay_prediction = (
                extract_result_fault(
                    diagnostic_result
                )
            )

        (
            actual_minus_commanded,
            commanded_minus_actual,
        ) = calculate_position_errors(
            raw_telemetry,
            target_position,
        )

        return TraceTrial(
            trial_id=trial_id,
            scenario_id=scenario_id,
            recorded_seed=recorded_seed,
            replay_seed=replay_seed,
            recorded_severity=severity,
            replay_severity=severity,
            ground_truth_fault=ground_truth,
            recorded_prediction=recorded_prediction,
            replay_prediction=(
                replay_prediction
            ),
            replay_multimodal_prediction=(
                multimodal_prediction
            ),
            telemetry_correct=(
                normalize_fault_label(
                    ground_truth
                )
                == normalize_fault_label(
                    replay_prediction
                )
            ),
            multimodal_correct=(
                normalize_fault_label(
                    ground_truth
                )
                == normalize_fault_label(
                    multimodal_prediction
                )
            ),
            raw_telemetry=(
                raw_telemetry
            ),
            target_position=(
                target_position
            ),
            signed_position_error_actual_minus_commanded=(
                actual_minus_commanded
            ),
            signed_position_error_commanded_minus_actual=(
                commanded_minus_actual
            ),
            diagnostic_result=(
                diagnostic_result
            ),
            diagnostic_evidence=(
                extract_diagnostic_evidence(
                    diagnostic_result
                )
            ),
            vision_information={
                "replay_failed": True,
            },
            execution_result=(
                execution_dict
            ),
            replay_status="FAILED",
            replay_message=str(
                exc
            ),
            created_at_utc=(
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        )

    finally:

        # Always restore production behavior.
        restore_diagnostic_capture(
            original_diagnose
        )

        LOGGER.info(
            "Replay finished: %s",
            trial_id,
        )


# ============================================================================
# TRACE ANALYSIS
# ============================================================================

def build_summary(
    traces: list[TraceTrial],
) -> dict[str, Any]:
    """
    Build a compact scientific summary.
    """

    total = len(
        traces
    )

    successful = sum(
        1
        for trace in traces
        if trace.replay_status == "SUCCESS"
    )

    failed = (
        total
        - successful
    )

    telemetry_correct = sum(
        1
        for trace in traces
        if trace.telemetry_correct
    )

    multimodal_correct = sum(
        1
        for trace in traces
        if trace.multimodal_correct
    )

    return {
        "targeted_trials": total,
        "successful_replays": successful,
        "failed_replays": failed,
        "telemetry_correct": telemetry_correct,
        "telemetry_accuracy": (
            telemetry_correct / total
            if total
            else 0.0
        ),
        "multimodal_correct": multimodal_correct,
        "multimodal_accuracy": (
            multimodal_correct / total
            if total
            else 0.0
        ),
        "known_failure_trials": sorted(
            KNOWN_FAILURES
        ),
        "interpretation": (
            "Targeted diagnostic evidence trace. "
            "This analysis investigates previously observed "
            "robustness failures and is not itself a new "
            "robustness benchmark."
        ),
    }


def print_trace_summary(
    traces: list[TraceTrial],
) -> None:
    """
    Print human-readable results.
    """

    print()
    print(
        "ROBOTWIN-AI DIAGNOSTIC EVIDENCE TRACE"
    )
    print(
        "=" * 96
    )

    print(
        f"Targeted trials : {len(traces)}"
    )

    print(
        f"Joint           : {JOINT_INDEX}"
    )

    print(
        f"Settling steps  : {SETTLING_STEPS}"
    )

    print(
        "=" * 96
    )

    print(
        "TRIAL ANALYSIS"
    )

    print(
        "-" * 96
    )

    header = (
        f"{'TRIAL':<12}"
        f"{'SCN':<6}"
        f"{'SEV':<8}"
        f"{'TRUTH':<22}"
        f"{'RECORDED':<22}"
        f"{'REPLAY':<22}"
        f"{'STATUS':<10}"
    )

    print(
        header
    )

    print(
        "-" * 96
    )

    for trace in traces:

        print(
            f"{trace.trial_id:<12}"
            f"{trace.scenario_id:<6}"
            f"{trace.replay_severity:<8.4f}"
            f"{trace.ground_truth_fault:<22}"
            f"{trace.recorded_prediction:<22}"
            f"{trace.replay_prediction:<22}"
            f"{trace.replay_status:<10}"
        )

    print(
        "-" * 96
    )

    print()

    for trace in traces:

        print(
            f"[{trace.trial_id}]"
        )

        print(
            f"  Scenario                         : "
            f"{trace.scenario_id}"
        )

        print(
            f"  Seed                             : "
            f"{trace.replay_seed}"
        )

        print(
            f"  Severity                         : "
            f"{trace.replay_severity:.4f}"
        )

        print(
            f"  Ground truth                     : "
            f"{trace.ground_truth_fault}"
        )

        print(
            f"  Recorded prediction              : "
            f"{trace.recorded_prediction}"
        )

        print(
            f"  Replay prediction                : "
            f"{trace.replay_prediction}"
        )

        print(
            f"  Replay multimodal prediction     : "
            f"{trace.replay_multimodal_prediction}"
        )

        print(
            f"  Telemetry correct                : "
            f"{trace.telemetry_correct}"
        )

        print(
            f"  Multimodal correct               : "
            f"{trace.multimodal_correct}"
        )

        print(
            f"  Target position                  : "
            f"{trace.target_position}"
        )

        print(
            f"  Actual - commanded error         : "
            f"{trace.signed_position_error_actual_minus_commanded}"
        )

        print(
            f"  Commanded - actual error         : "
            f"{trace.signed_position_error_commanded_minus_actual}"
        )

        telemetry = (
            trace.raw_telemetry
        )

        print(
            f"  Actual position                  : "
            f"{telemetry.get('position')}"
        )

        print(
            f"  Velocity                         : "
            f"{telemetry.get('velocity')}"
        )

        print(
            f"  Force                            : "
            f"{telemetry.get('force')}"
        )

        print(
            f"  Motor current                    : "
            f"{telemetry.get('motor_current')}"
        )

        print(
            f"  Temperature                      : "
            f"{telemetry.get('temperature')}"
        )

        print(
            f"  Replay status                    : "
            f"{trace.replay_status}"
        )

        print(
            f"  Replay message                   : "
            f"{trace.replay_message}"
        )

        print()


# ============================================================================
# SAVE OUTPUT
# ============================================================================

def save_trace(
    traces: list[TraceTrial],
) -> Path:
    """
    Save complete trace results.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "experiment": (
            "RoboTwin-AI targeted "
            "diagnostic evidence trace"
        ),
        "schema_version": "2.0",
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "configuration": {
            "joint_index": JOINT_INDEX,
            "settling_steps": SETTLING_STEPS,
            "source_results_file": str(
                ROBUSTNESS_RESULTS_FILE
            ),
        },
        "scientific_notes": {
            "purpose": (
                "Investigate known randomized robustness "
                "diagnostic failures."
            ),
            "production_diagnostic_modified": False,
            "diagnostic_method_temporarily_wrapped": True,
            "recorded_seed_replayed": True,
            "pybullet_bitwise_determinism_claimed": False,
            "vision_only_fault_classifier": (
                "not_fabricated"
            ),
            "position_error_conventions": {
                "fault_evidence_engine": (
                    "actual_minus_commanded"
                ),
                "fault_classifier": (
                    "commanded_minus_actual"
                ),
            },
        },
        "summary": build_summary(
            traces
        ),
        "trials": [
            json_safe(
                asdict(trace)
            )
            for trace in traces
        ],
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return OUTPUT_FILE


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """
    Run targeted diagnostic evidence tracing.
    """

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
        "ROBOTWIN-AI DIAGNOSTIC EVIDENCE TRACE"
    )
    print(
        "=" * 96
    )

    print(
        f"Source results : "
        f"{ROBUSTNESS_RESULTS_FILE}"
    )

    print(
        f"Output file    : "
        f"{OUTPUT_FILE}"
    )

    print(
        "=" * 96
    )

    # ------------------------------------------------------------------
    # Load known failures.
    # ------------------------------------------------------------------

    failed_trials = (
        load_failed_trials()
    )

    if not failed_trials:

        raise RuntimeError(
            "No failed robustness trials were found. "
            "Nothing to trace."
        )

    print(
        f"Targeted trials : "
        f"{len(failed_trials)}"
    )

    print(
        "Known failures  : "
        + ", ".join(
            sorted(
                KNOWN_FAILURES
            )
        )
    )

    print(
        "=" * 96
    )

    traces: list[TraceTrial] = []

    # ------------------------------------------------------------------
    # Replay one trial at a time.
    # ------------------------------------------------------------------

    for index, record in enumerate(
        failed_trials,
        start=1,
    ):

        trial_id = (
            extract_trial_id(record)
            or f"TRIAL-{index}"
        )

        print()
        print(
            f"[{index}/{len(failed_trials)}] "
            f"Tracing {trial_id}"
        )

        try:

            trace = replay_trial(
                record
            )

            traces.append(
                trace
            )

        except Exception as exc:

            LOGGER.exception(
                "Unable to trace %s.",
                trial_id,
            )

            scenario_id = (
                extract_scenario_id(
                    record
                )
                or "UNKNOWN"
            )

            severity = (
                extract_severity(
                    record
                )
                or DEFAULT_FAULT_SEVERITY
            )

            seed = (
                extract_seed(
                    record
                )
            )

            ground_truth = (
                extract_ground_truth(
                    record
                )
            )

            traces.append(
                TraceTrial(
                    trial_id=trial_id,
                    scenario_id=scenario_id,
                    recorded_seed=seed,
                    replay_seed=seed,
                    recorded_severity=severity,
                    replay_severity=severity,
                    ground_truth_fault=ground_truth,
                    recorded_prediction=(
                        extract_prediction(
                            record
                        )
                    ),
                    replay_prediction="UNKNOWN",
                    replay_multimodal_prediction="UNKNOWN",
                    telemetry_correct=False,
                    multimodal_correct=False,
                    raw_telemetry={},
                    target_position=None,
                    signed_position_error_actual_minus_commanded=None,
                    signed_position_error_commanded_minus_actual=None,
                    diagnostic_result={},
                    diagnostic_evidence=None,
                    vision_information={},
                    execution_result={},
                    replay_status="FAILED",
                    replay_message=str(
                        exc
                    ),
                    created_at_utc=(
                        datetime.now(
                            timezone.utc
                        ).isoformat()
                    ),
                )
            )

    # ------------------------------------------------------------------
    # Save complete trace.
    # ------------------------------------------------------------------

    output_path = save_trace(
        traces
    )

    # ------------------------------------------------------------------
    # Console report.
    # ------------------------------------------------------------------

    print_trace_summary(
        traces
    )

    print(
        "=" * 96
    )

    print(
        "TRACE COMPLETE"
    )

    print(
        f"Results saved to: {output_path}"
    )

    print(
        "=" * 96
    )


if __name__ == "__main__":
    main()