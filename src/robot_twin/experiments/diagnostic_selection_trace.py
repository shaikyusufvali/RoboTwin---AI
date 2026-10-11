
"""
RoboTwin-AI Diagnostic Selection Forensic Trace
================================================

Purpose
-------
Investigate the exact internal decision path that causes known
MOTOR_RESISTANCE trials to be classified as POSITION_DRIFT.

Known failures from the randomized robustness study:

    R02-S02
    R02-S05
    R02-S06
    R03-S06

IMPORTANT
---------
This module is analysis-only.

It does NOT:
    - modify DiagnosticBrain source code
    - monkey-patch DiagnosticBrain methods
    - replace methods
    - alter method signatures
    - modify FaultEvidenceEngine
    - modify FaultInjector
    - modify ScenarioExecutor

Instead it uses Python's sys.settrace() mechanism to observe:

    CALL
    RETURN
    EXCEPTION

events from the existing DiagnosticBrain implementation.

This avoids the method-binding problem that can occur when wrapping
instance methods dynamically.

The objective is forensic evidence, not a production diagnostic change.
"""

from __future__ import annotations

import json
import logging
import math
import random
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from robot_twin.diagnostics.diagnostic_brain import (
    DiagnosticBrain,
)

from robot_twin.experiments.experiment_runner import (
    ExperimentScenario,
    build_initial_scenarios,
)

from robot_twin.experiments.scenario_executor import (
    ScenarioExecutor,
)


LOGGER = logging.getLogger(__name__)


# ============================================================================
# PATHS
# ============================================================================

ROBUSTNESS_RESULTS_FILE = (
    Path("data")
    / "experiments"
    / "robustness"
    / "randomized_robustness_results.json"
)

PREVIOUS_TRACE_FILE = (
    Path("data")
    / "experiments"
    / "robustness"
    / "diagnostic_evidence_trace.json"
)

OUTPUT_FILE = (
    Path("data")
    / "experiments"
    / "robustness"
    / "diagnostic_selection_trace.json"
)


# ============================================================================
# KNOWN FAILURES
# ============================================================================

KNOWN_FAILURE_IDS = (
    "R02-S02",
    "R02-S05",
    "R02-S06",
    "R03-S06",
)


# ============================================================================
# DIAGNOSTIC METHODS TO TRACE
# ============================================================================

TARGET_METHODS = {
    "diagnose",
    "_select_fault_type",
    "_analyze_fault_evidence",
    "_infer_fault_type_from_reasons",
    "_predict_fault",
}


# ============================================================================
# SAFE SERIALIZATION
# ============================================================================

def safe_json_value(
    value: Any,
    depth: int = 0,
) -> Any:
    """
    Convert arbitrary Python objects into JSON-safe structures.

    The serializer is intentionally conservative because DiagnosticBrain
    locals can contain complex objects.
    """

    if depth > 8:
        return "<max-depth>"

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, bool),
    ):
        return value

    if isinstance(value, float):

        if math.isfinite(value):
            return value

        return str(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):

        result: dict[str, Any] = {}

        for key, item in value.items():

            result[str(key)] = safe_json_value(
                item,
                depth + 1,
            )

        return result

    if isinstance(
        value,
        (list, tuple, set),
    ):

        return [
            safe_json_value(
                item,
                depth + 1,
            )
            for item in value
        ]

    # Dataclass support.
    try:

        if hasattr(
            value,
            "__dataclass_fields__",
        ):

            return safe_json_value(
                asdict(value),
                depth + 1,
            )

    except Exception:
        pass

    # Enum support.
    try:

        enum_value = getattr(
            value,
            "value",
            None,
        )

        if enum_value is not None:

            if isinstance(
                enum_value,
                (str, int, float, bool),
            ):

                return enum_value

    except Exception:
        pass

    # Avoid recursively serializing arbitrary large model objects.
    if hasattr(
        value,
        "__dict__",
    ):

        try:

            attributes: dict[str, Any] = {}

            for key, item in vars(value).items():

                key_text = str(key)

                if key_text.startswith("__"):
                    continue

                # Never serialize the DiagnosticBrain object itself.
                if key_text in {
                    "_classifier",
                    "classifier",
                    "_model",
                    "model",
                }:
                    continue

                attributes[key_text] = safe_json_value(
                    item,
                    depth + 1,
                )

            return {
                "__class__": (
                    f"{value.__class__.__module__}."
                    f"{value.__class__.__name__}"
                ),
                "attributes": attributes,
            }

        except Exception:
            pass

    try:
        return str(value)

    except Exception:
        return "<unserializable>"


# ============================================================================
# BASIC HELPERS
# ============================================================================

def safe_float(
    value: Any,
) -> float | None:
    """Convert a value to finite float."""

    try:
        result = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(result):
        return None

    return result


def normalize_fault_label(
    value: Any,
) -> str:
    """Normalize a diagnostic fault label."""

    if value is None:
        return "UNKNOWN"

    text = str(value).strip()

    if not text:
        return "UNKNOWN"

    return text.upper()


def first_value(
    row: dict[str, Any],
    keys: tuple[str, ...],
    default: Any = None,
) -> Any:
    """Return the first available value."""

    for key in keys:

        if key in row:
            return row[key]

    return default


def get_trial_id(
    row: dict[str, Any],
) -> str:
    """Extract trial identifier."""

    for key in (
        "trial_id",
        "id",
        "record_id",
    ):

        value = row.get(key)

        if value:
            return str(value)

    repetition = row.get(
        "repetition"
    )

    scenario_id = row.get(
        "scenario_id"
    )

    if (
        repetition is not None
        and scenario_id
    ):

        try:

            return (
                f"R{int(repetition):02d}-"
                f"{scenario_id}"
            )

        except (
            TypeError,
            ValueError,
        ):
            pass

    if scenario_id:
        return str(scenario_id)

    return "UNKNOWN"


def get_scenario_id(
    row: dict[str, Any],
) -> str | None:
    """Extract S01-S06 scenario identifier."""

    value = first_value(
        row,
        (
            "scenario_id",
            "scenario",
        ),
    )

    if value is None:
        return None

    text = str(value)

    if (
        text.startswith("R")
        and "-" in text
    ):

        return text.split(
            "-",
            1,
        )[1]

    return text


def get_seed(
    row: dict[str, Any],
) -> int | None:
    """Extract experiment seed."""

    value = first_value(
        row,
        (
            "seed",
            "scenario_seed",
            "random_seed",
        ),
    )

    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


def get_severity(
    row: dict[str, Any],
) -> float | None:
    """Extract randomized severity."""

    value = first_value(
        row,
        (
            "severity",
            "fault_severity",
            "randomized_severity",
        ),
    )

    return safe_float(value)


# ============================================================================
# JSON LOADING
# ============================================================================

def load_json(
    path: Path,
) -> Any:
    """Load JSON file."""

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def extract_trial_rows(
    payload: Any,
) -> list[dict[str, Any]]:
    """
    Extract trial records from common result structures.
    """

    if isinstance(
        payload,
        list,
    ):

        return [
            item
            for item in payload
            if isinstance(
                item,
                dict,
            )
        ]

    if not isinstance(
        payload,
        dict,
    ):

        return []

    for key in (
        "trials",
        "results",
        "trial_results",
        "records",
    ):

        candidate = payload.get(key)

        if isinstance(
            candidate,
            list,
        ):

            rows = [
                item
                for item in candidate
                if isinstance(
                    item,
                    dict,
                )
            ]

            if rows:
                return rows

    # Recursive fallback.
    rows: list[dict[str, Any]] = []

    def visit(
        value: Any,
    ) -> None:

        if isinstance(
            value,
            dict,
        ):

            if (
                "trial_id" in value
                or "scenario_id" in value
                or "ground_truth_fault" in value
            ):

                rows.append(value)

            for child in value.values():
                visit(child)

        elif isinstance(
            value,
            list,
        ):

            for child in value:
                visit(child)

    visit(payload)

    return rows


def load_known_failures() -> list[
    dict[str, Any]
]:
    """
    Load the four known failure records.
    """

    payload = load_json(
        ROBUSTNESS_RESULTS_FILE
    )

    rows = extract_trial_rows(
        payload
    )

    selected: list[
        dict[str, Any]
    ] = []

    for row in rows:

        identifier = get_trial_id(
            row
        )

        if identifier in KNOWN_FAILURE_IDS:

            selected.append(
                row
            )

    # Previous forensic file is a fallback only.
    if len(selected) < len(
        KNOWN_FAILURE_IDS
    ):

        if PREVIOUS_TRACE_FILE.exists():

            previous_payload = load_json(
                PREVIOUS_TRACE_FILE
            )

            previous_rows = (
                extract_trial_rows(
                    previous_payload
                )
            )

            existing = {
                get_trial_id(row)
                for row in selected
            }

            for row in previous_rows:

                identifier = get_trial_id(
                    row
                )

                if (
                    identifier in KNOWN_FAILURE_IDS
                    and identifier not in existing
                ):

                    selected.append(
                        row
                    )

    order = {
        identifier: index
        for index, identifier in enumerate(
            KNOWN_FAILURE_IDS
        )
    }

    selected.sort(
        key=lambda row: order.get(
            get_trial_id(row),
            999,
        )
    )

    return selected


# ============================================================================
# SCENARIO RESOLUTION
# ============================================================================

def resolve_scenario(
    scenario_id: str,
) -> ExperimentScenario | None:
    """Resolve scenario from the existing experiment definitions."""

    scenarios = build_initial_scenarios()

    for scenario in scenarios:

        if (
            str(
                scenario.scenario_id
            )
            == scenario_id
        ):

            return scenario

    return None


# ============================================================================
# TRACE RECORD
# ============================================================================

class TraceRecord:
    """
    One observed DiagnosticBrain function event.
    """

    def __init__(
        self,
        *,
        event: str,
        function_name: str,
        file_name: str,
        line_number: int,
        arguments: Any = None,
        locals_snapshot: Any = None,
        return_value: Any = None,
        exception_value: Any = None,
        elapsed_seconds: float | None = None,
    ) -> None:

        self.event = event
        self.function_name = function_name
        self.file_name = file_name
        self.line_number = line_number
        self.arguments = arguments
        self.locals_snapshot = locals_snapshot
        self.return_value = return_value
        self.exception_value = exception_value
        self.elapsed_seconds = elapsed_seconds

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "event": self.event,
            "function_name": self.function_name,
            "file_name": self.file_name,
            "line_number": self.line_number,
            "arguments": self.arguments,
            "locals_snapshot": self.locals_snapshot,
            "return_value": self.return_value,
            "exception_value": self.exception_value,
            "elapsed_seconds": self.elapsed_seconds,
        }


# ============================================================================
# SYS.SETTRACE FORENSIC TRACER
# ============================================================================

class DiagnosticTraceCollector:
    """
    Observe DiagnosticBrain execution without replacing methods.

    This is the critical difference from the previous implementation.

    No wrapper functions are installed on DiagnosticBrain.

    Python itself reports:
        call
        return
        exception

    events to this collector.
    """

    def __init__(self) -> None:

        self.records: list[
            TraceRecord
        ] = []

        self._start_times: dict[
            int,
            float,
        ] = {}

        self._previous_trace = None

        self._active = False

    # ----------------------------------------------------------------------
    # Frame filtering
    # ----------------------------------------------------------------------

    @staticmethod
    def _is_target_frame(
        frame: Any,
    ) -> bool:
        """
        Return True only for DiagnosticBrain methods we care about.
        """

        code = frame.f_code

        function_name = code.co_name

        if function_name not in TARGET_METHODS:
            return False

        filename = str(
            code.co_filename
        ).replace(
            "\\",
            "/",
        )

        return (
            filename.endswith(
                "/robot_twin/diagnostics/"
                "diagnostic_brain.py"
            )
        )

    # ----------------------------------------------------------------------
    # Argument snapshot
    # ----------------------------------------------------------------------

    @staticmethod
    def _capture_arguments(
        frame: Any,
    ) -> dict[str, Any]:
        """
        Capture function arguments from the frame.

        The self object is intentionally excluded.
        """

        arguments: dict[str, Any] = {}

        local_variables = frame.f_locals

        argument_count = (
            frame.f_code.co_argcount
        )

        argument_names = (
            frame.f_code.co_varnames[
                :argument_count
            ]
        )

        for name in argument_names:

            if name == "self":
                continue

            if name not in local_variables:
                continue

            try:

                arguments[name] = (
                    safe_json_value(
                        local_variables[name]
                    )
                )

            except Exception:

                arguments[name] = (
                    "<capture-error>"
                )

        return arguments

    # ----------------------------------------------------------------------
    # Local snapshot
    # ----------------------------------------------------------------------

    @staticmethod
    def _capture_locals(
        frame: Any,
    ) -> dict[str, Any]:
        """
        Capture useful local variables.

        self and private implementation objects are excluded.
        """

        snapshot: dict[str, Any] = {}

        for name, value in frame.f_locals.items():

            if name == "self":
                continue

            # Skip very large/internal objects.
            if name in {
                "classifier",
                "model",
                "camera",
                "vision_observation",
                "telemetry",
            }:

                # Telemetry is important, so keep it.
                if name != "telemetry":
                    continue

            try:

                snapshot[name] = (
                    safe_json_value(
                        value
                    )
                )

            except Exception:

                snapshot[name] = (
                    "<capture-error>"
                )

        return snapshot

    # ----------------------------------------------------------------------
    # Trace callback
    # ----------------------------------------------------------------------

    def trace(
        self,
        frame: Any,
        event: str,
        arg: Any,
    ) -> Any:
        """
        Main sys.settrace callback.
        """

        if event == "call":

            if not self._is_target_frame(
                frame
            ):

                return self.trace

            frame_id = id(frame)

            self._start_times[
                frame_id
            ] = time.perf_counter()

            self.records.append(
                TraceRecord(
                    event="call",
                    function_name=frame.f_code.co_name,
                    file_name=frame.f_code.co_filename,
                    line_number=frame.f_lineno,
                    arguments=(
                        self._capture_arguments(
                            frame
                        )
                    ),
                )
            )

            return self.trace

        # Only process return/exception for tracked target frames.
        if not self._is_target_frame(
            frame
        ):

            return self.trace

        frame_id = id(frame)

        started = self._start_times.pop(
            frame_id,
            None,
        )

        elapsed = None

        if started is not None:

            elapsed = (
                time.perf_counter()
                - started
            )

        if event == "return":

            self.records.append(
                TraceRecord(
                    event="return",
                    function_name=frame.f_code.co_name,
                    file_name=frame.f_code.co_filename,
                    line_number=frame.f_lineno,
                    locals_snapshot=(
                        self._capture_locals(
                            frame
                        )
                    ),
                    return_value=(
                        safe_json_value(
                            arg
                        )
                    ),
                    elapsed_seconds=elapsed,
                )
            )

        elif event == "exception":

            exception_value = None

            try:

                if (
                    isinstance(
                        arg,
                        tuple,
                    )
                    and len(arg) >= 2
                ):

                    exception_type = (
                        arg[0].__name__
                        if hasattr(
                            arg[0],
                            "__name__",
                        )
                        else str(arg[0])
                    )

                    exception_value = (
                        f"{exception_type}: "
                        f"{arg[1]}"
                    )

                else:

                    exception_value = (
                        safe_json_value(
                            arg
                        )
                    )

            except Exception:

                exception_value = (
                    "<exception-capture-error>"
                )

            self.records.append(
                TraceRecord(
                    event="exception",
                    function_name=frame.f_code.co_name,
                    file_name=frame.f_code.co_filename,
                    line_number=frame.f_lineno,
                    locals_snapshot=(
                        self._capture_locals(
                            frame
                        )
                    ),
                    exception_value=(
                        exception_value
                    ),
                    elapsed_seconds=elapsed,
                )
            )

        return self.trace

    # ----------------------------------------------------------------------
    # Install
    # ----------------------------------------------------------------------

    def start(self) -> None:
        """Start tracing."""

        if self._active:
            return

        self._previous_trace = (
            sys.gettrace()
        )

        sys.settrace(
            self.trace
        )

        self._active = True

    # ----------------------------------------------------------------------
    # Stop
    # ----------------------------------------------------------------------

    def stop(self) -> None:
        """Stop tracing and restore previous trace."""

        if not self._active:
            return

        sys.settrace(
            self._previous_trace
        )

        self._previous_trace = None
        self._active = False

    # ----------------------------------------------------------------------
    # Context manager
    # ----------------------------------------------------------------------

    def __enter__(
        self,
    ) -> "DiagnosticTraceCollector":

        self.start()

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:

        self.stop()


# ============================================================================
# TRACE EXTRACTION
# ============================================================================

def records_for_function(
    records: list[TraceRecord],
    function_name: str,
) -> list[dict[str, Any]]:
    """Return all records for one DiagnosticBrain method."""

    return [
        record.to_dict()
        for record in records
        if record.function_name
        == function_name
    ]


def latest_return(
    records: list[TraceRecord],
    function_name: str,
) -> dict[str, Any] | None:
    """Return latest return event."""

    candidates = [
        record.to_dict()
        for record in records
        if (
            record.function_name
            == function_name
            and record.event
            == "return"
        )
    ]

    if not candidates:
        return None

    return candidates[-1]


def latest_exception(
    records: list[TraceRecord],
    function_name: str,
) -> dict[str, Any] | None:
    """Return latest exception event."""

    candidates = [
        record.to_dict()
        for record in records
        if (
            record.function_name
            == function_name
            and record.event
            == "exception"
        )
    ]

    if not candidates:
        return None

    return candidates[-1]


def extract_return_fault(
    record: dict[str, Any] | None,
) -> str | None:
    """
    Extract fault label from a return record.
    """

    if record is None:
        return None

    value = record.get(
        "return_value"
    )

    if value is None:
        return None

    if isinstance(
        value,
        str,
    ):

        return value

    if isinstance(
        value,
        dict,
    ):

        for key in (
            "fault_type",
            "predicted_fault",
            "prediction",
            "selected_fault",
        ):

            if key in value:

                return str(
                    value[key]
                )

        attributes = value.get(
            "attributes"
        )

        if isinstance(
            attributes,
            dict,
        ):

            for key in (
                "fault_type",
                "predicted_fault",
                "prediction",
                "selected_fault",
            ):

                if key in attributes:

                    return str(
                        attributes[key]
                    )

    return None


# ============================================================================
# REPLAY
# ============================================================================

def replay_trial(
    row: dict[str, Any],
) -> dict[str, Any]:
    """
    Replay one known failure using sys.settrace().
    """

    identifier = get_trial_id(
        row
    )

    scenario_id = get_scenario_id(
        row
    )

    seed = get_seed(
        row
    )

    severity = get_severity(
        row
    )

    if scenario_id is None:

        raise ValueError(
            f"{identifier}: missing scenario_id."
        )

    if seed is None:

        raise ValueError(
            f"{identifier}: missing seed."
        )

    if severity is None:

        raise ValueError(
            f"{identifier}: missing severity."
        )

    base_scenario = resolve_scenario(
        scenario_id
    )

    if base_scenario is None:

        raise ValueError(
            f"{identifier}: scenario "
            f"{scenario_id} was not found."
        )

    scenario = replace(
        base_scenario,
        seed=seed,
    )

    executor = ScenarioExecutor(
        joint_index=2,
        fault_severity=severity,
        settling_steps=240,
        output_directory=(
            Path("data")
            / "experiments"
        ),
    )

    random.seed(
        seed
    )

    collector = (
        DiagnosticTraceCollector()
    )

    replay_result: Any = None
    replay_error: str | None = None

    started = time.perf_counter()

    try:

        # IMPORTANT:
        # No DiagnosticBrain method is replaced.
        # Python only observes execution.
        with collector:

            replay_result = (
                executor.execute(
                    scenario
                )
            )

    except Exception as exc:

        replay_error = (
            f"{exc.__class__.__name__}: "
            f"{exc}"
        )

        LOGGER.exception(
            "Replay exception for %s",
            identifier,
        )

    elapsed = (
        time.perf_counter()
        - started
    )

    selection_return = (
        latest_return(
            collector.records,
            "_select_fault_type",
        )
    )

    evidence_return = (
        latest_return(
            collector.records,
            "_analyze_fault_evidence",
        )
    )

    inference_return = (
        latest_return(
            collector.records,
            "_infer_fault_type_from_reasons",
        )
    )

    prediction_return = (
        latest_return(
            collector.records,
            "_predict_fault",
        )
    )

    diagnose_return = (
        latest_return(
            collector.records,
            "diagnose",
        )
    )

    return {
        "trial_id": identifier,
        "scenario_id": scenario_id,
        "seed": seed,
        "fault_severity": severity,

        "ground_truth_fault": normalize_fault_label(
            first_value(
                row,
                (
                    "ground_truth_fault",
                    "truth",
                    "fault_type",
                ),
            )
        ),

        "recorded_prediction": normalize_fault_label(
            first_value(
                row,
                (
                    "multimodal_fault",
                    "telemetry_fault",
                    "prediction",
                    "predicted_fault",
                ),
            )
        ),

        "replay_status": (
            "SUCCESS"
            if replay_error is None
            else "FAILED"
        ),

        "replay_error": replay_error,
        "replay_elapsed_seconds": elapsed,

        "returns": {
            "diagnose": diagnose_return,
            "select_fault_type": selection_return,
            "analyze_fault_evidence": evidence_return,
            "infer_fault_type_from_reasons": (
                inference_return
            ),
            "predict_fault": prediction_return,
        },

        "extracted_outputs": {
            "diagnose": extract_return_fault(
                diagnose_return
            ),
            "select_fault_type": (
                extract_return_fault(
                    selection_return
                )
            ),
            "analyze_fault_evidence": (
                extract_return_fault(
                    evidence_return
                )
            ),
            "infer_fault_type_from_reasons": (
                extract_return_fault(
                    inference_return
                )
            ),
            "predict_fault": (
                extract_return_fault(
                    prediction_return
                )
            ),
        },

        "call_trace": [
            record.to_dict()
            for record in collector.records
        ],

        "function_call_counts": {
            method_name: len(
                records_for_function(
                    collector.records,
                    method_name,
                )
            )
            for method_name in sorted(
                TARGET_METHODS
            )
        },

        "replay_result": safe_json_value(
            replay_result
        ),
    }


# ============================================================================
# FORENSIC INTERPRETATION
# ============================================================================

def build_interpretation(
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Interpret the observed decision path.

    This does not modify or judge production behavior.
    It simply describes what the trace proves.
    """

    extracted = result[
        "extracted_outputs"
    ]

    selection = normalize_fault_label(
        extracted.get(
            "select_fault_type"
        )
    )

    evidence = normalize_fault_label(
        extracted.get(
            "analyze_fault_evidence"
        )
    )

    inference = normalize_fault_label(
        extracted.get(
            "infer_fault_type_from_reasons"
        )
    )

    prediction = normalize_fault_label(
        extracted.get(
            "predict_fault"
        )
    )

    findings: list[str] = []

    if prediction in {
        "UNKNOWN",
        "NORMAL",
    }:

        findings.append(
            "ML prediction did not provide a "
            "non-normal fault label."
        )

    elif prediction != "UNKNOWN":

        findings.append(
            "The ML prediction returned a "
            f"non-normal label: {prediction}."
        )

    if evidence in {
        "UNKNOWN",
        "NORMAL",
    }:

        findings.append(
            "The evidence-analysis method did not "
            "return a non-normal fault label."
        )

    elif evidence:

        findings.append(
            "The evidence-analysis method returned "
            f"{evidence}."
        )

    if inference:

        findings.append(
            "The reason-based inference method "
            f"returned {inference}."
        )

    else:

        findings.append(
            "The reason-based inference method did "
            "not expose a direct fault label."
        )

    if selection:

        findings.append(
            "The final fault-selection method "
            f"returned {selection}."
        )

    else:

        findings.append(
            "The final fault-selection method did "
            "not expose a return label."
        )

    # Identify the important disagreement.
    if (
        selection == "POSITION_DRIFT"
        and evidence != "POSITION_DRIFT"
        and inference == "POSITION_DRIFT"
    ):

        findings.append(
            "FORENSIC SIGNAL: POSITION_DRIFT appears "
            "to enter the final decision through "
            "_infer_fault_type_from_reasons(), not "
            "through _analyze_fault_evidence()."
        )

    elif (
        selection == "POSITION_DRIFT"
        and prediction == "POSITION_DRIFT"
    ):

        findings.append(
            "FORENSIC SIGNAL: POSITION_DRIFT is "
            "present in the ML prediction path."
        )

    elif (
        selection == "POSITION_DRIFT"
        and evidence == "POSITION_DRIFT"
    ):

        findings.append(
            "FORENSIC SIGNAL: POSITION_DRIFT is "
            "already present in evidence analysis."
        )

    elif (
        selection == "POSITION_DRIFT"
        and inference != "POSITION_DRIFT"
    ):

        findings.append(
            "FORENSIC SIGNAL: POSITION_DRIFT is "
            "selected despite the captured evidence "
            "and reason-inference outputs. Inspect "
            "the full select_fault_type locals."
        )

    return {
        "trial_id": result[
            "trial_id"
        ],
        "ground_truth_fault": result[
            "ground_truth_fault"
        ],
        "recorded_prediction": result[
            "recorded_prediction"
        ],
        "observed_selection": selection,
        "observed_evidence": evidence,
        "observed_inference": inference,
        "observed_ml_prediction": prediction,
        "findings": findings,
    }


# ============================================================================
# SAVE
# ============================================================================

def save_results(
    results: list[
        dict[str, Any]
    ],
) -> None:
    """Save forensic trace JSON."""

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "analysis_type": (
            "diagnostic_selection_forensic_trace"
        ),

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "production_code_modified": False,

        "diagnostic_methods_monkey_patched": False,

        "trace_mechanism": (
            "python_sys_settrace"
        ),

        "bit_for_bit_pybullet_determinism_claimed": (
            False
        ),

        "known_failure_ids": list(
            KNOWN_FAILURE_IDS
        ),

        "target_methods": sorted(
            TARGET_METHODS
        ),

        "source_files": {
            "robustness_results": str(
                ROBUSTNESS_RESULTS_FILE
            ),
            "previous_trace": str(
                PREVIOUS_TRACE_FILE
            ),
        },

        "results": results,
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


# ============================================================================
# CONSOLE REPORT
# ============================================================================

def print_result(
    result: dict[str, Any],
) -> None:
    """Print forensic result."""

    print()
    print(
        f"TRIAL: {result['trial_id']}"
    )
    print(
        "-" * 90
    )

    print(
        f"Truth              : "
        f"{result['ground_truth_fault']}"
    )

    print(
        f"Recorded prediction: "
        f"{result['recorded_prediction']}"
    )

    print(
        f"Replay status      : "
        f"{result['replay_status']}"
    )

    print(
        f"Severity           : "
        f"{result['fault_severity']:.4f}"
    )

    extracted = result[
        "extracted_outputs"
    ]

    print()
    print(
        "OBSERVED DECISION PATH"
    )
    print(
        "-" * 90
    )

    print(
        "ML prediction      : "
        f"{extracted['predict_fault']}"
    )

    print(
        "Evidence analysis  : "
        f"{extracted['analyze_fault_evidence']}"
    )

    print(
        "Reason inference   : "
        f"{extracted['infer_fault_type_from_reasons']}"
    )

    print(
        "Fault selection    : "
        f"{extracted['select_fault_type']}"
    )

    print(
        "Final diagnose     : "
        f"{extracted['diagnose']}"
    )

    print()
    print(
        "FUNCTION CALL COUNTS"
    )
    print(
        "-" * 90
    )

    for method_name, count in (
        result[
            "function_call_counts"
        ].items()
    ):

        print(
            f"{method_name:<38}: "
            f"{count}"
        )

    interpretation = build_interpretation(
        result
    )

    print()
    print(
        "FORENSIC FINDINGS"
    )
    print(
        "-" * 90
    )

    for finding in interpretation[
        "findings"
    ]:

        print(
            f"- {finding}"
        )

    if result["replay_error"]:

        print()
        print(
            "REPLAY ERROR"
        )
        print(
            result["replay_error"]
        )


def print_final_summary(
    results: list[
        dict[str, Any]
    ],
) -> None:
    """Print final summary."""

    print()
    print(
        "=" * 90
    )

    print(
        "FINAL DIAGNOSTIC SELECTION FORENSIC SUMMARY"
    )

    print(
        "=" * 90
    )

    successful = sum(
        result[
            "replay_status"
        ] == "SUCCESS"
        for result in results
    )

    print(
        f"Targeted trials    : "
        f"{len(results)}"
    )

    print(
        f"Successful replays : "
        f"{successful}"
    )

    print()
    print(
        "DECISION PATH"
    )

    print(
        "-" * 90
    )

    for result in results:

        extracted = result[
            "extracted_outputs"
        ]

        print(
            f"{result['trial_id']}: "
            f"ML={extracted['predict_fault']} | "
            f"Evidence={extracted['analyze_fault_evidence']} | "
            f"Inference={extracted['infer_fault_type_from_reasons']} | "
            f"Selection={extracted['select_fault_type']}"
        )

    print()
    print(
        "Important:"
    )

    print(
        "  Production DiagnosticBrain was NOT modified."
    )

    print(
        "  DiagnosticBrain methods were NOT monkey-patched."
    )

    print(
        "  sys.settrace() was used only for observation."
    )

    print()
    print(
        f"Trace saved to: {OUTPUT_FILE}"
    )

    print(
        "=" * 90
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Run forensic diagnostic selection analysis."""

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
        "ROBOTWIN-AI DIAGNOSTIC SELECTION FORENSIC TRACE"
    )

    print(
        "=" * 90
    )

    print(
        "Trace mechanism : Python sys.settrace()"
    )

    print(
        "Monkey patching : DISABLED"
    )

    print(
        "Production code : NOT MODIFIED"
    )

    print(
        f"Source          : "
        f"{ROBUSTNESS_RESULTS_FILE}"
    )

    print(
        f"Output          : "
        f"{OUTPUT_FILE}"
    )

    failures = load_known_failures()

    if not failures:

        raise RuntimeError(
            "The known robustness failures "
            "could not be located."
        )

    print()
    print(
        "KNOWN FAILURE TRIALS"
    )

    print(
        "-" * 90
    )

    for row in failures:

        print(
            f"{get_trial_id(row)} | "
            f"severity={get_severity(row)} | "
            f"seed={get_seed(row)}"
        )

    results: list[
        dict[str, Any]
    ] = []

    for row in failures:

        result = replay_trial(
            row
        )

        results.append(
            result
        )

        print_result(
            result
        )

    save_results(
        results
    )

    print_final_summary(
        results
    )


if __name__ == "__main__":
    main()

