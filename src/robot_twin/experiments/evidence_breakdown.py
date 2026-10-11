"""
RoboTwin-AI Diagnostic Evidence Breakdown
=========================================

Purpose
-------
Investigate known diagnostic misclassifications without rerunning PyBullet.

This module is intentionally analysis-only.

It does NOT:
    - modify DiagnosticBrain
    - modify FaultEvidenceEngine
    - modify FaultInjector
    - modify ScenarioExecutor
    - rerun simulation
    - change thresholds

It reconstructs the current FaultEvidenceEngine scoring rules and shows
exactly which evidence contributions are active for the four known
MOTOR_RESISTANCE -> POSITION_DRIFT failures.

Input:
    data/experiments/robustness/diagnostic_evidence_trace.json

Output:
    data/experiments/robustness/evidence_breakdown.json
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


INPUT_FILE = (
    Path("data")
    / "experiments"
    / "robustness"
    / "diagnostic_evidence_trace.json"
)

OUTPUT_FILE = (
    Path("data")
    / "experiments"
    / "robustness"
    / "evidence_breakdown.json"
)


# ---------------------------------------------------------------------------
# Existing FaultEvidenceEngine thresholds
# ---------------------------------------------------------------------------

POSITION_ERROR_THRESHOLD = 0.20
SEVERE_POSITION_ERROR_THRESHOLD = 1.00

LOW_VELOCITY_THRESHOLD = 0.50

HIGH_CURRENT_THRESHOLD = 2.50
LOW_CURRENT_THRESHOLD = 0.50

HIGH_TEMPERATURE_THRESHOLD = 45.0

HIGH_FORCE_THRESHOLD = 20.0
LOW_FORCE_THRESHOLD = 5.0

DIRECTIONAL_POSITION_ERROR_THRESHOLD = 0.20


# ---------------------------------------------------------------------------
# Existing diagnostic signature thresholds
# ---------------------------------------------------------------------------

POSITION_ERROR_SIGNATURE_LIMIT = 0.12

LOW_POWER_CURRENT_LIMIT = 3.0
LOW_POWER_FORCE_LIMIT = 10.0

RESISTANCE_CURRENT_LIMIT = 4.5
RESISTANCE_FORCE_LIMIT = 20.0

TEMPERATURE_HIGH_LIMIT = 65.0


KNOWN_FAILURES = {
    "R02-S02",
    "R02-S05",
    "R02-S06",
    "R03-S06",
}


@dataclass(frozen=True)
class ScoreContribution:
    """One individual evidence contribution."""

    rule: str
    active: bool
    contribution: float
    explanation: str


@dataclass(frozen=True)
class FaultScoreBreakdown:
    """Complete score breakdown for one candidate fault."""

    fault_type: str
    score: float
    contributions: list[ScoreContribution]


@dataclass(frozen=True)
class TrialBreakdown:
    """Complete diagnostic evidence analysis for one trial."""

    trial_id: str
    ground_truth: str
    recorded_prediction: str
    replay_prediction: str

    severity: float
    target_position: float

    position: float
    velocity: float
    force: float
    motor_current: float
    temperature: float

    actual_minus_commanded: float
    commanded_minus_actual: float

    evidence_scores: dict[str, float]
    best_evidence_fault: str

    telemetry_signature_candidates: dict[str, bool]
    telemetry_signature_notes: list[str]

    resistance_signature_score: float
    low_power_signature_score: float
    position_drift_signature_score: float

    normal_score: float

    important_findings: list[str]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Convert a value to a finite float."""

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(converted):
        return default

    return converted


def _normalise_label(value: Any) -> str:
    """Return a canonical uppercase diagnostic label."""

    text = str(value or "UNKNOWN").strip().upper()

    aliases = {
        "MOTOR RESISTANCE": "MOTOR_RESISTANCE",
        "LOW MOTOR POWER": "LOW_MOTOR_POWER",
        "POSITION DRIFT": "POSITION_DRIFT",
        "NORMAL": "NORMAL",
        "UNKNOWN": "UNKNOWN",
    }

    return aliases.get(text, text)


def _recursive_find(
    obj: Any,
    keys: set[str],
) -> Any | None:
    """
    Recursively find the first mapping containing one of the requested keys.
    """

    if isinstance(obj, dict):

        if any(key in obj for key in keys):
            return obj

        for value in obj.values():
            result = _recursive_find(value, keys)

            if result is not None:
                return result

    elif isinstance(obj, list):

        for item in obj:
            result = _recursive_find(item, keys)

            if result is not None:
                return result

    return None


def _recursive_find_value(
    obj: Any,
    keys: set[str],
) -> Any | None:
    """Recursively locate a value by key."""

    if isinstance(obj, dict):

        for key in keys:
            if key in obj:
                return obj[key]

        for value in obj.values():
            result = _recursive_find_value(
                value,
                keys,
            )

            if result is not None:
                return result

    elif isinstance(obj, list):

        for item in obj:
            result = _recursive_find_value(
                item,
                keys,
            )

            if result is not None:
                return result

    return None


def _load_trace() -> list[dict[str, Any]]:
    """Load targeted failure records from the existing trace."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Diagnostic trace not found: {INPUT_FILE}"
        )

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    if isinstance(payload, list):
        records = payload

    elif isinstance(payload, dict):

        records = None

        for key in (
            "trials",
            "records",
            "results",
            "trace",
            "failure_traces",
        ):
            candidate = payload.get(key)

            if isinstance(candidate, list):
                records = candidate
                break

        if records is None:
            records = [payload]

    else:
        raise ValueError(
            "Unsupported diagnostic trace JSON structure."
        )

    selected: list[dict[str, Any]] = []

    for record in records:

        if not isinstance(record, dict):
            continue

        scenario_id = str(
            _recursive_find_value(
                record,
                {
                    "scenario_id",
                    "trial_id",
                    "id",
                },
            )
            or ""
        )

        if scenario_id in KNOWN_FAILURES:
            selected.append(record)

    if not selected:
        raise ValueError(
            "None of the four known failure trials were found "
            "in diagnostic_evidence_trace.json."
        )

    return selected


# ---------------------------------------------------------------------------
# Telemetry extraction
# ---------------------------------------------------------------------------


def _extract_telemetry(
    record: dict[str, Any],
) -> dict[str, float]:
    """Extract telemetry fields from a trace record."""

    telemetry = _recursive_find(
        record,
        {
            "position",
            "velocity",
            "force",
            "motor_current",
            "temperature",
        },
    )

    if telemetry is None:
        raise ValueError(
            "Telemetry block not found in trace record."
        )

    return {
        "position": _safe_float(
            telemetry.get("position")
        ),
        "velocity": _safe_float(
            telemetry.get("velocity")
        ),
        "force": _safe_float(
            telemetry.get("force")
        ),
        "motor_current": _safe_float(
            telemetry.get("motor_current")
        ),
        "temperature": _safe_float(
            telemetry.get("temperature")
        ),
    }


def _extract_trial_metadata(
    record: dict[str, Any],
) -> tuple[str, str, str, float, float]:
    """Extract trial metadata."""

    trial_id = str(
        _recursive_find_value(
            record,
            {
                "trial_id",
                "scenario_id",
                "id",
            },
        )
        or "UNKNOWN"
    )

    ground_truth = _normalise_label(
        _recursive_find_value(
            record,
            {
                "ground_truth_fault",
                "ground_truth",
                "truth",
            },
        )
    )

    recorded_prediction = _normalise_label(
        _recursive_find_value(
            record,
            {
                "telemetry_fault",
                "recorded_prediction",
                "prediction",
            },
        )
    )

    replay_prediction = _normalise_label(
        _recursive_find_value(
            record,
            {
                "replay_fault",
                "replay_prediction",
            },
        )
    )

    severity = _safe_float(
        _recursive_find_value(
            record,
            {
                "fault_severity",
                "severity",
            },
        )
    )

    target_position = _safe_float(
        _recursive_find_value(
            record,
            {
                "target_position",
                "commanded_position",
            },
        ),
        default=0.8,
    )

    return (
        trial_id,
        ground_truth,
        recorded_prediction,
        replay_prediction,
        severity,
        target_position,
    )


# ---------------------------------------------------------------------------
# FaultEvidenceEngine reconstruction
# ---------------------------------------------------------------------------


def _calculate_low_power_score(
    *,
    position_error: float,
    velocity: float,
    force: float,
    motor_current: float,
    temperature: float,
) -> FaultScoreBreakdown:
    """
    Reproduce the current LOW_MOTOR_POWER evidence rules.
    """

    contributions: list[ScoreContribution] = []

    active = motor_current <= LOW_CURRENT_THRESHOLD

    contributions.append(
        ScoreContribution(
            rule="low_current",
            active=active,
            contribution=0.20 if active else 0.0,
            explanation=(
                f"current={motor_current:.6f} "
                f"<= {LOW_CURRENT_THRESHOLD:.2f}"
            ),
        )
    )

    active = abs(force) <= LOW_FORCE_THRESHOLD

    contributions.append(
        ScoreContribution(
            rule="low_force",
            active=active,
            contribution=0.20 if active else 0.0,
            explanation=(
                f"|force|={abs(force):.6f} "
                f"<= {LOW_FORCE_THRESHOLD:.2f}"
            ),
        )
    )

    active = abs(velocity) <= LOW_VELOCITY_THRESHOLD

    contributions.append(
        ScoreContribution(
            rule="low_velocity",
            active=active,
            contribution=0.15 if active else 0.0,
            explanation=(
                f"|velocity|={abs(velocity):.6f} "
                f"<= {LOW_VELOCITY_THRESHOLD:.2f}"
            ),
        )
    )

    active = abs(position_error) >= POSITION_ERROR_THRESHOLD

    contributions.append(
        ScoreContribution(
            rule="position_error",
            active=active,
            contribution=0.10 if active else 0.0,
            explanation=(
                f"|error|={abs(position_error):.6f} "
                f">= {POSITION_ERROR_THRESHOLD:.2f}"
            ),
        )
    )

    active = position_error <= -DIRECTIONAL_POSITION_ERROR_THRESHOLD

    contributions.append(
        ScoreContribution(
            rule="negative_position_error",
            active=active,
            contribution=0.35 if active else 0.0,
            explanation=(
                f"signed error={position_error:.6f} "
                f"<= -{DIRECTIONAL_POSITION_ERROR_THRESHOLD:.2f}"
            ),
        )
    )

    active = temperature < HIGH_TEMPERATURE_THRESHOLD

    contributions.append(
        ScoreContribution(
            rule="temperature_not_high",
            active=active,
            contribution=0.05 if active else 0.0,
            explanation=(
                f"temperature={temperature:.6f} "
                f"< {HIGH_TEMPERATURE_THRESHOLD:.2f}"
            ),
        )
    )

    score = sum(
        contribution.contribution
        for contribution in contributions
    )

    return FaultScoreBreakdown(
        fault_type="LOW_MOTOR_POWER",
        score=score,
        contributions=contributions,
    )


def _calculate_resistance_score(
    *,
    position_error: float,
    velocity: float,
    force: float,
    motor_current: float,
    temperature: float,
) -> FaultScoreBreakdown:
    """Reproduce the current MOTOR_RESISTANCE evidence rules."""

    contributions: list[ScoreContribution] = []

    effort_detected = (
        abs(force) >= HIGH_FORCE_THRESHOLD
        or motor_current >= HIGH_CURRENT_THRESHOLD
    )

    contributions.append(
        ScoreContribution(
            rule="effort_detected",
            active=effort_detected,
            contribution=0.25 if effort_detected else 0.0,
            explanation=(
                f"|force|={abs(force):.6f} >= "
                f"{HIGH_FORCE_THRESHOLD:.2f} OR "
                f"current={motor_current:.6f} >= "
                f"{HIGH_CURRENT_THRESHOLD:.2f}"
            ),
        )
    )

    active = (
        abs(position_error) >= POSITION_ERROR_THRESHOLD
        and effort_detected
    )

    contributions.append(
        ScoreContribution(
            rule="position_error_with_effort",
            active=active,
            contribution=0.30 if active else 0.0,
            explanation=(
                f"|error|={abs(position_error):.6f} >= "
                f"{POSITION_ERROR_THRESHOLD:.2f} "
                f"AND effort detected"
            ),
        )
    )

    active = (
        abs(position_error)
        >= SEVERE_POSITION_ERROR_THRESHOLD
        and effort_detected
    )

    contributions.append(
        ScoreContribution(
            rule="severe_position_error_with_effort",
            active=active,
            contribution=0.15 if active else 0.0,
            explanation=(
                f"|error|={abs(position_error):.6f} >= "
                f"{SEVERE_POSITION_ERROR_THRESHOLD:.2f} "
                f"AND effort detected"
            ),
        )
    )

    active = (
        abs(velocity) <= LOW_VELOCITY_THRESHOLD
        and effort_detected
    )

    contributions.append(
        ScoreContribution(
            rule="low_velocity_with_effort",
            active=active,
            contribution=0.15 if active else 0.0,
            explanation=(
                f"|velocity|={abs(velocity):.6f} <= "
                f"{LOW_VELOCITY_THRESHOLD:.2f} "
                f"AND effort detected"
            ),
        )
    )

    active = motor_current >= HIGH_CURRENT_THRESHOLD

    contributions.append(
        ScoreContribution(
            rule="high_current",
            active=active,
            contribution=0.10 if active else 0.0,
            explanation=(
                f"current={motor_current:.6f} >= "
                f"{HIGH_CURRENT_THRESHOLD:.2f}"
            ),
        )
    )

    active = temperature >= HIGH_TEMPERATURE_THRESHOLD

    contributions.append(
        ScoreContribution(
            rule="high_temperature",
            active=active,
            contribution=0.05 if active else 0.0,
            explanation=(
                f"temperature={temperature:.6f} >= "
                f"{HIGH_TEMPERATURE_THRESHOLD:.2f}"
            ),
        )
    )

    score = sum(
        contribution.contribution
        for contribution in contributions
    )

    return FaultScoreBreakdown(
        fault_type="MOTOR_RESISTANCE",
        score=score,
        contributions=contributions,
    )


def _calculate_position_drift_score(
    *,
    position_error: float,
    velocity: float,
    force: float,
    motor_current: float,
) -> FaultScoreBreakdown:
    """Reproduce the current POSITION_DRIFT evidence rules."""

    contributions: list[ScoreContribution] = []

    active = abs(position_error) >= POSITION_ERROR_THRESHOLD

    contributions.append(
        ScoreContribution(
            rule="position_error",
            active=active,
            contribution=0.25 if active else 0.0,
            explanation=(
                f"|error|={abs(position_error):.6f} >= "
                f"{POSITION_ERROR_THRESHOLD:.2f}"
            ),
        )
    )

    active = abs(position_error) >= SEVERE_POSITION_ERROR_THRESHOLD

    contributions.append(
        ScoreContribution(
            rule="severe_position_error",
            active=active,
            contribution=0.10 if active else 0.0,
            explanation=(
                f"|error|={abs(position_error):.6f} >= "
                f"{SEVERE_POSITION_ERROR_THRESHOLD:.2f}"
            ),
        )
    )

    active = position_error >= DIRECTIONAL_POSITION_ERROR_THRESHOLD

    contributions.append(
        ScoreContribution(
            rule="positive_position_error",
            active=active,
            contribution=0.45 if active else 0.0,
            explanation=(
                f"signed error={position_error:.6f} >= "
                f"{DIRECTIONAL_POSITION_ERROR_THRESHOLD:.2f}"
            ),
        )
    )

    effort_detected = (
        abs(force) >= HIGH_FORCE_THRESHOLD
        or motor_current >= HIGH_CURRENT_THRESHOLD
    )

    active = not effort_detected

    contributions.append(
        ScoreContribution(
            rule="no_effort",
            active=active,
            contribution=0.15 if active else 0.0,
            explanation=(
                f"no effort: |force|={abs(force):.6f}, "
                f"current={motor_current:.6f}"
            ),
        )
    )

    active = (
        position_error >= DIRECTIONAL_POSITION_ERROR_THRESHOLD
        and motor_current <= LOW_CURRENT_THRESHOLD
    )

    contributions.append(
        ScoreContribution(
            rule="positive_error_low_current",
            active=active,
            contribution=0.10 if active else 0.0,
            explanation=(
                "positive position error combined with low current"
            ),
        )
    )

    active = (
        position_error >= DIRECTIONAL_POSITION_ERROR_THRESHOLD
        and abs(force) < HIGH_FORCE_THRESHOLD
    )

    contributions.append(
        ScoreContribution(
            rule="positive_error_low_force",
            active=active,
            contribution=0.05 if active else 0.0,
            explanation=(
                "positive position error combined with "
                "force below resistance threshold"
            ),
        )
    )

    active = (
        position_error >= DIRECTIONAL_POSITION_ERROR_THRESHOLD
        and abs(velocity) > LOW_VELOCITY_THRESHOLD
    )

    contributions.append(
        ScoreContribution(
            rule="positive_error_active_velocity",
            active=active,
            contribution=0.05 if active else 0.0,
            explanation=(
                "positive position error combined with "
                "active velocity"
            ),
        )
    )

    score = sum(
        contribution.contribution
        for contribution in contributions
    )

    return FaultScoreBreakdown(
        fault_type="POSITION_DRIFT",
        score=score,
        contributions=contributions,
    )


def _calculate_normal_score(
    *,
    position_error: float,
    velocity: float,
    force: float,
    motor_current: float,
    temperature: float,
) -> float:
    """Reproduce the current NORMAL evidence score."""

    score = 0.0

    if abs(position_error) < POSITION_ERROR_THRESHOLD:
        score += 0.30

    if (
        LOW_CURRENT_THRESHOLD
        <= motor_current
        <= HIGH_CURRENT_THRESHOLD
    ):
        score += 0.15

    if abs(velocity) > LOW_VELOCITY_THRESHOLD:
        score += 0.15

    if temperature < HIGH_TEMPERATURE_THRESHOLD:
        score += 0.20

    if abs(force) < HIGH_FORCE_THRESHOLD:
        score += 0.20

    return score


# ---------------------------------------------------------------------------
# Telemetry signature analysis
# ---------------------------------------------------------------------------


def _analyse_telemetry_signature(
    *,
    position_error: float,
    velocity: float,
    force: float,
    motor_current: float,
    temperature: float,
) -> tuple[
    dict[str, bool],
    list[str],
    float,
    float,
    float,
]:
    """
    Analyse the same telemetry regions used by DiagnosticBrain.

    This is intentionally reported separately from FaultEvidenceEngine.
    """

    candidates = {
        "MOTOR_RESISTANCE": False,
        "LOW_MOTOR_POWER": False,
        "POSITION_DRIFT": False,
    }

    notes: list[str] = []

    resistance_score = 0.0
    low_power_score = 0.0
    drift_score = 0.0

    # ---------------------------------------------------------
    # Motor resistance signature
    # ---------------------------------------------------------

    resistance_effort = (
        abs(force) >= RESISTANCE_FORCE_LIMIT
        or motor_current >= RESISTANCE_CURRENT_LIMIT
    )

    if resistance_effort:
        candidates["MOTOR_RESISTANCE"] = True
        resistance_score += 1.0
        notes.append(
            "Resistance signature: high actuator effort detected."
        )

    if temperature >= TEMPERATURE_HIGH_LIMIT:
        candidates["MOTOR_RESISTANCE"] = True
        resistance_score += 0.25
        notes.append(
            "Resistance signature: high temperature detected."
        )

    # ---------------------------------------------------------
    # Low motor power signature
    # ---------------------------------------------------------

    if (
        motor_current <= LOW_POWER_CURRENT_LIMIT
        and abs(force) <= LOW_POWER_FORCE_LIMIT
        and abs(velocity) <= LOW_VELOCITY_THRESHOLD
    ):
        candidates["LOW_MOTOR_POWER"] = True
        low_power_score += 1.0
        notes.append(
            "Low-power region: current, force and velocity "
            "are all low."
        )

    # ---------------------------------------------------------
    # Position drift signature
    # ---------------------------------------------------------

    if abs(position_error) >= POSITION_ERROR_SIGNATURE_LIMIT:
        candidates["POSITION_DRIFT"] = True
        drift_score += 1.0
        notes.append(
            "Position-drift region: position error exceeds "
            "signature limit."
        )

    if (
        position_error > 0
        and abs(force) < RESISTANCE_FORCE_LIMIT
    ):
        candidates["POSITION_DRIFT"] = True
        drift_score += 0.5
        notes.append(
            "Position-drift region: positive position error "
            "with non-resistance force."
        )

    return (
        candidates,
        notes,
        resistance_score,
        low_power_score,
        drift_score,
    )


# ---------------------------------------------------------------------------
# Important finding generation
# ---------------------------------------------------------------------------


def _build_findings(
    *,
    resistance: FaultScoreBreakdown,
    low_power: FaultScoreBreakdown,
    drift: FaultScoreBreakdown,
    normal_score: float,
    position_error: float,
    motor_current: float,
    force: float,
    velocity: float,
    temperature: float,
    signature_candidates: dict[str, bool],
) -> list[str]:
    """Generate forensic findings without changing the diagnosis."""

    findings: list[str] = []

    scores = {
        "MOTOR_RESISTANCE": resistance.score,
        "LOW_MOTOR_POWER": low_power.score,
        "POSITION_DRIFT": drift.score,
        "NORMAL": normal_score,
    }

    sorted_scores = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    best_fault, best_score = sorted_scores[0]

    findings.append(
        f"Highest reconstructed evidence score: "
        f"{best_fault}={best_score:.4f}."
    )

    if (
        abs(force) < LOW_FORCE_THRESHOLD
        and abs(velocity) < LOW_VELOCITY_THRESHOLD
    ):
        findings.append(
            "Telemetry is near-static: force and velocity are "
            "both below low-effort thresholds."
        )

    if (
        LOW_CURRENT_THRESHOLD
        < motor_current
        < HIGH_CURRENT_THRESHOLD
    ):
        findings.append(
            "Motor current is moderate: above low-current threshold "
            "but below high-current resistance threshold."
        )

    if temperature >= HIGH_TEMPERATURE_THRESHOLD:
        findings.append(
            "Temperature contributes to elevated evidence because "
            "it is above the current 45°C threshold."
        )

    if abs(position_error) < POSITION_ERROR_THRESHOLD:
        findings.append(
            "Position error is below the main 0.20 threshold, "
            "so ordinary position-error evidence is inactive."
        )

    if abs(position_error) >= POSITION_ERROR_SIGNATURE_LIMIT:
        findings.append(
            "Position error is above the DiagnosticBrain telemetry "
            "signature limit of 0.12."
        )

    if position_error < 0:
        findings.append(
            "Signed error is negative: actual position is below "
            "the commanded target."
        )

    if position_error > 0:
        findings.append(
            "Signed error is positive: actual position is above "
            "the commanded target."
        )

    if signature_candidates["POSITION_DRIFT"]:
        findings.append(
            "Telemetry signature logic marks POSITION_DRIFT as a "
            "candidate even though directional evidence may be weak."
        )

    if signature_candidates["LOW_MOTOR_POWER"]:
        findings.append(
            "Telemetry signature logic marks LOW_MOTOR_POWER as a "
            "candidate because actuator activity is low."
        )

    if signature_candidates["MOTOR_RESISTANCE"]:
        findings.append(
            "Telemetry signature logic marks MOTOR_RESISTANCE as a "
            "candidate."
        )

    return findings


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyse_record(
    record: dict[str, Any],
) -> TrialBreakdown:
    """Analyse one targeted failure."""

    (
        trial_id,
        ground_truth,
        recorded_prediction,
        replay_prediction,
        severity,
        target_position,
    ) = _extract_trial_metadata(record)

    telemetry = _extract_telemetry(record)

    position = telemetry["position"]
    velocity = telemetry["velocity"]
    force = telemetry["force"]
    motor_current = telemetry["motor_current"]
    temperature = telemetry["temperature"]

    # IMPORTANT:
    #
    # FaultEvidenceEngine expects:
    #
    # negative = actual below commanded
    # positive = actual above commanded
    #
    actual_minus_commanded = (
        position - target_position
    )

    commanded_minus_actual = (
        target_position - position
    )

    low_power = _calculate_low_power_score(
        position_error=actual_minus_commanded,
        velocity=velocity,
        force=force,
        motor_current=motor_current,
        temperature=temperature,
    )

    resistance = _calculate_resistance_score(
        position_error=actual_minus_commanded,
        velocity=velocity,
        force=force,
        motor_current=motor_current,
        temperature=temperature,
    )

    drift = _calculate_position_drift_score(
        position_error=actual_minus_commanded,
        velocity=velocity,
        force=force,
        motor_current=motor_current,
    )

    normal_score = _calculate_normal_score(
        position_error=actual_minus_commanded,
        velocity=velocity,
        force=force,
        motor_current=motor_current,
        temperature=temperature,
    )

    scores = {
        "MOTOR_RESISTANCE": resistance.score,
        "LOW_MOTOR_POWER": low_power.score,
        "POSITION_DRIFT": drift.score,
        "NORMAL": normal_score,
    }

    non_normal_scores = {
        key: value
        for key, value in scores.items()
        if key != "NORMAL"
    }

    best_fault = max(
        non_normal_scores,
        key=non_normal_scores.get,
    )

    (
        signature_candidates,
        signature_notes,
        resistance_signature_score,
        low_power_signature_score,
        drift_signature_score,
    ) = _analyse_telemetry_signature(
        position_error=actual_minus_commanded,
        velocity=velocity,
        force=force,
        motor_current=motor_current,
        temperature=temperature,
    )

    findings = _build_findings(
        resistance=resistance,
        low_power=low_power,
        drift=drift,
        normal_score=normal_score,
        position_error=actual_minus_commanded,
        motor_current=motor_current,
        force=force,
        velocity=velocity,
        temperature=temperature,
        signature_candidates=signature_candidates,
    )

    return TrialBreakdown(
        trial_id=trial_id,
        ground_truth=ground_truth,
        recorded_prediction=recorded_prediction,
        replay_prediction=replay_prediction,
        severity=severity,
        target_position=target_position,
        position=position,
        velocity=velocity,
        force=force,
        motor_current=motor_current,
        temperature=temperature,
        actual_minus_commanded=actual_minus_commanded,
        commanded_minus_actual=commanded_minus_actual,
        evidence_scores=scores,
        best_evidence_fault=best_fault,
        telemetry_signature_candidates=signature_candidates,
        telemetry_signature_notes=signature_notes,
        resistance_signature_score=resistance_signature_score,
        low_power_signature_score=low_power_signature_score,
        position_drift_signature_score=drift_signature_score,
        normal_score=normal_score,
        important_findings=findings,
    )


def _print_score_table(
    breakdown: TrialBreakdown,
) -> None:
    """Print compact forensic output."""

    print()
    print(
        f"TRIAL: {breakdown.trial_id}"
    )
    print("-" * 78)

    print(
        f"Truth              : "
        f"{breakdown.ground_truth}"
    )

    print(
        f"Recorded prediction: "
        f"{breakdown.recorded_prediction}"
    )

    print(
        f"Replay prediction  : "
        f"{breakdown.replay_prediction}"
    )

    print(
        f"Severity           : "
        f"{breakdown.severity:.4f}"
    )

    print()

    print("TELEMETRY")
    print("-" * 78)

    print(
        f"Position           : "
        f"{breakdown.position:.9f}"
    )

    print(
        f"Target             : "
        f"{breakdown.target_position:.9f}"
    )

    print(
        f"Actual - commanded : "
        f"{breakdown.actual_minus_commanded:+.9f}"
    )

    print(
        f"Commanded - actual : "
        f"{breakdown.commanded_minus_actual:+.9f}"
    )

    print(
        f"Velocity           : "
        f"{breakdown.velocity:.9f}"
    )

    print(
        f"Force              : "
        f"{breakdown.force:.9f}"
    )

    print(
        f"Motor current      : "
        f"{breakdown.motor_current:.6f}"
    )

    print(
        f"Temperature        : "
        f"{breakdown.temperature:.6f}"
    )

    print()

    print("RECONSTRUCTED EVIDENCE SCORES")
    print("-" * 78)

    for fault_type, score in breakdown.evidence_scores.items():
        print(
            f"{fault_type:<20}: "
            f"{score:.4f}"
        )

    print(
        f"Best non-normal evidence: "
        f"{breakdown.best_evidence_fault}"
    )

    print()

    print("TELEMETRY SIGNATURE CANDIDATES")
    print("-" * 78)

    for name, active in (
        breakdown.telemetry_signature_candidates.items()
    ):
        print(
            f"{name:<20}: "
            f"{'ACTIVE' if active else 'inactive'}"
        )

    print(
        f"Resistance signature score : "
        f"{breakdown.resistance_signature_score:.4f}"
    )

    print(
        f"Low-power signature score   : "
        f"{breakdown.low_power_signature_score:.4f}"
    )

    print(
        f"Position-drift signature    : "
        f"{breakdown.position_drift_signature_score:.4f}"
    )

    print()

    print("IMPORTANT FINDINGS")
    print("-" * 78)

    for finding in breakdown.important_findings:
        print(f"- {finding}")


def main() -> None:
    """Run evidence-only forensic analysis."""

    print()
    print(
        "ROBOTWIN-AI DIAGNOSTIC EVIDENCE BREAKDOWN"
    )
    print("=" * 78)

    print(
        f"Input : {INPUT_FILE}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    records = _load_trace()

    print(
        f"Targeted failure records: {len(records)}"
    )

    breakdowns: list[TrialBreakdown] = []

    for record in records:

        breakdown = analyse_record(
            record
        )

        breakdowns.append(
            breakdown
        )

        _print_score_table(
            breakdown
        )

    payload = {
        "study": (
            "RoboTwin-AI diagnostic evidence "
            "breakdown"
        ),
        "analysis_type": (
            "non_simulation_forensic_analysis"
        ),
        "production_code_modified": False,
        "simulation_rerun": False,
        "thresholds_modified": False,
        "input_file": str(INPUT_FILE),
        "known_failures": sorted(
            KNOWN_FAILURES
        ),
        "records": [
            asdict(item)
            for item in breakdowns
        ],
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

    print()
    print("=" * 78)
    print(
        "Evidence breakdown completed."
    )
    print(
        f"Saved to: {OUTPUT_FILE}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()