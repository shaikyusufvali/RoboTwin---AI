"""Temporal diagnostic reasoning for RoboTwin-AI.

This module adds time-aware reasoning on top of the existing
DiagnosticBrain without modifying the existing diagnostic pipeline.

Flow:

    telemetry snapshot
            ↓
      DiagnosticBrain
            ↓
       diagnosis result
            ↓
    TemporalDiagnosticEngine
            ↓
    history + persistence + trend
            ↓
      temporal diagnosis
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Optional


@dataclass(frozen=True)
class TemporalDiagnosis:
    """Time-aware diagnostic result."""

    fault_type: str
    confidence: float

    history_size: int
    persistence_score: float

    state: str
    trend: str

    developing: bool
    persistent: bool
    stable: bool

    message: str


@dataclass(frozen=True)
class TemporalObservation:
    """One diagnostic observation stored in temporal history."""

    fault_type: str
    confidence: float
    position: float
    velocity: float
    force: float
    motor_current: float
    temperature: float


class TemporalDiagnosticEngine:
    """Add temporal reasoning around the existing DiagnosticBrain.

    The existing DiagnosticBrain remains responsible for diagnosing
    individual telemetry snapshots.

    This class only reasons about repeated observations over time.
    """

    DEFAULT_HISTORY_SIZE = 10

    MIN_PERSISTENCE_COUNT = 3
    DEVELOPING_COUNT = 2

    CONFIDENCE_INCREASE_THRESHOLD = 0.03
    CONFIDENCE_DECREASE_THRESHOLD = 0.03

    def __init__(
        self,
        diagnostic_brain: Any,
        history_size: int = DEFAULT_HISTORY_SIZE,
    ) -> None:
        if history_size < 2:
            raise ValueError(
                "history_size must be at least 2."
            )

        self._diagnostic_brain = diagnostic_brain

        self._history: Deque[
            TemporalObservation
        ] = deque(
            maxlen=history_size
        )

    @property
    def history(self) -> tuple[TemporalObservation, ...]:
        """Return immutable temporal history."""

        return tuple(self._history)

    def reset(self) -> None:
        """Clear temporal history."""

        self._history.clear()

    def update(
        self,
        telemetry: Any,
        target_position: float,
        vision_observation: Optional[Any] = None,
    ) -> TemporalDiagnosis:
        """Diagnose one snapshot and update temporal reasoning."""

        diagnosis = self._diagnose_snapshot(
            telemetry=telemetry,
            target_position=target_position,
            vision_observation=vision_observation,
        )

        observation = TemporalObservation(
            fault_type=self._extract_fault_type(
                diagnosis
            ),
            confidence=self._extract_confidence(
                diagnosis
            ),
            position=self._extract_float(
                telemetry,
                "position",
            ),
            velocity=self._extract_float(
                telemetry,
                "velocity",
            ),
            force=self._extract_float(
                telemetry,
                "force",
            ),
            motor_current=self._extract_float(
                telemetry,
                "motor_current",
            ),
            temperature=self._extract_float(
                telemetry,
                "temperature",
            ),
        )

        self._history.append(observation)

        return self._build_temporal_diagnosis()

    def _diagnose_snapshot(
        self,
        telemetry: Any,
        target_position: float,
        vision_observation: Optional[Any],
    ) -> Any:
        """Use the existing DiagnosticBrain API."""

        if vision_observation is None:
            return self._diagnostic_brain.diagnose(
                telemetry,
                target_position,
            )

        return self._diagnostic_brain.diagnose(
            telemetry,
            target_position,
            vision_observation=vision_observation,
        )

    def _build_temporal_diagnosis(
        self,
    ) -> TemporalDiagnosis:
        """Build a temporal diagnosis from history."""

        if not self._history:
            return TemporalDiagnosis(
                fault_type="UNKNOWN",
                confidence=0.0,
                history_size=0,
                persistence_score=0.0,
                state="NO_DATA",
                trend="UNKNOWN",
                developing=False,
                persistent=False,
                stable=False,
                message="No telemetry history available.",
            )

        latest = self._history[-1]

        persistence_score = (
            self._calculate_persistence_score()
        )

        trend = self._calculate_confidence_trend()

        persistent = (
            len(self._history)
            >= self.MIN_PERSISTENCE_COUNT
            and persistence_score >= 0.67
        )

        developing = (
            len(self._history)
            >= self.DEVELOPING_COUNT
            and latest.fault_type != "NORMAL"
            and persistence_score >= 0.50
        )

        stable = (
            latest.fault_type != "NORMAL"
            and persistent
            and trend == "STABLE"
        )

        state = self._determine_state(
            latest_fault=latest.fault_type,
            developing=developing,
            persistent=persistent,
            stable=stable,
        )

        message = self._build_message(
            fault_type=latest.fault_type,
            confidence=latest.confidence,
            state=state,
            trend=trend,
            persistence_score=persistence_score,
        )

        return TemporalDiagnosis(
            fault_type=latest.fault_type,
            confidence=latest.confidence,
            history_size=len(self._history),
            persistence_score=persistence_score,
            state=state,
            trend=trend,
            developing=developing,
            persistent=persistent,
            stable=stable,
            message=message,
        )

    def _calculate_persistence_score(self) -> float:
        """Calculate how consistently the latest fault appears."""

        if not self._history:
            return 0.0

        latest_fault = self._history[-1].fault_type

        matching = sum(
            observation.fault_type == latest_fault
            for observation in self._history
        )

        return matching / len(self._history)

    def _calculate_confidence_trend(self) -> str:
        """Determine whether confidence is rising, falling, or stable."""

        if len(self._history) < 2:
            return "UNKNOWN"

        previous = self._history[-2].confidence
        current = self._history[-1].confidence

        delta = current - previous

        if delta >= self.CONFIDENCE_INCREASE_THRESHOLD:
            return "INCREASING"

        if delta <= -self.CONFIDENCE_DECREASE_THRESHOLD:
            return "DECREASING"

        return "STABLE"

    @staticmethod
    def _determine_state(
        latest_fault: str,
        developing: bool,
        persistent: bool,
        stable: bool,
    ) -> str:
        """Determine temporal state."""

        if latest_fault == "NORMAL":
            return "HEALTHY"

        if stable:
            return "STABLE_FAULT"

        if persistent:
            return "PERSISTENT_FAULT"

        if developing:
            return "DEVELOPING_FAULT"

        return "SINGLE_OBSERVATION"

    @staticmethod
    def _build_message(
        fault_type: str,
        confidence: float,
        state: str,
        trend: str,
        persistence_score: float,
    ) -> str:
        """Create a human-readable temporal diagnosis."""

        if state == "HEALTHY":
            return (
                "Robot telemetry is currently healthy."
            )

        if state == "DEVELOPING_FAULT":
            return (
                f"{fault_type} appears to be developing. "
                f"Confidence={confidence:.3f}, "
                f"persistence={persistence_score:.2f}, "
                f"trend={trend}."
            )

        if state == "PERSISTENT_FAULT":
            return (
                f"{fault_type} is persistent across recent "
                f"telemetry observations. "
                f"Confidence={confidence:.3f}, "
                f"persistence={persistence_score:.2f}."
            )

        if state == "STABLE_FAULT":
            return (
                f"{fault_type} remains stable across the "
                f"recent telemetry window. "
                f"Confidence={confidence:.3f}."
            )

        return (
            f"{fault_type} detected from the latest "
            f"observation. Temporal evidence is still limited."
        )

    @staticmethod
    def _extract_fault_type(
        diagnosis: Any,
    ) -> str:
        """Extract fault type from an existing diagnosis object."""

        candidates = (
            "fault_type",
            "ml_fault_type",
            "predicted_fault",
            "diagnosis",
        )

        for attribute in candidates:
            value = getattr(
                diagnosis,
                attribute,
                None,
            )

            if value is None:
                continue

            if hasattr(value, "value"):
                value = value.value

            return str(value)

        if isinstance(diagnosis, dict):
            for key in candidates:
                if key in diagnosis:
                    value = diagnosis[key]

                    if hasattr(value, "value"):
                        value = value.value

                    return str(value)

        return "UNKNOWN"

    @staticmethod
    def _extract_confidence(
        diagnosis: Any,
    ) -> float:
        """Extract confidence from an existing diagnosis object."""

        candidates = (
            "confidence",
            "ml_confidence",
            "diagnostic_confidence",
        )

        for attribute in candidates:
            value = getattr(
                diagnosis,
                attribute,
                None,
            )

            if value is not None:
                return float(value)

        if isinstance(diagnosis, dict):
            for key in candidates:
                if key in diagnosis:
                    return float(
                        diagnosis[key]
                    )

        return 0.0

    @staticmethod
    def _extract_float(
        telemetry: Any,
        attribute: str,
    ) -> float:
        """Safely extract a numeric telemetry value."""

        value = getattr(
            telemetry,
            attribute,
            0.0,
        )

        if value is None:
            return 0.0

        return float(value)