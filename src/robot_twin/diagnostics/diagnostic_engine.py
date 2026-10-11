"""Diagnostic reasoning engine for RoboTwin-AI."""

from __future__ import annotations

import logging
from dataclasses import dataclass


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiagnosticEvidence:
    """Telemetry evidence used to explain a detected fault."""

    position_error: float
    velocity: float
    force: float
    motor_current: float
    temperature: float


@dataclass(frozen=True)
class DiagnosticReport:
    """Structured robot fault diagnosis."""

    joint_index: int
    fault_type: str
    confidence: float
    probable_cause: str
    evidence: tuple[str, ...]
    recommendation: str


class DiagnosticEngine:
    """Convert classifier output and telemetry into a diagnosis."""

    _CAUSES = {
        "normal": (
            "No significant abnormal behavior was detected "
            "in the monitored joint."
        ),
        "motor_resistance": (
            "Mechanical resistance may be affecting "
            "joint movement."
        ),
        "low_motor_power": (
            "Insufficient motor power may be limiting "
            "joint movement."
        ),
        "position_drift": (
            "The joint position may be deviating from "
            "the commanded trajectory."
        ),
    }

    _RECOMMENDATIONS = {
        "normal": (
            "Continue monitoring the joint during operation."
        ),
        "motor_resistance": (
            "Inspect the joint mechanism, transmission, "
            "and motor load before normal operation."
        ),
        "low_motor_power": (
            "Inspect motor power delivery, actuator load, "
            "and related electrical signals."
        ),
        "position_drift": (
            "Inspect position feedback, calibration, "
            "and mechanical alignment."
        ),
    }

    def create_report(
        self,
        joint_index: int,
        fault_type: str,
        confidence: float,
        evidence: DiagnosticEvidence,
    ) -> DiagnosticReport:
        """Create a human-readable diagnostic report."""

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0.0 and 1.0."
            )

        if fault_type not in self._CAUSES:
            raise ValueError(
                f"Unsupported fault type: {fault_type}"
            )

        evidence_items = self._build_evidence(
            fault_type,
            evidence,
        )

        report = DiagnosticReport(
            joint_index=joint_index,
            fault_type=fault_type,
            confidence=confidence,
            probable_cause=self._CAUSES[fault_type],
            evidence=tuple(evidence_items),
            recommendation=self._RECOMMENDATIONS[fault_type],
        )

        LOGGER.info(
            "Diagnostic created for joint %d: %s",
            joint_index,
            fault_type,
        )

        return report

    def _build_evidence(
        self,
        fault_type: str,
        evidence: DiagnosticEvidence,
    ) -> list[str]:
        """Build evidence statements for the detected fault."""

        items: list[str] = []

        if abs(evidence.position_error) > 0.10:
            items.append(
                "Position error is elevated."
            )

        if abs(evidence.velocity) < 0.05:
            items.append(
                "Joint movement velocity is low."
            )

        if abs(evidence.force) > 20.0:
            items.append(
                "Joint force is elevated."
            )

        if abs(evidence.motor_current) > 1.0:
            items.append(
                "Motor current is elevated."
            )

        if evidence.temperature > 50.0:
            items.append(
                "Motor temperature is elevated."
            )

        if not items:
            items.append(
                "Telemetry values are within the "
                "configured diagnostic thresholds."
            )

        return items


def format_report(
    report: DiagnosticReport,
) -> str:
    """Format a diagnostic report for display."""

    lines = [
        "",
        "=" * 72,
        "ROBOTWIN-AI DIAGNOSTIC REPORT",
        "=" * 72,
        f"Joint: {report.joint_index}",
        f"Detected fault: {report.fault_type}",
        f"Confidence: {report.confidence:.1%}",
        "",
        "Probable cause:",
        report.probable_cause,
        "",
        "Evidence:",
    ]

    lines.extend(
        f"- {item}"
        for item in report.evidence
    )

    lines.extend(
        [
            "",
            "Recommended inspection:",
            report.recommendation,
            "=" * 72,
        ]
    )

    return "\n".join(lines)


def main() -> None:
    """Run a diagnostic engine demonstration."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    engine = DiagnosticEngine()

    evidence = DiagnosticEvidence(
        position_error=0.24,
        velocity=0.03,
        force=32.0,
        motor_current=1.8,
        temperature=62.0,
    )

    report = engine.create_report(
        joint_index=2,
        fault_type="motor_resistance",
        confidence=0.91,
        evidence=evidence,
    )

    print(format_report(report))


if __name__ == "__main__":
    main()