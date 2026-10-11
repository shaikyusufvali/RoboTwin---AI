"""
Multimodal diagnostic brain for RoboTwin-AI.

Combines:
    - Telemetry-based diagnosis
    - Vision-derived evidence
    - Vision reliability
    - Reliability-aware multimodal fusion

Design principle:
    Telemetry remains the primary source for internal robot-fault
    diagnosis. Vision provides supporting environmental and motion
    evidence when the visual observation is reliable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from robot_twin.diagnostics.diagnostic_brain import (
    DiagnosticBrain,
    DiagnosticResult,
)
from robot_twin.diagnostics.vision_evidence import (
    VisionEvidence,
    VisionEvidenceEngine,
)
from robot_twin.telemetry.robot_adapter import JointTelemetry
from robot_twin.vision.vision_observer import VisionObservation


DEFAULT_VISION_CONFIRMATION_BONUS = 0.10
DEFAULT_VISION_ANOMALY_THRESHOLD = 0.60
DEFAULT_HIGH_RELIABILITY_THRESHOLD = 0.80
DEFAULT_LOW_RELIABILITY_THRESHOLD = 0.50


@dataclass(frozen=True)
class MultimodalDiagnosticResult:
    """
    Final diagnostic result after telemetry and vision fusion.
    """

    joint_index: int
    status: str
    fault_type: str
    severity: str
    confidence: float

    telemetry_confidence: float

    vision_score: float
    vision_reliability: float
    combined_score: float

    obstacle_detected: bool

    fusion_mode: str

    evidence: tuple[str, ...]

    probable_cause: str
    recommended_action: str


class MultimodalDiagnosticBrain:
    """
    Reliability-aware multimodal diagnostic engine.

    Telemetry is the primary signal for internal robot faults.

    Vision contributes only when:
        - a valid visual observation exists,
        - the observation has sufficient reliability,
        - the visual evidence is meaningful.

    A healthy visual observation must not reduce telemetry confidence.
    """

    def __init__(
        self,
        diagnostic_brain: DiagnosticBrain | None = None,
        vision_engine: VisionEvidenceEngine | None = None,
        vision_confirmation_bonus: float = DEFAULT_VISION_CONFIRMATION_BONUS,
        vision_anomaly_threshold: float = DEFAULT_VISION_ANOMALY_THRESHOLD,
        high_reliability_threshold: float = DEFAULT_HIGH_RELIABILITY_THRESHOLD,
        low_reliability_threshold: float = DEFAULT_LOW_RELIABILITY_THRESHOLD,
    ) -> None:
        self.diagnostic_brain = diagnostic_brain or DiagnosticBrain()
        self.vision_engine = vision_engine or VisionEvidenceEngine()

        self.vision_confirmation_bonus = self._validate_ratio(
            vision_confirmation_bonus,
            "vision_confirmation_bonus",
        )

        self.vision_anomaly_threshold = self._validate_ratio(
            vision_anomaly_threshold,
            "vision_anomaly_threshold",
        )

        self.high_reliability_threshold = self._validate_ratio(
            high_reliability_threshold,
            "high_reliability_threshold",
        )

        self.low_reliability_threshold = self._validate_ratio(
            low_reliability_threshold,
            "low_reliability_threshold",
        )

        if self.low_reliability_threshold > self.high_reliability_threshold:
            raise ValueError(
                "low_reliability_threshold cannot be greater than "
                "high_reliability_threshold."
            )

    @staticmethod
    def _validate_ratio(value: float, name: str) -> float:
        """Validate a value expected to be between 0 and 1."""

        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")

        value = float(value)

        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")

        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")

        return value

    @staticmethod
    def _clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> float:
        """Clamp a numeric value into a safe range."""

        return max(minimum, min(maximum, float(value)))

    def diagnose(
        self,
        diagnostic_result: DiagnosticResult,
        vision_observation: VisionObservation | None = None,
    ) -> MultimodalDiagnosticResult:
        """
        Fuse telemetry diagnosis with optional vision evidence.
        """

        if not isinstance(diagnostic_result, DiagnosticResult):
            raise TypeError(
                "diagnostic_result must be a DiagnosticResult instance."
            )

        vision_evidence = self._build_vision_evidence(
            vision_observation=vision_observation,
        )

        telemetry_confidence = self._safe_confidence(
            diagnostic_result.confidence,
        )

        combined_score = self._calculate_combined_score(
            telemetry_confidence=telemetry_confidence,
            vision_evidence=vision_evidence,
        )

        status = self._select_status(
            telemetry_result=diagnostic_result,
            vision_evidence=vision_evidence,
        )

        fusion_mode = self._determine_fusion_mode(
            telemetry_result=diagnostic_result,
            vision_evidence=vision_evidence,
        )

        evidence = self._build_evidence(
            diagnostic_result=diagnostic_result,
            vision_evidence=vision_evidence,
            fusion_mode=fusion_mode,
        )

        probable_cause = self._build_probable_cause(
            diagnostic_result=diagnostic_result,
            vision_evidence=vision_evidence,
        )

        recommended_action = self._build_recommended_action(
            diagnostic_result=diagnostic_result,
            vision_evidence=vision_evidence,
        )

        return MultimodalDiagnosticResult(
            joint_index=diagnostic_result.joint_index,
            status=status,
            fault_type=diagnostic_result.fault_type,
            severity=diagnostic_result.severity,
            confidence=combined_score,
            telemetry_confidence=telemetry_confidence,
            vision_score=vision_evidence.score,
            vision_reliability=vision_evidence.reliability,
            combined_score=combined_score,
            obstacle_detected=vision_evidence.obstacle_detected,
            fusion_mode=fusion_mode,
            evidence=tuple(evidence),
            probable_cause=probable_cause,
            recommended_action=recommended_action,
        )

    def _build_vision_evidence(
        self,
        vision_observation: VisionObservation | None,
    ) -> VisionEvidence:
        """Convert a vision observation into normalized visual evidence."""

        if vision_observation is None:
            return VisionEvidence(
                obstacle_detected=False,
                abnormal_motion=False,
                visibility_ok=False,
                depth_available=False,
                segmentation_available=False,
                valid_depth_ratio=0.0,
                score=0.0,
                reliability=0.0,
                reasons=(
                    "Vision observation is unavailable.",
                ),
            )

        if not isinstance(vision_observation, VisionObservation):
            raise TypeError(
                "vision_observation must be a VisionObservation instance."
            )

        return self.vision_engine.analyze(
            obstacle_distance=vision_observation.obstacle_distance,
            observed_motion_error=0.0,
            visibility_ok=vision_observation.visibility_ok,
            depth_available=vision_observation.depth_available,
            segmentation_available=(
                vision_observation.segmentation_available
            ),
            valid_depth_ratio=vision_observation.valid_depth_ratio,
        )

    def _calculate_combined_score(
        self,
        telemetry_confidence: float,
        vision_evidence: VisionEvidence,
    ) -> float:
        """
        Calculate reliability-aware confidence.

        A visual score of zero means no visual abnormality.
        It does not mean that the camera is unreliable.
        """

        telemetry_confidence = self._clamp(
            telemetry_confidence,
        )

        if vision_evidence.reliability <= 0.0:
            return telemetry_confidence

        if vision_evidence.score <= 0.0:
            return telemetry_confidence

        visual_bonus = (
            self.vision_confirmation_bonus
            * vision_evidence.score
            * vision_evidence.reliability
        )

        return self._clamp(
            telemetry_confidence + visual_bonus,
        )

    def _select_status(
        self,
        telemetry_result: DiagnosticResult,
        vision_evidence: VisionEvidence,
    ) -> str:
        """
        Select final diagnostic status.

        Telemetry anomaly remains the strongest internal-fault signal.

        A strong and highly reliable visual anomaly can also trigger
        an anomaly state when telemetry itself is normal.
        """

        if telemetry_result.status.upper() == "ANOMALY":
            return "ANOMALY"

        if (
            vision_evidence.score >= self.vision_anomaly_threshold
            and vision_evidence.reliability
            >= self.high_reliability_threshold
        ):
            return "ANOMALY"

        return "NORMAL"

    def _determine_fusion_mode(
        self,
        telemetry_result: DiagnosticResult,
        vision_evidence: VisionEvidence,
    ) -> str:
        """Determine how telemetry and vision contributed."""

        if vision_evidence.reliability <= 0.0:
            return "TELEMETRY_ONLY"

        if vision_evidence.reliability < self.low_reliability_threshold:
            return "VISION_LOW_RELIABILITY"

        telemetry_anomaly = (
            telemetry_result.status.upper() == "ANOMALY"
        )

        strong_visual_evidence = (
            vision_evidence.score >= self.vision_anomaly_threshold
        )

        high_visual_reliability = (
            vision_evidence.reliability
            >= self.high_reliability_threshold
        )

        if (
            telemetry_anomaly
            and strong_visual_evidence
            and high_visual_reliability
        ):
            return "TELEMETRY_CONFIRMED_WITH_VISION"

        if strong_visual_evidence:
            return "VISION_AUGMENTED"

        return "TELEMETRY_PRIMARY"

    @staticmethod
    def _safe_confidence(value: float) -> float:
        """Normalize diagnostic confidence."""

        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0

        if not math.isfinite(value):
            return 0.0

        return max(0.0, min(1.0, value))

    def _build_evidence(
        self,
        diagnostic_result: DiagnosticResult,
        vision_evidence: VisionEvidence,
        fusion_mode: str,
    ) -> list[str]:
        """Build human-readable diagnostic evidence."""

        evidence: list[str] = []

        for item in diagnostic_result.evidence:
            evidence.append(str(item))

        evidence.append(
            f"Vision evidence score: {vision_evidence.score:.4f}."
        )

        evidence.append(
            "Vision observation reliability: "
            f"{vision_evidence.reliability:.4f}."
        )

        evidence.append(
            f"Fusion mode: {fusion_mode}."
        )

        for reason in vision_evidence.reasons:
            evidence.append(str(reason))

        return self._unique_strings(evidence)

    def _build_probable_cause(
        self,
        diagnostic_result: DiagnosticResult,
        vision_evidence: VisionEvidence,
    ) -> str:
        """
        Build probable cause.

        Vision provides environmental context and does not independently
        claim an internal mechanical fault.
        """

        base_cause = diagnostic_result.probable_cause

        if (
            vision_evidence.obstacle_detected
            and vision_evidence.reliability
            >= self.low_reliability_threshold
        ):
            obstacle_context = (
                " Camera evidence indicates a nearby obstacle or "
                "environmental interaction that may contribute to "
                "increased mechanical load."
            )

            return f"{base_cause}{obstacle_context}"

        return base_cause

    def _build_recommended_action(
        self,
        diagnostic_result: DiagnosticResult,
        vision_evidence: VisionEvidence,
    ) -> str:
        """Build recommended action with visual safety context."""

        base_action = diagnostic_result.recommended_action

        if (
            vision_evidence.obstacle_detected
            and vision_evidence.reliability
            >= self.low_reliability_threshold
        ):
            obstacle_action = (
                " Verify the robot's workspace for obstacles or "
                "unexpected physical interference before continuing "
                "normal operation."
            )

            return f"{base_action}{obstacle_action}"

        return base_action

    @staticmethod
    def _unique_strings(
        items: Iterable[str],
    ) -> list[str]:
        """Remove duplicate strings while preserving order."""

        unique: list[str] = []
        seen: set[str] = set()

        for item in items:
            normalized = str(item).strip()

            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(normalized)
            unique.append(normalized)

        return unique


def main() -> None:
    """
    Run a standalone multimodal diagnostic demonstration.

    Uses a valid JointTelemetry object instead of passing None
    to DiagnosticBrain.
    """

    diagnostic_brain = DiagnosticBrain()
    multimodal_brain = MultimodalDiagnosticBrain()

    telemetry = JointTelemetry(
        joint_index=2,
        position=0.59,
        velocity=0.0,
        force=0.0,
        motor_current=3.325,
        temperature=51.6,
    )

    diagnostic_result = diagnostic_brain.diagnose(
        telemetry=telemetry,
        target_position=telemetry.position,
    )

    vision_observation = VisionObservation(
        obstacle_detected=False,
        obstacle_distance=3.436,
        visibility_ok=True,
        depth_available=True,
        segmentation_available=True,
        valid_depth_ratio=1.0,
        visible_object_count=10,
        image_width=640,
        image_height=480,
        reason=(
            "No obstacle detected inside the configured "
            "safety distance."
        ),
    )

    result = multimodal_brain.diagnose(
        diagnostic_result=diagnostic_result,
        vision_observation=vision_observation,
    )

    print()
    print("ROBOTWIN-AI MULTIMODAL DIAGNOSTIC BRAIN")
    print("=" * 65)

    print(f"Joint                : {result.joint_index}")
    print(f"Status               : {result.status}")
    print(f"Fault                : {result.fault_type}")
    print(f"Severity             : {result.severity}")
    print(
        f"Telemetry confidence : "
        f"{result.telemetry_confidence:.4f}"
    )
    print(
        f"Vision score         : "
        f"{result.vision_score:.4f}"
    )
    print(
        f"Vision reliability   : "
        f"{result.vision_reliability:.4f}"
    )
    print(
        f"Combined confidence : "
        f"{result.combined_score:.4f}"
    )
    print(
        f"Obstacle detected    : "
        f"{result.obstacle_detected}"
    )
    print(
        f"Fusion mode          : "
        f"{result.fusion_mode}"
    )

    print()
    print("Evidence:")

    for item in result.evidence:
        print(f"  - {item}")

    print()
    print("Probable Cause:")
    print(f"  {result.probable_cause}")

    print()
    print("Recommended Action:")
    print(f"  {result.recommended_action}")

    print("=" * 65)


if __name__ == "__main__":
    main()