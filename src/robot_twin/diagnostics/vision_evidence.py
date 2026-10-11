
from __future__ import annotations

import math
from dataclasses import dataclass


DEFAULT_OBSTACLE_DISTANCE_THRESHOLD = 1.0
DEFAULT_ABNORMAL_MOTION_THRESHOLD = 0.20

OBSTACLE_EVIDENCE_WEIGHT = 0.60
ABNORMAL_MOTION_EVIDENCE_WEIGHT = 0.40

MIN_DEPTH_VALID_RATIO = 0.10


@dataclass(frozen=True)
class VisionEvidence:
    """
    Structured evidence produced from camera observations.

    score:
        Strength of observed visual abnormality.

    reliability:
        Trustworthiness of the visual observation itself.

    These two values are intentionally separate. A reliable camera
    reporting no abnormality has a score of 0.0 and a high reliability.
    """

    obstacle_detected: bool
    abnormal_motion: bool
    visibility_ok: bool
    depth_available: bool
    segmentation_available: bool
    valid_depth_ratio: float
    score: float
    reliability: float
    reasons: tuple[str, ...]


class VisionEvidenceEngine:
    """
    Convert camera observations into diagnostic evidence.

    The engine separates:

    1. Visual abnormality evidence.
    2. Visual observation reliability.

    This prevents a healthy visual observation from being interpreted
    as negative diagnostic evidence.

    The engine does not independently declare an internal robot fault.
    It only provides evidence that can be fused with telemetry diagnosis.
    """

    def __init__(
        self,
        obstacle_distance_threshold: float = (
            DEFAULT_OBSTACLE_DISTANCE_THRESHOLD
        ),
        abnormal_motion_threshold: float = (
            DEFAULT_ABNORMAL_MOTION_THRESHOLD
        ),
    ) -> None:
        self._validate_positive(
            obstacle_distance_threshold,
            "obstacle_distance_threshold",
        )
        self._validate_positive(
            abnormal_motion_threshold,
            "abnormal_motion_threshold",
        )

        self.obstacle_distance_threshold = obstacle_distance_threshold
        self.abnormal_motion_threshold = abnormal_motion_threshold

    def analyze(
        self,
        obstacle_distance: float | None,
        observed_motion_error: float = 0.0,
        visibility_ok: bool = True,
        depth_available: bool = True,
        segmentation_available: bool = True,
        valid_depth_ratio: float = 1.0,
    ) -> VisionEvidence:
        """
        Analyze camera-derived observations.

        Parameters
        ----------
        obstacle_distance:
            Nearest detected obstacle distance in meters.
            ``None`` means no obstacle distance is available.

        observed_motion_error:
            Difference between expected and observed robot motion.
            This should be supplied only when the motion error was
            derived from a vision-based observation.

        visibility_ok:
            Whether the robot is sufficiently visible.

        depth_available:
            Whether a valid depth image is available.

        segmentation_available:
            Whether segmentation information is available.

        valid_depth_ratio:
            Fraction of depth pixels considered valid, in [0, 1].

        Returns
        -------
        VisionEvidence
            Structured visual abnormality and reliability information.
        """

        self._validate_boolean(visibility_ok, "visibility_ok")
        self._validate_boolean(depth_available, "depth_available")
        self._validate_boolean(
            segmentation_available,
            "segmentation_available",
        )

        self._validate_ratio(
            valid_depth_ratio,
            "valid_depth_ratio",
        )

        if obstacle_distance is not None:
            self._validate_finite_non_negative(
                obstacle_distance,
                "obstacle_distance",
            )

        self._validate_finite(
            observed_motion_error,
            "observed_motion_error",
        )

        obstacle_detected = self._detect_obstacle(
            obstacle_distance=obstacle_distance,
        )

        abnormal_motion = self._detect_abnormal_motion(
            observed_motion_error=observed_motion_error,
        )

        score = self._calculate_evidence_score(
            obstacle_detected=obstacle_detected,
            abnormal_motion=abnormal_motion,
        )

        reliability = self._calculate_reliability(
            visibility_ok=visibility_ok,
            depth_available=depth_available,
            segmentation_available=segmentation_available,
            valid_depth_ratio=valid_depth_ratio,
        )

        reasons = self._build_reasons(
            obstacle_detected=obstacle_detected,
            abnormal_motion=abnormal_motion,
            visibility_ok=visibility_ok,
            depth_available=depth_available,
            segmentation_available=segmentation_available,
            valid_depth_ratio=valid_depth_ratio,
            score=score,
            reliability=reliability,
        )

        return VisionEvidence(
            obstacle_detected=obstacle_detected,
            abnormal_motion=abnormal_motion,
            visibility_ok=visibility_ok,
            depth_available=depth_available,
            segmentation_available=segmentation_available,
            valid_depth_ratio=valid_depth_ratio,
            score=score,
            reliability=reliability,
            reasons=tuple(reasons),
        )

    def _detect_obstacle(
        self,
        obstacle_distance: float | None,
    ) -> bool:
        """Determine whether a detected obstacle is inside the danger zone."""

        if obstacle_distance is None:
            return False

        return (
            obstacle_distance
            <= self.obstacle_distance_threshold
        )

    def _detect_abnormal_motion(
        self,
        observed_motion_error: float,
    ) -> bool:
        """Determine whether visual motion differs significantly from expectation."""

        return (
            abs(observed_motion_error)
            >= self.abnormal_motion_threshold
        )

    @staticmethod
    def _calculate_evidence_score(
        obstacle_detected: bool,
        abnormal_motion: bool,
    ) -> float:
        """
        Calculate visual abnormality strength.

        This score measures abnormality evidence only. It does not
        represent camera reliability.
        """

        score = 0.0

        if obstacle_detected:
            score += OBSTACLE_EVIDENCE_WEIGHT

        if abnormal_motion:
            score += ABNORMAL_MOTION_EVIDENCE_WEIGHT

        return min(max(score, 0.0), 1.0)

    @staticmethod
    def _calculate_reliability(
        visibility_ok: bool,
        depth_available: bool,
        segmentation_available: bool,
        valid_depth_ratio: float,
    ) -> float:
        """
        Calculate trustworthiness of the visual observation.

        Reliability is based on observation quality, not on whether
        the robot appears healthy or abnormal.
        """

        if not visibility_ok:
            return 0.0

        reliability = 0.0

        # Robot visibility is the primary prerequisite.
        reliability += 0.40

        # Depth quality is important for obstacle reasoning.
        if depth_available:
            depth_quality = max(
                0.0,
                min(valid_depth_ratio, 1.0),
            )

            reliability += 0.40 * depth_quality

        # Segmentation provides additional scene confidence.
        if segmentation_available:
            reliability += 0.20

        if valid_depth_ratio < MIN_DEPTH_VALID_RATIO:
            reliability = min(reliability, 0.60)

        return min(max(reliability, 0.0), 1.0)

    @staticmethod
    def _build_reasons(
        obstacle_detected: bool,
        abnormal_motion: bool,
        visibility_ok: bool,
        depth_available: bool,
        segmentation_available: bool,
        valid_depth_ratio: float,
        score: float,
        reliability: float,
    ) -> list[str]:
        """Build human-readable diagnostic evidence."""

        reasons: list[str] = []

        if obstacle_detected:
            reasons.append(
                "Camera detected an obstacle inside the configured "
                "danger distance."
            )

        if abnormal_motion:
            reasons.append(
                "Vision-derived robot motion differs significantly "
                "from the expected motion."
            )

        if not obstacle_detected and not abnormal_motion:
            reasons.append(
                "No significant visual abnormality detected."
            )

        if not visibility_ok:
            reasons.append(
                "Robot visibility is insufficient for reliable "
                "visual analysis."
            )

        if not depth_available:
            reasons.append(
                "Depth information is unavailable."
            )

        if not segmentation_available:
            reasons.append(
                "Segmentation information is unavailable."
            )

        if valid_depth_ratio < MIN_DEPTH_VALID_RATIO:
            reasons.append(
                "Valid depth coverage is below the minimum "
                "reliability threshold."
            )

        reasons.append(
            f"Visual abnormality score: {score:.4f}."
        )

        reasons.append(
            f"Visual observation reliability: {reliability:.4f}."
        )

        return reasons

    @staticmethod
    def _validate_positive(
        value: float,
        name: str,
    ) -> None:
        """Validate a finite positive numeric configuration value."""

        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                f"{name} must be a numeric value."
            )

        if not math.isfinite(float(value)):
            raise ValueError(
                f"{name} must be finite."
            )

        if float(value) <= 0:
            raise ValueError(
                f"{name} must be greater than zero."
            )

    @staticmethod
    def _validate_finite(
        value: float,
        name: str,
    ) -> None:
        """Validate a finite numeric runtime value."""

        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                f"{name} must be a numeric value."
            )

        if not math.isfinite(float(value)):
            raise ValueError(
                f"{name} must be finite."
            )

    @staticmethod
    def _validate_finite_non_negative(
        value: float,
        name: str,
    ) -> None:
        """Validate a finite non-negative numeric value."""

        VisionEvidenceEngine._validate_finite(
            value=value,
            name=name,
        )

        if float(value) < 0:
            raise ValueError(
                f"{name} cannot be negative."
            )

    @staticmethod
    def _validate_ratio(
        value: float,
        name: str,
    ) -> None:
        """Validate a numeric ratio in the closed interval [0, 1]."""

        VisionEvidenceEngine._validate_finite(
            value=value,
            name=name,
        )

        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"{name} must be between 0.0 and 1.0."
            )

    @staticmethod
    def _validate_boolean(
        value: bool,
        name: str,
    ) -> None:
        """Validate a strict boolean value."""

        if not isinstance(value, bool):
            raise TypeError(
                f"{name} must be a boolean."
            )


def main() -> None:
    """Run a deterministic demonstration of the evidence engine."""

    engine = VisionEvidenceEngine()

    evidence = engine.analyze(
        obstacle_distance=0.65,
        observed_motion_error=0.28,
        visibility_ok=True,
        depth_available=True,
        segmentation_available=True,
        valid_depth_ratio=1.0,
    )

    print("ROBOT VISION EVIDENCE ENGINE")
    print("=" * 60)
    print(f"Obstacle detected       : {evidence.obstacle_detected}")
    print(f"Abnormal motion         : {evidence.abnormal_motion}")
    print(f"Visibility OK           : {evidence.visibility_ok}")
    print(f"Depth available         : {evidence.depth_available}")
    print(
        "Segmentation available  : "
        f"{evidence.segmentation_available}"
    )
    print(
        "Valid depth ratio       : "
        f"{evidence.valid_depth_ratio:.4f}"
    )
    print(
        "Visual evidence score   : "
        f"{evidence.score:.4f}"
    )
    print(
        "Visual reliability      : "
        f"{evidence.reliability:.4f}"
    )

    print("\nReasons:")

    for reason in evidence.reasons:
        print(f"  - {reason}")


if __name__ == "__main__":
    main()
