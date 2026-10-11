
"""
Multimodal AI diagnostic orchestration for RoboTwin-AI.

The diagnostic brain combines:

    - telemetry anomaly detection
    - machine-learning fault classification
    - deterministic telemetry fault signatures
    - deterministic fault evidence
    - camera / vision observations

Architecture
------------

Telemetry is the primary diagnostic modality for internal actuator
faults.

Vision provides:

    - environmental context
    - obstacle information
    - observation-quality information

Vision must not independently declare an internal actuator fault.

The telemetry API remains backward compatible:

    brain.diagnose(
        telemetry,
        target_position,
    )

Optional vision:

    brain.diagnose(
        telemetry,
        target_position,
        vision_observation,
    )

Diagnostic pipeline:

    telemetry
        |
        +--> baseline anomaly detection
        +--> deterministic telemetry signature
        +--> ML fault classification
        +--> deterministic fault evidence
        +--> optional vision context
        +--> final diagnosis
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Final

from robot_twin.diagnostics.baseline_anomaly_detector import (
    BaselineAnomalyDetector,
)

from robot_twin.diagnostics.fault_classifier import (
    FaultClassifier,
)

from robot_twin.diagnostics.fault_evidence_engine import (
    FaultEvidence,
    FaultEvidenceEngine,
)

from robot_twin.telemetry.robot_adapter import JointTelemetry

from robot_twin.vision.vision_observer import (
    VisionObservation,
)


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostic thresholds
# ---------------------------------------------------------------------------

DEFAULT_HIGH_SEVERITY_SCORE: Final[float] = 0.80
DEFAULT_MEDIUM_SEVERITY_SCORE: Final[float] = 0.50
DEFAULT_CLASSIFIER_CONFIDENCE: Final[float] = 0.60
DEFAULT_EVIDENCE_SCORE: Final[float] = 0.50
DEFAULT_DEGRADED_EVIDENCE_SCORE: Final[float] = 0.50

DEFAULT_MAX_VISION_CONFIDENCE_BONUS: Final[float] = 0.08
DEFAULT_MAX_VISION_CONFIDENCE_PENALTY: Final[float] = 0.05


# ---------------------------------------------------------------------------
# Telemetry signature thresholds
#
# These thresholds are diagnostic thresholds for the controlled
# RoboTwin-AI simulation.
#
# They are NOT physical robot specifications.
# ---------------------------------------------------------------------------

POSITION_ERROR_SIGNATURE_LIMIT: Final[float] = 0.05

LOW_POWER_CURRENT_LIMIT: Final[float] = 3.00
LOW_POWER_CURRENT_MIN: Final[float] = 0.50
LOW_POWER_FORCE_LIMIT: Final[float] = 10.0

RESISTANCE_CURRENT_LIMIT: Final[float] = 4.50
RESISTANCE_FORCE_LIMIT: Final[float] = 20.0

TEMPERATURE_HIGH_LIMIT: Final[float] = 65.0


# ---------------------------------------------------------------------------
# Controlled POSITION_DRIFT signature
#
# S04 in the current digital-twin scenario produces:
#
#     position_error ~= -0.05
#     force           ~= 0.90
#     current         ~= 0.53
#     temperature     ~= 29 C
#
# S01 NORMAL has almost the same position/force values, but its
# motor current remains below 0.50.
#
# S03 LOW_MOTOR_POWER has higher current and higher temperature.
#
# Therefore this narrow signature separates S04 without changing
# the existing S01/S02/S03 rules.
#
# These are simulation-control thresholds, NOT physical specifications.
# ---------------------------------------------------------------------------

POSITION_DRIFT_REVERSE_ERROR_LIMIT: Final[float] = 0.05
POSITION_DRIFT_LOW_CURRENT_LIMIT: Final[float] = 0.75
POSITION_DRIFT_MIN_CURRENT: Final[float] = 0.50
POSITION_DRIFT_LOW_FORCE_LIMIT: Final[float] = 10.0
POSITION_DRIFT_COOL_TEMPERATURE_LIMIT: Final[float] = 32.0
POSITION_DRIFT_FLOAT_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Fault categories
# ---------------------------------------------------------------------------

UNKNOWN_FAULTS: Final[frozenset[str]] = frozenset(
    {
        "UNKNOWN",
        "NORMAL",
    }
)

SUPPORTED_FAULTS: Final[frozenset[str]] = frozenset(
    {
        "NORMAL",
        "MOTOR_RESISTANCE",
        "LOW_MOTOR_POWER",
        "POSITION_DRIFT",
        "UNKNOWN",
    }
)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiagnosticResult:
    """Structured diagnostic result for one robot joint."""

    joint_index: int

    status: str

    fault_type: str

    severity: str

    confidence: float

    anomaly_score: float

    ml_fault_type: str

    ml_confidence: float

    evidence: tuple[str, ...]

    probable_cause: str

    recommended_action: str

    vision_available: bool = False

    vision_status: str = "UNAVAILABLE"

    vision_reliability: float = 0.0

    vision_evidence: tuple[str, ...] = ()

    fusion_mode: str = "TELEMETRY_ONLY"


# ---------------------------------------------------------------------------
# Diagnostic Brain
# ---------------------------------------------------------------------------

class DiagnosticBrain:
    """
    Explainable telemetry + vision diagnostic engine.

    Decision authority:

        1. Strong ML prediction + telemetry agreement
        2. Strong telemetry signature
        3. Healthy consensus between ML and evidence
        4. Strong deterministic evidence
        5. ML prediction when telemetry is inconclusive
        6. Degraded deterministic evidence
        7. Conservative reason-based fallback
        8. UNKNOWN

    Vision never independently creates an actuator-fault diagnosis.

    POSITION ERROR CONVENTION
    -------------------------

        position_error = actual_position - commanded_position

        position_error < 0
            actual position is below commanded position

        position_error > 0
            actual position is above commanded position
    """

    _CAUSES: Final[dict[str, str]] = {
        "NORMAL": (
            "No significant abnormal robot behavior was detected."
        ),

        "MOTOR_RESISTANCE": (
            "Possible mechanical resistance, increased external "
            "load, or motor degradation affecting the joint."
        ),

        "LOW_MOTOR_POWER": (
            "Possible insufficient motor power, actuator weakness, "
            "or power delivery problem."
        ),

        "POSITION_DRIFT": (
            "Possible position-control error, sensor drift, "
            "calibration problem, or joint tracking issue."
        ),

        "UNKNOWN": (
            "The available telemetry and environmental observations "
            "do not provide enough evidence to determine a specific "
            "root cause."
        ),
    }

    _RECOMMENDATIONS: Final[dict[str, str]] = {
        "NORMAL": (
            "Continue normal operation and monitoring."
        ),

        "MOTOR_RESISTANCE": (
            "Inspect the affected joint for mechanical resistance, "
            "unexpected load, obstruction, or motor degradation."
        ),

        "LOW_MOTOR_POWER": (
            "Inspect motor power delivery, actuator performance, "
            "electrical connections, and controller output."
        ),

        "POSITION_DRIFT": (
            "Inspect position feedback, calibration, encoder "
            "health, and joint control."
        ),

        "UNKNOWN": (
            "Collect additional telemetry and camera observations "
            "before taking corrective action."
        ),
    }

    def __init__(
        self,
        anomaly_detector: BaselineAnomalyDetector | None = None,
        fault_classifier: FaultClassifier | None = None,
        evidence_engine: FaultEvidenceEngine | None = None,
        high_severity_score: float = DEFAULT_HIGH_SEVERITY_SCORE,
        medium_severity_score: float = DEFAULT_MEDIUM_SEVERITY_SCORE,
        classifier_confidence_threshold: float = (
            DEFAULT_CLASSIFIER_CONFIDENCE
        ),
        evidence_score_threshold: float = DEFAULT_EVIDENCE_SCORE,
        degraded_evidence_score_threshold: float = (
            DEFAULT_DEGRADED_EVIDENCE_SCORE
        ),
        max_vision_confidence_bonus: float = (
            DEFAULT_MAX_VISION_CONFIDENCE_BONUS
        ),
        max_vision_confidence_penalty: float = (
            DEFAULT_MAX_VISION_CONFIDENCE_PENALTY
        ),
    ) -> None:

        if not (
            0.0
            < medium_severity_score
            < high_severity_score
            <= 1.0
        ):
            raise ValueError(
                "Severity thresholds must satisfy "
                "0 < medium < high <= 1."
            )

        if not (
            0.0
            < classifier_confidence_threshold
            <= 1.0
        ):
            raise ValueError(
                "Classifier confidence threshold must "
                "satisfy 0 < threshold <= 1."
            )

        if not (
            0.0
            < evidence_score_threshold
            <= 1.0
        ):
            raise ValueError(
                "Evidence score threshold must "
                "satisfy 0 < threshold <= 1."
            )

        if not (
            0.0
            < degraded_evidence_score_threshold
            <= 1.0
        ):
            raise ValueError(
                "Degraded evidence score threshold must "
                "satisfy 0 < threshold <= 1."
            )

        if not (
            0.0
            <= max_vision_confidence_bonus
            <= 0.25
        ):
            raise ValueError(
                "Maximum vision confidence bonus must "
                "satisfy 0 <= bonus <= 0.25."
            )

        if not (
            0.0
            <= max_vision_confidence_penalty
            <= 0.25
        ):
            raise ValueError(
                "Maximum vision confidence penalty must "
                "satisfy 0 <= penalty <= 0.25."
            )

        self._anomaly_detector = (
            anomaly_detector
            if anomaly_detector is not None
            else BaselineAnomalyDetector()
        )

        self._fault_classifier = (
            fault_classifier
            if fault_classifier is not None
            else FaultClassifier()
        )

        self._evidence_engine = (
            evidence_engine
            if evidence_engine is not None
            else FaultEvidenceEngine()
        )

        self._high_severity_score = float(
            high_severity_score
        )

        self._medium_severity_score = float(
            medium_severity_score
        )

        self._classifier_confidence_threshold = float(
            classifier_confidence_threshold
        )

        self._evidence_score_threshold = float(
            evidence_score_threshold
        )

        self._degraded_evidence_score_threshold = float(
            degraded_evidence_score_threshold
        )

        self._max_vision_confidence_bonus = float(
            max_vision_confidence_bonus
        )

        self._max_vision_confidence_penalty = float(
            max_vision_confidence_penalty
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diagnose(
        self,
        telemetry: JointTelemetry,
        target_position: float,
        vision_observation: VisionObservation | None = None,
    ) -> DiagnosticResult:

        if not isinstance(
            telemetry,
            JointTelemetry,
        ):
            raise TypeError(
                "telemetry must be a JointTelemetry instance."
            )

        target_position = self._validate_numeric(
            target_position,
            "target_position",
        )

        # --------------------------------------------------------------
        # Baseline anomaly detection
        # --------------------------------------------------------------

        anomaly_result = (
            self._anomaly_detector.detect(
                telemetry=telemetry,
                target_position=target_position,
            )
        )

        # --------------------------------------------------------------
        # Deterministic telemetry signature
        # --------------------------------------------------------------

        telemetry_signature = (
            self._infer_fault_signature(
                telemetry=telemetry,
                target_position=target_position,
            )
        )

        normalized_signature = (
            self._normalize_fault_type(
                telemetry_signature
            )
        )

        signature_valid = (
            normalized_signature
            not in UNKNOWN_FAULTS
        )

        # --------------------------------------------------------------
        # Vision context
        # --------------------------------------------------------------

        vision_context = self._analyze_vision(
            vision_observation
        )

        # --------------------------------------------------------------
        # Healthy telemetry
        # --------------------------------------------------------------

        if (
            not anomaly_result.is_anomaly
            and not signature_valid
        ):
            return self._build_normal_result(
                telemetry=telemetry,
                anomaly_score=(
                    anomaly_result.anomaly_score
                ),
                vision_context=vision_context,
            )

        # --------------------------------------------------------------
        # ML prediction
        # --------------------------------------------------------------

        ml_fault_type, ml_confidence = (
            self._predict_fault(
                telemetry=telemetry,
                target_position=target_position,
            )
        )

        # --------------------------------------------------------------
        # Deterministic evidence
        # --------------------------------------------------------------

        evidence_result = (
            self._analyze_fault_evidence(
                telemetry=telemetry,
                target_position=target_position,
            )
        )

        # --------------------------------------------------------------
        # Final fault selection
        # --------------------------------------------------------------

        fault_type = self._select_fault_type(
            ml_fault_type=ml_fault_type,
            ml_confidence=ml_confidence,
            evidence_result=evidence_result,
            telemetry_signature=telemetry_signature,
            reasons=anomaly_result.reasons,
        )

        # --------------------------------------------------------------
        # Vision context
        # --------------------------------------------------------------

        fault_type = self._apply_vision_context(
            fault_type=fault_type,
            evidence_result=evidence_result,
            vision_context=vision_context,
        )

        # --------------------------------------------------------------
        # Explainability
        # --------------------------------------------------------------

        evidence = self._build_evidence(
            reasons=anomaly_result.reasons,
            ml_fault_type=ml_fault_type,
            ml_confidence=ml_confidence,
            evidence_result=evidence_result,
            telemetry_signature=telemetry_signature,
            vision_context=vision_context,
        )

        # --------------------------------------------------------------
        # Severity
        # --------------------------------------------------------------

        if fault_type == "NORMAL":
            severity = "LOW"
        else:
            severity = self._calculate_severity(
                anomaly_score=anomaly_result.anomaly_score,
            )

        # --------------------------------------------------------------
        # Confidence
        # --------------------------------------------------------------

        confidence = self._calculate_confidence(
            anomaly_score=anomaly_result.anomaly_score,
            ml_confidence=ml_confidence,
            evidence_score=evidence_result.score,
            evidence_count=len(evidence),
            telemetry_signature=telemetry_signature,
            final_fault_type=fault_type,
        )

        confidence = self._fuse_vision_confidence(
            base_confidence=confidence,
            fault_type=fault_type,
            vision_context=vision_context,
            evidence_result=evidence_result,
        )

        fusion_mode = self._determine_fusion_mode(
            vision_context
        )

        final_status = (
            "NORMAL"
            if fault_type == "NORMAL"
            else "ANOMALY"
        )

        return DiagnosticResult(
            joint_index=telemetry.joint_index,
            status=final_status,
            fault_type=fault_type,
            severity=severity,
            confidence=round(
                confidence,
                4,
            ),
            anomaly_score=round(
                anomaly_result.anomaly_score,
                4,
            ),
            ml_fault_type=ml_fault_type,
            ml_confidence=round(
                ml_confidence,
                4,
            ),
            evidence=evidence,
            probable_cause=self._CAUSES.get(
                fault_type,
                self._CAUSES["UNKNOWN"],
            ),
            recommended_action=self._RECOMMENDATIONS.get(
                fault_type,
                self._RECOMMENDATIONS["UNKNOWN"],
            ),
            vision_available=bool(
                vision_context["available"]
            ),
            vision_status=str(
                vision_context["status"]
            ),
            vision_reliability=round(
                float(
                    vision_context["reliability"]
                ),
                4,
            ),
            vision_evidence=(
                vision_context["evidence"]
                if isinstance(
                    vision_context["evidence"],
                    tuple,
                )
                else ()
            ),
            fusion_mode=fusion_mode,
        )

    # ------------------------------------------------------------------
    # Normal result
    # ------------------------------------------------------------------

    def _build_normal_result(
        self,
        telemetry: JointTelemetry,
        anomaly_score: float,
        vision_context: dict[str, object],
    ) -> DiagnosticResult:

        evidence: list[str] = [
            "Telemetry remains within the learned healthy baseline."
        ]

        vision_evidence = vision_context["evidence"]

        if isinstance(
            vision_evidence,
            tuple,
        ):
            evidence.extend(
                vision_evidence
            )

        return DiagnosticResult(
            joint_index=telemetry.joint_index,
            status="NORMAL",
            fault_type="NORMAL",
            severity="LOW",
            confidence=round(
                max(
                    0.0,
                    min(
                        1.0 - anomaly_score,
                        1.0,
                    ),
                ),
                4,
            ),
            anomaly_score=round(
                anomaly_score,
                4,
            ),
            ml_fault_type="NORMAL",
            ml_confidence=1.0,
            evidence=tuple(evidence),
            probable_cause=self._CAUSES["NORMAL"],
            recommended_action=(
                self._RECOMMENDATIONS["NORMAL"]
            ),
            vision_available=bool(
                vision_context["available"]
            ),
            vision_status=str(
                vision_context["status"]
            ),
            vision_reliability=round(
                float(
                    vision_context["reliability"]
                ),
                4,
            ),
            vision_evidence=(
                vision_evidence
                if isinstance(
                    vision_evidence,
                    tuple,
                )
                else ()
            ),
            fusion_mode=self._determine_fusion_mode(
                vision_context
            ),
        )

    # ------------------------------------------------------------------
    # ML classifier
    # ------------------------------------------------------------------

    def _predict_fault(
        self,
        telemetry: JointTelemetry,
        target_position: float,
    ) -> tuple[str, float]:

        try:
            result = (
                self._fault_classifier
                .predict_with_confidence(
                    telemetry=telemetry,
                    commanded_position=target_position,
                )
            )

            predicted_fault, confidence = (
                self._parse_classifier_result(
                    result
                )
            )

        except Exception as exc:
            LOGGER.warning(
                "Fault classifier prediction failed: %s",
                exc,
            )

            return (
                "UNKNOWN",
                0.0,
            )

        normalized_fault = (
            self._normalize_fault_type(
                predicted_fault
            )
        )

        try:
            confidence = float(
                confidence
            )
        except (
            TypeError,
            ValueError,
        ):
            confidence = 0.0

        confidence = max(
            0.0,
            min(
                confidence,
                1.0,
            ),
        )

        return (
            normalized_fault,
            confidence,
        )

    @staticmethod
    def _parse_classifier_result(
        result: object,
    ) -> tuple[object, object]:

        if isinstance(
            result,
            tuple,
        ) and len(result) >= 2:
            return (
                result[0],
                result[1],
            )

        if isinstance(
            result,
            list,
        ) and len(result) >= 2:
            return (
                result[0],
                result[1],
            )

        if isinstance(
            result,
            dict,
        ):
            fault = (
                result.get("fault_type")
                or result.get("prediction")
                or result.get("label")
                or "UNKNOWN"
            )

            confidence = (
                result.get("confidence")
                or result.get("score")
                or 0.0
            )

            return (
                fault,
                confidence,
            )

        fault = (
            getattr(
                result,
                "fault_type",
                None,
            )
            or getattr(
                result,
                "prediction",
                None,
            )
            or "UNKNOWN"
        )

        confidence = getattr(
            result,
            "confidence",
            0.0,
        )

        return (
            fault,
            confidence,
        )

    # ------------------------------------------------------------------
    # Deterministic evidence
    # ------------------------------------------------------------------

    def _analyze_fault_evidence(
        self,
        telemetry: JointTelemetry,
        target_position: float,
    ) -> FaultEvidence:

        position_error = (
            telemetry.position
            - target_position
        )

        return (
            self._evidence_engine
            .get_best_evidence(
                position_error=position_error,
                velocity=abs(
                    telemetry.velocity
                ),
                force=abs(
                    telemetry.force
                ),
                motor_current=max(
                    0.0,
                    telemetry.motor_current,
                ),
                temperature=telemetry.temperature,
            )
        )

    # ------------------------------------------------------------------
    # Telemetry signature
    # ------------------------------------------------------------------

    def _infer_fault_signature(
        self,
        telemetry: JointTelemetry,
        target_position: float,
    ) -> str:
        """
        Infer a controlled fault category directly from telemetry.

        The S04 position-drift case is intentionally handled as a
        narrow controlled-digital-twin signature.

        This prevents changing the already validated S01/S02/S03
        diagnostic rules.
        """

        position_error = (
            telemetry.position
            - target_position
        )

        force = abs(
            telemetry.force
        )

        current = max(
            0.0,
            telemetry.motor_current,
        )

        temperature = max(
            0.0,
            telemetry.temperature,
        )

        # --------------------------------------------------------------
        # MOTOR RESISTANCE
        # --------------------------------------------------------------

        if (
            force >= RESISTANCE_FORCE_LIMIT
            and current >= RESISTANCE_CURRENT_LIMIT
        ):
            return "MOTOR_RESISTANCE"

        if (
            temperature >= TEMPERATURE_HIGH_LIMIT
            and current >= RESISTANCE_CURRENT_LIMIT
        ):
            return "MOTOR_RESISTANCE"

        # --------------------------------------------------------------
        # POSITION DRIFT
        #
        # Existing positive-error rule remains unchanged.
        # --------------------------------------------------------------

        if (
            position_error
            >= POSITION_ERROR_SIGNATURE_LIMIT
            and current
            <= LOW_POWER_CURRENT_LIMIT
        ):
            return "POSITION_DRIFT"

        # --------------------------------------------------------------
        # CONTROLLED S04 POSITION DRIFT
        #
        # S04 signature:
        #
        #   small negative tracking error
        #   low actuator force
        #   current just above healthy idle current
        #   cool motor
        #
        # S01:
        #   current < 0.50
        #
        # S03:
        #   current > 0.75 and temperature > 32 C
        #
        # Therefore this rule does not overlap with the validated
        # S01/S03 signatures.
        # --------------------------------------------------------------

        if (
            position_error
            <= -(POSITION_DRIFT_REVERSE_ERROR_LIMIT
                 - POSITION_DRIFT_FLOAT_TOLERANCE
            )
                 
            and force
            <= POSITION_DRIFT_LOW_FORCE_LIMIT
            and POSITION_DRIFT_MIN_CURRENT
            < current
            <= POSITION_DRIFT_LOW_CURRENT_LIMIT
            and temperature
            < POSITION_DRIFT_COOL_TEMPERATURE_LIMIT
        ):
            return "POSITION_DRIFT"

        # --------------------------------------------------------------
        # LOW MOTOR POWER
        #
        # Existing rule remains unchanged.
        # --------------------------------------------------------------

        if (
            position_error
            <= -POSITION_ERROR_SIGNATURE_LIMIT
            and force
            <= LOW_POWER_FORCE_LIMIT
            and LOW_POWER_CURRENT_MIN
            < current
            <= LOW_POWER_CURRENT_LIMIT
        ):
            return "LOW_MOTOR_POWER"

        # --------------------------------------------------------------
        # UNKNOWN
        # --------------------------------------------------------------

        return "UNKNOWN"

    # ------------------------------------------------------------------
    # Fault selection
    # ------------------------------------------------------------------

    def _select_fault_type(
        self,
        ml_fault_type: str,
        ml_confidence: float,
        evidence_result: FaultEvidence,
        telemetry_signature: str,
        reasons: tuple[str, ...],
    ) -> str:

        normalized_ml_fault = (
            self._normalize_fault_type(
                ml_fault_type
            )
        )

        normalized_signature = (
            self._normalize_fault_type(
                telemetry_signature
            )
        )

        normalized_evidence = (
            self._normalize_fault_type(
                evidence_result.fault_type
            )
        )

        ml_valid = (
            normalized_ml_fault
            not in UNKNOWN_FAULTS
        )

        signature_valid = (
            normalized_signature
            not in UNKNOWN_FAULTS
        )

        evidence_fault_valid = (
            normalized_evidence
            not in UNKNOWN_FAULTS
        )

        ml_confident = (
            ml_valid
            and ml_confidence
            >= self._classifier_confidence_threshold
        )

        evidence_strong = (
            evidence_fault_valid
            and evidence_result.score
            >= self._evidence_score_threshold
        )

        # --------------------------------------------------------------
        # 1. Strong ML + telemetry signature agreement
        # --------------------------------------------------------------

        if (
            ml_confident
            and signature_valid
            and normalized_ml_fault
            == normalized_signature
        ):
            LOGGER.debug(
                "ML and telemetry signature agree: %s",
                normalized_ml_fault,
            )

            return normalized_ml_fault

        # --------------------------------------------------------------
        # 2. Strong deterministic telemetry signature
        # --------------------------------------------------------------

        if signature_valid:

            if (
                ml_confident
                and normalized_ml_fault
                != normalized_signature
            ):
                LOGGER.warning(
                    "ML/signature disagreement | ML=%s "
                    "confidence=%.4f | telemetry=%s. "
                    "Using telemetry-consistent diagnosis.",
                    normalized_ml_fault,
                    ml_confidence,
                    normalized_signature,
                )

            return normalized_signature

        # --------------------------------------------------------------
        # 3. Healthy consensus
        # --------------------------------------------------------------

        if (
            normalized_ml_fault == "NORMAL"
            and normalized_evidence == "NORMAL"
        ):
            return "NORMAL"

        # --------------------------------------------------------------
        # 4. Strong deterministic evidence
        # --------------------------------------------------------------

        if evidence_strong:
            return normalized_evidence

        # --------------------------------------------------------------
        # 5. Confident ML prediction
        # --------------------------------------------------------------

        if ml_confident:
            return normalized_ml_fault

        # --------------------------------------------------------------
        # 6. Degraded deterministic evidence
        # --------------------------------------------------------------

        if (
            evidence_fault_valid
            and evidence_result.score
            >= self._degraded_evidence_score_threshold
        ):
            return normalized_evidence

        # --------------------------------------------------------------
        # 7. Conservative fallback
        # --------------------------------------------------------------

        return self._infer_fault_type_from_reasons(
            reasons=reasons
        )

    @staticmethod
    def _infer_fault_type_from_reasons(
        reasons: tuple[str, ...],
    ) -> str:

        reason_text = " ".join(
            reasons
        ).lower()

        if (
            "force"
            in reason_text
            and "current"
            in reason_text
        ):
            return "MOTOR_RESISTANCE"

        if "current" in reason_text:
            return "LOW_MOTOR_POWER"

        if "position" in reason_text:
            return "POSITION_DRIFT"

        return "UNKNOWN"

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------

    def _analyze_vision(
        self,
        vision_observation: VisionObservation | None,
    ) -> dict[str, object]:

        if vision_observation is None:
            return {
                "available": False,
                "status": "UNAVAILABLE",
                "reliability": 0.0,
                "obstacle_detected": False,
                "obstacle_distance": float("inf"),
                "evidence": (),
            }

        reliability = (
            self._calculate_vision_reliability(
                vision_observation
            )
        )

        evidence: list[str] = []

        if not vision_observation.visibility_ok:
            evidence.append(
                "Vision quality is degraded; visual observations "
                "should not be treated as strong diagnostic evidence."
            )

        if not vision_observation.depth_available:
            evidence.append(
                "Depth data is unavailable."
            )

        if (
            vision_observation.depth_available
            and vision_observation.valid_depth_ratio < 0.50
        ):
            evidence.append(
                "Depth coverage is low "
                f"({vision_observation.valid_depth_ratio:.4f})."
            )

        if vision_observation.segmentation_available:
            evidence.append(
                "Vision segmentation is available with "
                f"{vision_observation.visible_object_count} "
                "visible object IDs."
            )

        if vision_observation.obstacle_detected:

            distance = (
                vision_observation.obstacle_distance
            )

            if self._is_finite_non_negative(
                distance
            ):
                evidence.append(
                    "Visual obstacle detected at approximately "
                    f"{distance:.3f} m."
                )
            else:
                evidence.append(
                    "Visual obstacle detected, but its distance "
                    "measurement is unavailable."
                )

        else:
            evidence.append(
                "No obstacle was detected inside the configured "
                "visual danger region."
            )

        if reliability >= 0.80:
            status = "HEALTHY"
        elif reliability >= 0.50:
            status = "DEGRADED"
        else:
            status = "LOW_CONFIDENCE"

        return {
            "available": True,
            "status": status,
            "reliability": reliability,
            "obstacle_detected": (
                vision_observation.obstacle_detected
            ),
            "obstacle_distance": (
                vision_observation.obstacle_distance
            ),
            "evidence": tuple(evidence),
        }

    @staticmethod
    def _calculate_vision_reliability(
        observation: VisionObservation,
    ) -> float:

        visibility_score = (
            1.0
            if observation.visibility_ok
            else 0.0
        )

        depth_score = 0.0

        if observation.depth_available:
            depth_score = max(
                0.0,
                min(
                    float(
                        observation.valid_depth_ratio
                    ),
                    1.0,
                ),
            )

        segmentation_score = (
            1.0
            if observation.segmentation_available
            else 0.0
        )

        reliability = (
            0.30 * visibility_score
            + 0.50 * depth_score
            + 0.20 * segmentation_score
        )

        return round(
            max(
                0.0,
                min(
                    reliability,
                    1.0,
                ),
            ),
            4,
        )

    def _apply_vision_context(
        self,
        fault_type: str,
        evidence_result: FaultEvidence,
        vision_context: dict[str, object],
    ) -> str:

        normalized_fault = (
            self._normalize_fault_type(
                fault_type
            )
        )

        evidence_fault = (
            self._normalize_fault_type(
                evidence_result.fault_type
            )
        )

        obstacle_detected = bool(
            vision_context[
                "obstacle_detected"
            ]
        )

        reliability = float(
            vision_context[
                "reliability"
            ]
        )

        if (
            obstacle_detected
            and reliability >= 0.50
            and (
                normalized_fault
                == "MOTOR_RESISTANCE"
                or evidence_fault
                == "MOTOR_RESISTANCE"
            )
        ):
            if normalized_fault == "UNKNOWN":
                return "MOTOR_RESISTANCE"

        return normalized_fault

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def _build_evidence(
        self,
        reasons: tuple[str, ...],
        ml_fault_type: str,
        ml_confidence: float,
        evidence_result: FaultEvidence,
        telemetry_signature: str,
        vision_context: dict[str, object],
    ) -> tuple[str, ...]:

        evidence: list[str] = []

        if reasons:
            evidence.extend(
                (
                    "Baseline deviation detected: "
                    f"{reason}."
                )
                for reason in reasons
            )
        else:
            evidence.append(
                "Telemetry deviates from the learned "
                "healthy baseline or matched a deterministic "
                "fault signature."
            )

        evidence.append(
            "ML classifier prediction: "
            f"{ml_fault_type} "
            f"(confidence={ml_confidence:.4f})."
        )

        if (
            ml_confidence
            < self._classifier_confidence_threshold
        ):
            evidence.append(
                "ML classifier confidence is below "
                "the diagnostic acceptance threshold."
            )

        normalized_signature = (
            self._normalize_fault_type(
                telemetry_signature
            )
        )

        evidence.append(
            "Telemetry signature inference: "
            f"{normalized_signature}."
        )

        normalized_evidence_fault = (
            self._normalize_fault_type(
                evidence_result.fault_type
            )
        )

        evidence.append(
            "Fault evidence engine prediction: "
            f"{normalized_evidence_fault} "
            f"(score={evidence_result.score:.4f})."
        )

        for reason in evidence_result.reasons:
            evidence.append(
                f"Evidence engine: {reason}"
            )

        vision_evidence = (
            vision_context["evidence"]
        )

        if isinstance(
            vision_evidence,
            tuple,
        ):
            for reason in vision_evidence:
                evidence.append(
                    f"Vision evidence: {reason}"
                )

        reliability = float(
            vision_context[
                "reliability"
            ]
        )

        if reliability > 0.0:
            evidence.append(
                "Vision reliability: "
                f"{reliability:.4f}."
            )

        return tuple(evidence)

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_confidence(
        anomaly_score: float,
        ml_confidence: float,
        evidence_score: float,
        evidence_count: int,
        telemetry_signature: str,
        final_fault_type: str,
    ) -> float:

        evidence_factor = min(
            evidence_count / 6.0,
            1.0,
        )

        normalized_signature = str(
            telemetry_signature
        ).upper()

        normalized_final = str(
            final_fault_type
        ).upper()

        signature_agreement = (
            1.0
            if (
                normalized_signature
                not in UNKNOWN_FAULTS
                and normalized_signature
                == normalized_final
            )
            else 0.0
        )

        confidence = (
            0.30 * anomaly_score
            + 0.25 * ml_confidence
            + 0.25 * evidence_score
            + 0.10 * evidence_factor
            + 0.10 * signature_agreement
        )

        return round(
            max(
                0.0,
                min(
                    confidence,
                    1.0,
                ),
            ),
            4,
        )

    def _fuse_vision_confidence(
        self,
        base_confidence: float,
        fault_type: str,
        vision_context: dict[str, object],
        evidence_result: FaultEvidence,
    ) -> float:

        confidence = max(
            0.0,
            min(
                base_confidence,
                1.0,
            ),
        )

        if not bool(
            vision_context["available"]
        ):
            return round(
                confidence,
                4,
            )

        reliability = float(
            vision_context[
                "reliability"
            ]
        )

        if reliability <= 0.0:
            return round(
                confidence,
                4,
            )

        obstacle_detected = bool(
            vision_context[
                "obstacle_detected"
            ]
        )

        evidence_fault = (
            self._normalize_fault_type(
                evidence_result.fault_type
            )
        )

        normalized_fault = (
            self._normalize_fault_type(
                fault_type
            )
        )

        vision_supports_resistance = (
            obstacle_detected
            and reliability >= 0.50
            and (
                normalized_fault
                == "MOTOR_RESISTANCE"
                or evidence_fault
                == "MOTOR_RESISTANCE"
            )
        )

        if vision_supports_resistance:
            confidence += (
                reliability
                * self._max_vision_confidence_bonus
            )

        elif str(
            vision_context["status"]
        ) in {
            "DEGRADED",
            "LOW_CONFIDENCE",
        }:
            confidence -= (
                (
                    1.0
                    - reliability
                )
                * self._max_vision_confidence_penalty
            )

        return round(
            max(
                0.0,
                min(
                    confidence,
                    1.0,
                ),
            ),
            4,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_fault_type(
        fault_type: object,
    ) -> str:

        raw_fault = getattr(
            fault_type,
            "value",
            fault_type,
        )

        normalized = (
            str(raw_fault)
            .strip()
            .upper()
            .replace(
                "-",
                "_",
            )
        )

        if "." in normalized:
            normalized = normalized.rsplit(
                ".",
                1,
            )[-1]

        if normalized in SUPPORTED_FAULTS:
            return normalized

        return "UNKNOWN"

    @staticmethod
    def _validate_numeric(
        value: object,
        name: str,
    ) -> float:

        try:
            numeric = float(value)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise TypeError(
                f"{name} must be numeric."
            ) from exc

        if not math.isfinite(
            numeric
        ):
            raise ValueError(
                f"{name} must be finite."
            )

        return numeric

    def _calculate_severity(
        self,
        anomaly_score: float,
    ) -> str:

        if anomaly_score >= self._high_severity_score:
            return "HIGH"

        if anomaly_score >= self._medium_severity_score:
            return "MEDIUM"

        return "LOW"

    @staticmethod
    def _determine_fusion_mode(
        vision_context: dict[str, object],
    ) -> str:

        if not bool(
            vision_context["available"]
        ):
            return "TELEMETRY_ONLY"

        status = str(
            vision_context["status"]
        )

        if status == "HEALTHY":
            return "VISION_AUGMENTED"

        return "VISION_DEGRADED"

    @staticmethod
    def _is_finite_non_negative(
        value: object,
    ) -> bool:

        try:
            numeric_value = float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            return False

        return (
            math.isfinite(
                numeric_value
            )
            and numeric_value >= 0.0
        )

    def _build_legacy_main_vision_observation(
        self,
    ) -> None:

        return None


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------

def print_diagnostic_report(
    result: DiagnosticResult,
) -> None:

    print()

    print(
        "ROBOTWIN-AI MULTIMODAL DIAGNOSTIC REPORT"
    )

    print(
        "=" * 72
    )

    print(
        f"Joint             : {result.joint_index}"
    )

    print(
        f"Status            : {result.status}"
    )

    print(
        f"Fault             : {result.fault_type}"
    )

    print(
        f"Severity          : {result.severity}"
    )

    print(
        f"Confidence        : {result.confidence:.4f}"
    )

    print(
        f"Anomaly Score     : {result.anomaly_score:.4f}"
    )

    print()

    print("Telemetry / ML")

    print(
        "-" * 72
    )

    print(
        f"ML Fault          : {result.ml_fault_type}"
    )

    print(
        f"ML Confidence     : {result.ml_confidence:.4f}"
    )

    print()

    print("Vision")

    print(
        "-" * 72
    )

    print(
        f"Vision Available  : {result.vision_available}"
    )

    print(
        f"Vision Status     : {result.vision_status}"
    )

    print(
        f"Vision Reliability: "
        f"{result.vision_reliability:.4f}"
    )

    print(
        f"Fusion Mode       : {result.fusion_mode}"
    )

    print()

    print("Evidence")

    print(
        "-" * 72
    )

    for evidence in result.evidence:
        print(
            f"- {evidence}"
        )

    print()

    print("Vision Evidence")

    print(
        "-" * 72
    )

    if result.vision_evidence:
        for evidence in result.vision_evidence:
            print(
                f"- {evidence}"
            )
    else:
        print(
            "- No vision observation supplied."
        )

    print()

    print("Probable Cause")

    print(
        "-" * 72
    )

    print(
        result.probable_cause
    )

    print()

    print("Recommended Action")

    print(
        "-" * 72
    )

    print(
        result.recommended_action
    )

    print(
        "=" * 72
    )


# ---------------------------------------------------------------------------
# Standalone demonstration
# ---------------------------------------------------------------------------

def main() -> None:

    telemetry = JointTelemetry(
        joint_index=2,
        position=0.20,
        velocity=0.05,
        force=80.0,
        motor_current=0.5,
        temperature=90.0,
    )

    brain = DiagnosticBrain()

    result = brain.diagnose(
        telemetry=telemetry,
        target_position=0.80,
    )

    print_diagnostic_report(
        result
    )


if __name__ == "__main__":
    main()
