
"""
Baseline-based anomaly detection for RoboTwin-AI.

This module compares live joint telemetry against a healthy
telemetry baseline and produces deterministic anomaly evidence.

Design goals:
- Load both new and legacy baseline JSON formats.
- Support zero-variance baseline features safely.
- Avoid treating absolute position as a generic anomaly.
- Use velocity, force, motor current and temperature for
  baseline anomaly detection.
- Keep command-dependent position-error reasoning available
  to the existing diagnostic pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from robot_twin.telemetry.robot_adapter import JointTelemetry


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_BASELINE_FILE: Final[Path] = (
    Path("data") / "healthy_baseline.json"
)

DEFAULT_Z_SCORE_THRESHOLD: Final[float] = 3.0

# Prevent division by zero when a healthy baseline feature
# has zero or extremely small variance.
MIN_STANDARD_DEVIATION: Final[float] = 1e-9

# Position is command-dependent.
# Therefore position error is handled separately from
# the physical baseline z-score calculation.
POSITION_ERROR_WARNING_THRESHOLD: Final[float] = 0.20
POSITION_ERROR_SEVERE_THRESHOLD: Final[float] = 1.00


# ============================================================================
# Data structures
# ============================================================================

@dataclass(frozen=True)
class FeatureBaseline:
    """Statistical baseline for one telemetry feature."""

    mean: float
    standard_deviation: float


@dataclass(frozen=True)
class BaselineAnomalyResult:
    """Result produced by baseline anomaly detection."""

    is_anomaly: bool
    anomaly_score: float
    reasons: list[str]
    z_scores: dict[str, float]


# ============================================================================
# Baseline anomaly detector
# ============================================================================

class BaselineAnomalyDetector:
    """
    Detect telemetry anomalies using a healthy robot baseline.

    Absolute joint position is NOT directly compared against
    the healthy position baseline because position depends on
    the commanded robot state.

    Physical baseline features:

        velocity
        force
        motor_current
        temperature

    Position error remains available to the diagnostic pipeline:

        position_error = actual_position - target_position
    """

    # ------------------------------------------------------------------------
    # Features used during live anomaly scoring
    # ------------------------------------------------------------------------

    _ANALYZED_FEATURES: Final[tuple[str, ...]] = (
        "velocity",
        "force",
        "motor_current",
        "temperature",
        "position_error",
    )

    # ------------------------------------------------------------------------
    # Features stored inside healthy_baseline.json
    #
    # IMPORTANT:
    #
    # "position" is intentionally excluded.
    # "position_error" is intentionally excluded.
    #
    # Both depend on the operating/commanded condition.
    # ------------------------------------------------------------------------

    _BASELINE_FEATURES: Final[tuple[str, ...]] = (
        "velocity",
        "force",
        "motor_current",
        "temperature",
    )

    # ------------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------------

    def __init__(
        self,
        baseline_file: Path | str = DEFAULT_BASELINE_FILE,
        z_score_threshold: float = DEFAULT_Z_SCORE_THRESHOLD,
    ) -> None:
        """Initialize the baseline anomaly detector."""

        self._baseline_file = Path(
            baseline_file
        )

        self._z_score_threshold = float(
            z_score_threshold
        )

        self._validate_configuration()

        self._baseline = self._load_baseline()

    # ------------------------------------------------------------------------
    # Configuration validation
    # ------------------------------------------------------------------------

    def _validate_configuration(self) -> None:
        """Validate detector configuration."""

        if self._z_score_threshold <= 0.0:
            raise ValueError(
                "z_score_threshold must be greater than zero."
            )

    # ------------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------------

    def detect(
        self,
        telemetry: JointTelemetry,
        target_position: float,
    ) -> BaselineAnomalyResult:
        """
        Compare telemetry against the healthy baseline.

        Absolute position is intentionally excluded from
        baseline anomaly scoring.

        Position tracking is evaluated through:

            position_error = actual_position - target_position

        Position error is command-dependent, so it is not
        compared against a fixed baseline distribution.
        """

        joint_baseline = self._get_joint_baseline(
            telemetry.joint_index
        )

        # --------------------------------------------------------------------
        # Command-aware position error
        # --------------------------------------------------------------------

        position_error = (
            float(telemetry.position)
            - float(target_position)
        )

        actual_values: dict[str, float] = {
            "velocity": float(
                telemetry.velocity
            ),
            "force": float(
                telemetry.force
            ),
            "motor_current": float(
                telemetry.motor_current
            ),
            "temperature": float(
                telemetry.temperature
            ),
            "position_error": position_error,
        }

        z_scores: dict[str, float] = {}
        reasons: list[str] = []

        # --------------------------------------------------------------------
        # Physical telemetry baseline comparison
        # --------------------------------------------------------------------

        for feature_name in self._ANALYZED_FEATURES:

            # Position error is not stored in healthy_baseline.json.
            #
            # It is calculated dynamically from the actual command.
            if feature_name == "position_error":

                z_scores[feature_name] = 0.0

                continue

            statistics = joint_baseline.get(
                feature_name
            )

            # This should not normally happen because the loader validates
            # every required baseline feature.
            if statistics is None:
                raise KeyError(
                    "Baseline feature "
                    f"'{feature_name}' is missing for "
                    f"joint {telemetry.joint_index}."
                )

            actual_value = actual_values[
                feature_name
            ]

            z_score = self._calculate_z_score(
                actual_value=actual_value,
                baseline=statistics,
            )

            z_scores[feature_name] = z_score

            if abs(z_score) >= (
                self._z_score_threshold
            ):
                reasons.append(
                    self._build_anomaly_reason(
                        feature_name=feature_name,
                        actual_value=actual_value,
                        baseline=statistics,
                        z_score=z_score,
                    )
                )

        # --------------------------------------------------------------------
        # Position-error evidence
        # --------------------------------------------------------------------

        if abs(position_error) >= (
            POSITION_ERROR_WARNING_THRESHOLD
        ):
            reasons.append(
                "Position error is significant "
                f"(actual={telemetry.position:.4f}, "
                f"target={target_position:.4f}, "
                f"error={position_error:.4f})."
            )

        # --------------------------------------------------------------------
        # Physical anomaly score
        # --------------------------------------------------------------------

        physical_z_scores = [
            abs(
                z_scores[feature_name]
            )
            for feature_name in (
                "velocity",
                "force",
                "motor_current",
                "temperature",
            )
        ]

        maximum_z_score = max(
            physical_z_scores,
            default=0.0,
        )

        anomaly_score = min(
            maximum_z_score
            / self._z_score_threshold,
            1.0,
        )

        # --------------------------------------------------------------------
        # Position-error contribution
        # --------------------------------------------------------------------

        if abs(position_error) >= (
            POSITION_ERROR_SEVERE_THRESHOLD
        ):
            anomaly_score = max(
                anomaly_score,
                1.0,
            )

        elif abs(position_error) >= (
            POSITION_ERROR_WARNING_THRESHOLD
        ):
            anomaly_score = max(
                anomaly_score,
                0.50,
            )

        # --------------------------------------------------------------------
        # Final result
        # --------------------------------------------------------------------

        return BaselineAnomalyResult(
            is_anomaly=bool(reasons),
            anomaly_score=float(
                anomaly_score
            ),
            reasons=reasons,
            z_scores=z_scores,
        )

    # ------------------------------------------------------------------------
    # Z-score calculation
    # ------------------------------------------------------------------------

    @staticmethod
    def _calculate_z_score(
        actual_value: float,
        baseline: FeatureBaseline,
    ) -> float:
        """
        Calculate a safe z-score.

        Near-zero standard deviation is clamped to
        MIN_STANDARD_DEVIATION so healthy zero-variance
        features do not cause division-by-zero errors.
        """

        standard_deviation = max(
            abs(
                float(
                    baseline.standard_deviation
                )
            ),
            MIN_STANDARD_DEVIATION,
        )

        return (
            float(actual_value)
            - float(baseline.mean)
        ) / standard_deviation

    # ------------------------------------------------------------------------
    # Baseline loader
    # ------------------------------------------------------------------------

    def _load_baseline(
        self,
    ) -> dict[
        int,
        dict[str, FeatureBaseline],
    ]:
        """
        Load baseline statistics from healthy_baseline.json.

        Supported standard-deviation field names:

            standard_deviation
            std

        New RoboTwin-AI baseline files use:

            standard_deviation

        Older baseline files using:

            std

        remain supported.
        """

        if not self._baseline_file.exists():
            raise FileNotFoundError(
                "Baseline file not found: "
                f"{self._baseline_file}"
            )

        with self._baseline_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)

        if not isinstance(
            payload,
            dict,
        ):
            raise ValueError(
                "Baseline JSON root must be an object."
            )

        joints = payload.get(
            "joints"
        )

        if not isinstance(
            joints,
            list,
        ):
            raise ValueError(
                "Baseline file does not contain "
                "a valid 'joints' list."
            )

        baseline: dict[
            int,
            dict[str, FeatureBaseline],
        ] = {}

        for joint in joints:

            if not isinstance(
                joint,
                dict,
            ):
                raise ValueError(
                    "Each item in 'joints' must "
                    "be an object."
                )

            if "joint_index" not in joint:
                raise ValueError(
                    "A joint entry is missing "
                    "'joint_index'."
                )

            try:
                joint_index = int(
                    joint["joint_index"]
                )

            except (
                TypeError,
                ValueError,
            ) as exc:

                raise ValueError(
                    "Invalid joint_index in "
                    "baseline: "
                    f"{joint.get('joint_index')!r}"
                ) from exc

            # --------------------------------------------------------------
            # Prevent duplicate joint entries.
            # --------------------------------------------------------------

            if joint_index in baseline:
                raise ValueError(
                    "Duplicate baseline entry "
                    f"for joint {joint_index}."
                )

            feature_baselines: dict[
                str,
                FeatureBaseline,
            ] = {}

            # --------------------------------------------------------------
            # Load every required physical feature.
            # --------------------------------------------------------------

            for feature_name in (
                self._BASELINE_FEATURES
            ):

                statistics = joint.get(
                    feature_name
                )

                if not isinstance(
                    statistics,
                    dict,
                ):
                    raise ValueError(
                        "Missing baseline statistics "
                        f"for joint {joint_index}, "
                        f"feature '{feature_name}'."
                    )

                # ----------------------------------------------------------
                # Mean
                # ----------------------------------------------------------

                if "mean" not in statistics:
                    raise ValueError(
                        "Missing 'mean' for joint "
                        f"{joint_index}, feature "
                        f"'{feature_name}'."
                    )

                try:
                    mean = float(
                        statistics["mean"]
                    )

                except (
                    TypeError,
                    ValueError,
                ) as exc:

                    raise ValueError(
                        "Invalid mean for joint "
                        f"{joint_index}, feature "
                        f"'{feature_name}': "
                        f"{statistics.get('mean')!r}"
                    ) from exc

                # ----------------------------------------------------------
                # Standard deviation
                #
                # New format:
                #     standard_deviation
                #
                # Legacy format:
                #     std
                # ----------------------------------------------------------

                if (
                    "standard_deviation"
                    in statistics
                ):

                    raw_standard_deviation = (
                        statistics[
                            "standard_deviation"
                        ]
                    )

                elif "std" in statistics:

                    raw_standard_deviation = (
                        statistics[
                            "std"
                        ]
                    )

                else:

                    raise ValueError(
                        "Missing standard deviation "
                        f"for joint {joint_index}, "
                        f"feature '{feature_name}'. "
                        "Expected "
                        "'standard_deviation' "
                        "or 'std'."
                    )

                try:
                    standard_deviation = float(
                        raw_standard_deviation
                    )

                except (
                    TypeError,
                    ValueError,
                ) as exc:

                    raise ValueError(
                        "Invalid standard deviation "
                        f"for joint {joint_index}, "
                        f"feature '{feature_name}': "
                        f"{raw_standard_deviation!r}"
                    ) from exc

                # ----------------------------------------------------------
                # Zero / near-zero variance protection
                # ----------------------------------------------------------

                standard_deviation = max(
                    abs(
                        standard_deviation
                    ),
                    MIN_STANDARD_DEVIATION,
                )

                feature_baselines[
                    feature_name
                ] = FeatureBaseline(
                    mean=mean,
                    standard_deviation=(
                        standard_deviation
                    ),
                )

            baseline[
                joint_index
            ] = feature_baselines

        # --------------------------------------------------------------------
        # Final validation
        # --------------------------------------------------------------------

        if not baseline:
            raise ValueError(
                "Baseline file does not contain "
                "any joint statistics."
            )

        return baseline

    # ------------------------------------------------------------------------
    # Baseline access
    # ------------------------------------------------------------------------

    def _get_joint_baseline(
        self,
        joint_index: int,
    ) -> dict[
        str,
        FeatureBaseline,
    ]:
        """Return baseline statistics for one joint."""

        if joint_index not in self._baseline:
            raise KeyError(
                "No healthy baseline exists "
                f"for joint {joint_index}."
            )

        return self._baseline[
            joint_index
        ]

    # ------------------------------------------------------------------------
    # Reason generation
    # ------------------------------------------------------------------------

    @staticmethod
    def _build_anomaly_reason(
        feature_name: str,
        actual_value: float,
        baseline: FeatureBaseline,
        z_score: float,
    ) -> str:
        """Build a human-readable anomaly explanation."""

        direction = (
            "above"
            if z_score > 0.0
            else "below"
        )

        return (
            "Baseline deviation detected: "
            f"{feature_name} is {direction} "
            "the healthy baseline "
            f"(actual={actual_value:.4f}, "
            f"baseline_mean={baseline.mean:.4f}, "
            f"z_score={z_score:.2f})."
        )


# ============================================================================
# Standalone verification
# ============================================================================

def main() -> None:
    """Verify that the healthy baseline loads correctly."""

    print(
        "RoboTwin-AI Baseline Anomaly Detector"
    )

    print(
        "=" * 50
    )

    detector = BaselineAnomalyDetector()

    print(
        f"Baseline file : "
        f"{detector._baseline_file}"
    )

    print(
        f"Z-score limit : "
        f"{detector._z_score_threshold}"
    )

    print(
        f"Joints loaded : "
        f"{len(detector._baseline)}"
    )

    print(
        "Features      : "
        + ", ".join(
            detector._BASELINE_FEATURES
        )
    )

    print()

    print(
        "Baseline loaded successfully."
    )

    print(
        "Baseline format supports:"
    )

    print(
        "  - standard_deviation"
    )

    print(
        "  - std"
    )

    print(
        "Zero/near-zero variance is "
        "safely clamped."
    )

    print(
        "Position is excluded from "
        "generic baseline scoring."
    )

    print(
        "Position error remains "
        "command-aware."
    )


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    main()
