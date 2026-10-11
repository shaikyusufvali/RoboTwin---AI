
"""Baseline-based anomaly detection for RoboTwin-AI."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from robot_twin.telemetry.robot_adapter import JointTelemetry


DEFAULT_BASELINE_FILE: Final[Path] = (
    Path("data") / "healthy_baseline.json"
)

DEFAULT_Z_SCORE_THRESHOLD: Final[float] = 3.0
MIN_STANDARD_DEVIATION: Final[float] = 1e-9

POSITION_ERROR_WARNING_THRESHOLD: Final[float] = 0.20
POSITION_ERROR_SEVERE_THRESHOLD: Final[float] = 1.00


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


class BaselineAnomalyDetector:
    """Detect telemetry anomalies using a healthy robot baseline."""

    _ANALYZED_FEATURES: Final[tuple[str, ...]] = (
        "velocity",
        "force",
        "motor_current",
        "temperature",
        "position_error",
    )

    # Position error is command-dependent, so it is not loaded
    # from the fixed per-joint baseline.
    _BASELINE_FEATURES: Final[tuple[str, ...]] = (
        "velocity",
        "force",
        "motor_current",
        "temperature",
    )

    def __init__(
        self,
        baseline_file: Path = DEFAULT_BASELINE_FILE,
        z_score_threshold: float = DEFAULT_Z_SCORE_THRESHOLD,
    ) -> None:
        """Initialize the detector and load the baseline."""

        self._baseline_file = Path(baseline_file)
        self._z_score_threshold = float(z_score_threshold)

        self._validate_configuration()
        self._baseline = self._load_baseline()

    def detect(
        self,
        telemetry: JointTelemetry,
        target_position: float,
    ) -> BaselineAnomalyResult:
        """Compare telemetry with the baseline."""

        joint_baseline = self._get_joint_baseline(
            int(telemetry.joint_index)
        )

        target_position = float(target_position)

        # Keep the same error sign convention used by the
        # project's diagnostic brain.
        position_error = (
            float(telemetry.position) - target_position
        )

        actual_values = {
            "velocity": float(telemetry.velocity),
            "force": float(telemetry.force),
            "motor_current": float(telemetry.motor_current),
            "temperature": float(telemetry.temperature),
            "position_error": position_error,
        }

        z_scores: dict[str, float] = {}
        reasons: list[str] = []
        physical_z_scores: list[float] = []

        for feature_name in self._ANALYZED_FEATURES:

            actual_value = actual_values[feature_name]

            # Position error does not have fixed baseline statistics.
            # It is checked separately below.
            if feature_name == "position_error":
                z_scores[feature_name] = 0.0
                continue

            statistics = joint_baseline[feature_name]

            z_score = self._calculate_z_score(
                actual_value=actual_value,
                baseline=statistics,
            )

            z_scores[feature_name] = z_score
            physical_z_scores.append(abs(z_score))

            if abs(z_score) >= self._z_score_threshold:
                reasons.append(
                    self._build_reason(
                        feature_name=feature_name,
                        z_score=z_score,
                        actual_value=actual_value,
                        baseline=statistics,
                    )
                )

        # Evaluate command tracking independently from the
        # physical-feature baseline.
        absolute_position_error = abs(position_error)
        position_error_score = 0.0

        if (
            absolute_position_error
            >= POSITION_ERROR_SEVERE_THRESHOLD
        ):
            position_error_score = 1.0

            reasons.append(
                "Command tracking error is severe "
                f"(actual={telemetry.position:.4f}, "
                f"target={target_position:.4f}, "
                f"position_error={position_error:.4f})."
            )

        elif (
            absolute_position_error
            >= POSITION_ERROR_WARNING_THRESHOLD
        ):
            position_error_score = 0.50

            reasons.append(
                "Command tracking error exceeds the warning "
                f"threshold (actual={telemetry.position:.4f}, "
                f"target={target_position:.4f}, "
                f"position_error={position_error:.4f})."
            )

        maximum_physical_z = max(
            physical_z_scores,
            default=0.0,
        )

        physical_score = min(
            maximum_physical_z
            / (self._z_score_threshold * 2.0),
            1.0,
        )

        anomaly_score = max(
            physical_score,
            position_error_score,
        )

        return BaselineAnomalyResult(
            is_anomaly=bool(reasons),
            anomaly_score=float(anomaly_score),
            reasons=reasons,
            z_scores=z_scores,
        )

    @staticmethod
    def _calculate_z_score(
        actual_value: float,
        baseline: FeatureBaseline,
    ) -> float:
        """Calculate a numerically safe z-score."""

        standard_deviation = max(
            abs(float(baseline.standard_deviation)),
            MIN_STANDARD_DEVIATION,
        )

        return float(
            (
                float(actual_value)
                - float(baseline.mean)
            )
            / standard_deviation
        )

    def _get_joint_baseline(
        self,
        joint_index: int,
    ) -> dict[str, FeatureBaseline]:
        """Return baseline statistics for one joint."""

        if joint_index not in self._baseline:
            raise ValueError(
                f"No baseline found for joint {joint_index} "
                f"in '{self._baseline_file}'."
            )

        return self._baseline[joint_index]

    def _load_baseline(
        self,
    ) -> dict[int, dict[str, FeatureBaseline]]:
        """Load baseline statistics from JSON.

        Supports both standard_deviation and legacy std keys.
        """

        if not self._baseline_file.exists():
            raise FileNotFoundError(
                "Baseline file not found: "
                f"{self._baseline_file}"
            )

        try:
            with self._baseline_file.open(
                "r",
                encoding="utf-8",
            ) as file:
                payload = json.load(file)

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Baseline file contains invalid JSON: "
                f"{self._baseline_file}"
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                "Baseline JSON root must be an object."
            )

        joints = payload.get("joints")

        if not isinstance(joints, list):
            raise ValueError(
                "Baseline file does not contain a valid "
                "'joints' list."
            )

        baseline: dict[
            int,
            dict[str, FeatureBaseline],
        ] = {}

        for joint in joints:

            if not isinstance(joint, dict):
                raise ValueError(
                    "Each item in 'joints' must be an object."
                )

            if "joint_index" not in joint:
                raise ValueError(
                    "A joint entry is missing 'joint_index'."
                )

            joint_index = int(joint["joint_index"])

            if joint_index in baseline:
                raise ValueError(
                    f"Duplicate baseline entry for joint "
                    f"{joint_index}."
                )

            feature_baselines: dict[
                str,
                FeatureBaseline,
            ] = {}

            for feature_name in self._BASELINE_FEATURES:

                statistics = joint.get(feature_name)

                if not isinstance(statistics, dict):
                    raise ValueError(
                        "Missing baseline statistics for "
                        f"joint {joint_index}, "
                        f"feature '{feature_name}'."
                    )

                if "mean" not in statistics:
                    raise ValueError(
                        f"Missing 'mean' for joint "
                        f"{joint_index}, feature "
                        f"'{feature_name}'."
                    )

                # Accept both supported JSON formats.
                if "standard_deviation" in statistics:
                    raw_standard_deviation = (
                        statistics["standard_deviation"]
                    )

                elif "std" in statistics:
                    raw_standard_deviation = (
                        statistics["std"]
                    )

                else:
                    raise ValueError(
                        "Missing 'standard_deviation' or "
                        f"'std' for joint {joint_index}, "
                        f"feature '{feature_name}'."
                    )

                mean = float(statistics["mean"])

                standard_deviation = float(
                    raw_standard_deviation
                )

                if not math.isfinite(mean):
                    raise ValueError(
                        f"Non-finite mean for joint "
                        f"{joint_index}, feature "
                        f"'{feature_name}'."
                    )

                if not math.isfinite(standard_deviation):
                    raise ValueError(
                        "Non-finite standard deviation for "
                        f"joint {joint_index}, "
                        f"feature '{feature_name}'."
                    )

                # Constant or near-constant signals are allowed.
                # Protect the denominator instead of rejecting
                # the entire baseline.
                standard_deviation = max(
                    abs(standard_deviation),
                    MIN_STANDARD_DEVIATION,
                )

                feature_baselines[feature_name] = (
                    FeatureBaseline(
                        mean=mean,
                        standard_deviation=standard_deviation,
                    )
                )

            baseline[joint_index] = feature_baselines

        if not baseline:
            raise ValueError(
                "Baseline file does not contain "
                "any joint statistics."
            )

        return baseline

    @staticmethod
    def _build_reason(
        feature_name: str,
        z_score: float,
        actual_value: float,
        baseline: FeatureBaseline,
    ) -> str:
        """Create a human-readable anomaly explanation."""

        direction = (
            "above"
            if z_score > 0
            else "below"
        )

        return (
            f"{feature_name} is {direction} the "
            f"healthy baseline "
            f"(actual={actual_value:.4f}, "
            f"baseline_mean={baseline.mean:.4f}, "
            f"z_score={z_score:.2f})."
        )

    def _validate_configuration(self) -> None:
        """Validate detector configuration."""

        if not math.isfinite(self._z_score_threshold):
            raise ValueError(
                "z_score_threshold must be finite."
            )

        if self._z_score_threshold <= 0:
            raise ValueError(
                "z_score_threshold must be greater "
                "than zero."
            )


def main() -> None:
    """Load the baseline and run command-aware smoke tests."""

    detector = BaselineAnomalyDetector()

    print()
    print("ROBOTWIN-AI BASELINE ANOMALY DETECTOR")
    print("=" * 70)
    print(f"Baseline file : {detector._baseline_file}")
    print(f"Joints loaded : {len(detector._baseline)}")
    print("Baseline loaded successfully.")

    healthy_telemetry = JointTelemetry(
        joint_index=2,
        position=0.8,
        velocity=0.0,
        force=0.34,
        motor_current=0.34,
        temperature=27.7,
    )

    abnormal_telemetry = JointTelemetry(
        joint_index=2,
        position=0.5,
        velocity=0.1,
        force=25.0,
        motor_current=2.8,
        temperature=61.0,
    )

    healthy_result = detector.detect(
        telemetry=healthy_telemetry,
        target_position=0.8,
    )

    abnormal_result = detector.detect(
        telemetry=abnormal_telemetry,
        target_position=0.8,
    )

    print()
    print("HEALTHY COMMAND-AWARE TEST")
    print("-" * 70)
    print(
        f"Anomaly detected : "
        f"{healthy_result.is_anomaly}"
    )
    print(
        f"Anomaly score    : "
        f"{healthy_result.anomaly_score:.4f}"
    )
    print(f"Z-scores         : {healthy_result.z_scores}")

    print()
    print("ABNORMAL COMMAND-AWARE TEST")
    print("-" * 70)
    print(
        f"Anomaly detected : "
        f"{abnormal_result.is_anomaly}"
    )
    print(
        f"Anomaly score    : "
        f"{abnormal_result.anomaly_score:.4f}"
    )
    print(f"Z-scores         : {abnormal_result.z_scores}")

    for reason in abnormal_result.reasons:
        print(f"- {reason}")

    print("=" * 70)


if __name__ == "__main__":
    main()

