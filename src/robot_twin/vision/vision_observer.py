
"""Robot visual observation layer for RoboTwin-AI.

This module converts raw camera frames into structured observations
that can be consumed by navigation, safety, and diagnostic systems.

The observer intentionally does not make high-level fault claims.
It reports measurable visual evidence only.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np

from robot_twin.vision.camera import CameraFrame, RobotCamera
from robot_twin.vision.obstacle_detector import ObstacleDetector


LOGGER = logging.getLogger(__name__)


DEFAULT_MIN_VALID_DEPTH_METERS: Final[float] = 0.10
DEFAULT_MAX_VALID_DEPTH_METERS: Final[float] = 10.0
DEFAULT_MIN_VALID_DEPTH_RATIO: Final[float] = 0.10
DEFAULT_MIN_RGB_MEAN: Final[float] = 1.0


class CameraSource(Protocol):
    """Interface required by the visual observation pipeline."""

    def capture(self) -> CameraFrame:
        """Capture one camera frame."""


@dataclass(frozen=True)
class VisionObservation:
    """Structured visual evidence produced by the observation layer."""

    obstacle_detected: bool
    obstacle_distance: float

    visibility_ok: bool
    depth_available: bool
    segmentation_available: bool

    valid_depth_ratio: float
    visible_object_count: int

    image_width: int
    image_height: int

    reason: str


class VisionObserver:
    """Convert camera data into diagnostic-friendly visual evidence.

    The observer is deliberately independent from the final diagnostic
    decision. It captures a frame, validates the visual data, invokes
    obstacle detection, and returns structured evidence.

    A real robot camera can later implement the same CameraSource
    interface without changing the diagnostic layer.
    """

    def __init__(
        self,
        camera: CameraSource,
        obstacle_detector: ObstacleDetector,
        min_valid_depth_meters: float = (
            DEFAULT_MIN_VALID_DEPTH_METERS
        ),
        max_valid_depth_meters: float = (
            DEFAULT_MAX_VALID_DEPTH_METERS
        ),
        min_valid_depth_ratio: float = (
            DEFAULT_MIN_VALID_DEPTH_RATIO
        ),
        min_rgb_mean: float = DEFAULT_MIN_RGB_MEAN,
    ) -> None:
        self._camera = camera
        self._obstacle_detector = obstacle_detector

        self._min_valid_depth_meters = (
            min_valid_depth_meters
        )
        self._max_valid_depth_meters = (
            max_valid_depth_meters
        )
        self._min_valid_depth_ratio = (
            min_valid_depth_ratio
        )
        self._min_rgb_mean = min_rgb_mean

        self._validate_configuration()

    def observe(self) -> VisionObservation:
        """Capture and evaluate one visual observation."""

        frame = self._camera.capture()

        self._validate_frame(frame)

        image_height, image_width = frame.rgb.shape[:2]

        depth_available = self._has_valid_depth(
            frame.depth
        )

        segmentation_available = (
            self._has_valid_segmentation(
                frame.segmentation
            )
        )

        valid_depth_ratio = (
            self._calculate_valid_depth_ratio(
                frame.depth
            )
        )

        visible_object_count = (
            self._count_visible_objects(
                frame.segmentation
            )
        )

        visual_quality_ok = (
            self._is_visual_quality_acceptable(
                frame=frame,
                valid_depth_ratio=valid_depth_ratio,
            )
        )

        obstacle_result = (
            self._obstacle_detector.detect(frame)
        )

        visibility_ok = (
            visual_quality_ok
            and depth_available
            and segmentation_available
        )

        reason = self._build_reason(
            visibility_ok=visibility_ok,
            visual_quality_ok=visual_quality_ok,
            depth_available=depth_available,
            segmentation_available=(
                segmentation_available
            ),
            obstacle_reason=obstacle_result.reason,
        )

        observation = VisionObservation(
            obstacle_detected=bool(
                obstacle_result.detected
            ),
            obstacle_distance=float(
                obstacle_result.minimum_distance
            ),
            visibility_ok=visibility_ok,
            depth_available=depth_available,
            segmentation_available=(
                segmentation_available
            ),
            valid_depth_ratio=valid_depth_ratio,
            visible_object_count=visible_object_count,
            image_width=image_width,
            image_height=image_height,
            reason=reason,
        )

        LOGGER.debug(
            "Visual observation captured | "
            "obstacle=%s | distance=%.3f m | "
            "visibility=%s | depth_ratio=%.3f | "
            "objects=%d",
            observation.obstacle_detected,
            observation.obstacle_distance,
            observation.visibility_ok,
            observation.valid_depth_ratio,
            observation.visible_object_count,
        )

        return observation

    def _is_visual_quality_acceptable(
        self,
        frame: CameraFrame,
        valid_depth_ratio: float,
    ) -> bool:
        """Determine whether the captured frame is usable."""

        rgb = frame.rgb

        if rgb.size == 0:
            return False

        if rgb.ndim != 3:
            return False

        if rgb.shape[2] != 3:
            return False

        if not np.isfinite(rgb).all():
            return False

        rgb_mean = float(
            np.mean(rgb)
        )

        if rgb_mean < self._min_rgb_mean:
            return False

        if (
            valid_depth_ratio
            < self._min_valid_depth_ratio
        ):
            return False

        return True

    def _has_valid_depth(
        self,
        depth: np.ndarray,
    ) -> bool:
        """Return whether the depth frame contains usable data."""

        if depth.size == 0:
            return False

        if depth.ndim != 2:
            return False

        finite_depth = np.isfinite(depth)

        if not finite_depth.any():
            return False

        valid_depth = (
            finite_depth
            & (depth >= self._min_valid_depth_meters)
            & (depth <= self._max_valid_depth_meters)
        )

        return bool(
            np.any(valid_depth)
        )

    def _has_valid_segmentation(
        self,
        segmentation: np.ndarray,
    ) -> bool:
        """Return whether segmentation data is structurally valid."""

        if segmentation.size == 0:
            return False

        if segmentation.ndim != 2:
            return False

        if not np.isfinite(segmentation).all():
            return False

        return True

    def _calculate_valid_depth_ratio(
        self,
        depth: np.ndarray,
    ) -> float:
        """Calculate the fraction of pixels with valid depth."""

        if depth.size == 0:
            return 0.0

        finite_depth = np.isfinite(depth)

        valid_depth = (
            finite_depth
            & (depth >= self._min_valid_depth_meters)
            & (depth <= self._max_valid_depth_meters)
        )

        return float(
            np.count_nonzero(valid_depth)
            / depth.size
        )

    @staticmethod
    def _count_visible_objects(
        segmentation: np.ndarray,
    ) -> int:
        """Count unique segmentation IDs visible in the frame."""

        if segmentation.size == 0:
            return 0

        unique_ids = np.unique(
            segmentation
        )

        return int(
            unique_ids.size
        )

    @staticmethod
    def _build_reason(
        visibility_ok: bool,
        visual_quality_ok: bool,
        depth_available: bool,
        segmentation_available: bool,
        obstacle_reason: str,
    ) -> str:
        """Build a concise explanation of the visual observation."""

        if not visual_quality_ok:
            return (
                "Visual frame quality is insufficient "
                "for reliable observation."
            )

        if not depth_available:
            return (
                "RGB frame is available, but usable "
                "depth information is unavailable."
            )

        if not segmentation_available:
            return (
                "RGB and depth are available, but "
                "segmentation information is unavailable."
            )

        if not visibility_ok:
            return (
                "Visual observation is incomplete."
            )

        return str(
            obstacle_reason
        )

    @staticmethod
    def _validate_frame(
        frame: CameraFrame,
    ) -> None:
        """Validate the basic structure of a camera frame."""

        if not isinstance(
            frame,
            CameraFrame,
        ):
            raise TypeError(
                "camera.capture() must return "
                "a CameraFrame instance."
            )

        if not isinstance(
            frame.rgb,
            np.ndarray,
        ):
            raise TypeError(
                "CameraFrame.rgb must be a numpy array."
            )

        if not isinstance(
            frame.depth,
            np.ndarray,
        ):
            raise TypeError(
                "CameraFrame.depth must be a numpy array."
            )

        if not isinstance(
            frame.segmentation,
            np.ndarray,
        ):
            raise TypeError(
                "CameraFrame.segmentation must be "
                "a numpy array."
            )

        if frame.rgb.ndim != 3:
            raise ValueError(
                "Camera RGB data must be a 3D array."
            )

        if frame.rgb.shape[2] != 3:
            raise ValueError(
                "Camera RGB data must contain "
                "three channels."
            )

        if frame.depth.ndim != 2:
            raise ValueError(
                "Camera depth data must be a 2D array."
            )

        if frame.segmentation.ndim != 2:
            raise ValueError(
                "Camera segmentation data must "
                "be a 2D array."
            )

        expected_height = frame.rgb.shape[0]
        expected_width = frame.rgb.shape[1]

        if frame.depth.shape != (
            expected_height,
            expected_width,
        ):
            raise ValueError(
                "RGB and depth dimensions do not match."
            )

        if frame.segmentation.shape != (
            expected_height,
            expected_width,
        ):
            raise ValueError(
                "RGB and segmentation dimensions "
                "do not match."
            )

    def _validate_configuration(self) -> None:
        """Validate observation thresholds."""

        if (
            not math.isfinite(
                self._min_valid_depth_meters
            )
            or self._min_valid_depth_meters <= 0.0
        ):
            raise ValueError(
                "min_valid_depth_meters must be "
                "finite and greater than zero."
            )

        if (
            not math.isfinite(
                self._max_valid_depth_meters
            )
            or self._max_valid_depth_meters
            <= self._min_valid_depth_meters
        ):
            raise ValueError(
                "max_valid_depth_meters must be "
                "greater than min_valid_depth_meters."
            )

        if (
            not math.isfinite(
                self._min_valid_depth_ratio
            )
            or not 0.0
            <= self._min_valid_depth_ratio
            <= 1.0
        ):
            raise ValueError(
                "min_valid_depth_ratio must be "
                "between 0.0 and 1.0."
            )

        if (
            not math.isfinite(
                self._min_rgb_mean
            )
            or self._min_rgb_mean < 0.0
        ):
            raise ValueError(
                "min_rgb_mean must be finite "
                "and non-negative."
            )


def main() -> None:
    """Run a standalone visual observation test."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    from robot_twin.simulation.robot_simulator import (
        RobotSimulator,
    )

    simulator = RobotSimulator()

    try:
        simulator.connect()
        simulator.load_world()
        simulator.load_robot()

        camera = RobotCamera(
            simulator=simulator
        )

        obstacle_detector = ObstacleDetector()

        observer = VisionObserver(
            camera=camera,
            obstacle_detector=obstacle_detector,
        )

        observation = observer.observe()

        print()
        print("ROBOTWIN-AI VISION OBSERVATION")
        print("=" * 60)
        print(
            f"Obstacle detected      : "
            f"{observation.obstacle_detected}"
        )
        print(
            f"Obstacle distance      : "
            f"{observation.obstacle_distance:.3f} m"
        )
        print(
            f"Visibility OK          : "
            f"{observation.visibility_ok}"
        )
        print(
            f"Depth available        : "
            f"{observation.depth_available}"
        )
        print(
            f"Segmentation available : "
            f"{observation.segmentation_available}"
        )
        print(
            f"Valid depth ratio      : "
            f"{observation.valid_depth_ratio:.3f}"
        )
        print(
            f"Visible object count   : "
            f"{observation.visible_object_count}"
        )
        print(
            f"Image size             : "
            f"{observation.image_width}x"
            f"{observation.image_height}"
        )
        print(
            f"Reason                 : "
            f"{observation.reason}"
        )

    except Exception:
        LOGGER.exception(
            "RoboTwin-AI vision observation failed."
        )
        raise

    finally:
        simulator.close()


if __name__ == "__main__":
    main()