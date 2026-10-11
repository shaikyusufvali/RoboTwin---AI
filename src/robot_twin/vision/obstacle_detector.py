
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from robot_twin.vision.camera import CameraFrame


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObstacleDetectorConfig:
    danger_distance: float = 1.0
    min_detection_distance: float = 0.10
    max_detection_distance: float = 10.0

    region_width_ratio: float = 0.40
    region_height_ratio: float = 0.40

    depth_percentile: float = 10.0
    min_valid_pixel_ratio: float = 0.10
    min_danger_pixel_ratio: float = 0.02


@dataclass(frozen=True)
class ObstacleDetection:
    detected: bool
    minimum_distance: float
    danger_distance: float
    reason: str

    valid_pixel_ratio: float = 0.0
    danger_pixel_ratio: float = 0.0
    analyzed_pixel_count: int = 0


class ObstacleDetector:
    """Detect nearby obstacles using depth-camera data."""

    def __init__(
        self,
        config: ObstacleDetectorConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else ObstacleDetectorConfig()
        )

        self._validate_config()

    def detect(
        self,
        frame: CameraFrame,
    ) -> ObstacleDetection:
        """Analyze one camera frame."""

        self._validate_frame(frame)

        depth_region = self._get_detection_region(
            frame.depth
        )

        if depth_region.size == 0:
            return self._no_data(
                "Detection region is empty."
            )

        valid_mask = self._get_valid_depth_mask(
            depth_region
        )

        total_pixels = depth_region.size
        valid_pixels = int(
            np.count_nonzero(valid_mask)
        )

        valid_ratio = (
            valid_pixels / total_pixels
        )

        if valid_pixels == 0:
            return ObstacleDetection(
                detected=False,
                minimum_distance=float("inf"),
                danger_distance=self.config.danger_distance,
                reason="No valid depth measurements found.",
                valid_pixel_ratio=valid_ratio,
                danger_pixel_ratio=0.0,
                analyzed_pixel_count=0,
            )

        if valid_ratio < self.config.min_valid_pixel_ratio:
            return ObstacleDetection(
                detected=False,
                minimum_distance=float("inf"),
                danger_distance=self.config.danger_distance,
                reason=(
                    "Insufficient valid depth data "
                    "for reliable detection."
                ),
                valid_pixel_ratio=valid_ratio,
                danger_pixel_ratio=0.0,
                analyzed_pixel_count=valid_pixels,
            )

        valid_depth = depth_region[valid_mask]

        minimum_distance = float(
            np.percentile(
                valid_depth,
                self.config.depth_percentile,
            )
        )

        danger_pixels = int(
            np.count_nonzero(
                valid_depth <= self.config.danger_distance
            )
        )

        danger_ratio = (
            danger_pixels / valid_pixels
        )

        detected = (
            danger_ratio
            >= self.config.min_danger_pixel_ratio
        )

        if detected:
            reason = (
                "Obstacle detected inside the "
                "configured danger distance."
            )
        else:
            reason = (
                "No obstacle detected inside the "
                "configured danger distance."
            )

        return ObstacleDetection(
            detected=detected,
            minimum_distance=minimum_distance,
            danger_distance=self.config.danger_distance,
            reason=reason,
            valid_pixel_ratio=valid_ratio,
            danger_pixel_ratio=danger_ratio,
            analyzed_pixel_count=valid_pixels,
        )

    def _get_detection_region(
        self,
        depth: np.ndarray,
    ) -> np.ndarray:
        """Return the central region of the depth image."""

        height, width = depth.shape

        region_width = max(
            1,
            int(
                width
                * self.config.region_width_ratio
            ),
        )

        region_height = max(
            1,
            int(
                height
                * self.config.region_height_ratio
            ),
        )

        x_start = (
            width - region_width
        ) // 2

        y_start = (
            height - region_height
        ) // 2

        x_end = x_start + region_width
        y_end = y_start + region_height

        return depth[
            y_start:y_end,
            x_start:x_end,
        ]

    def _get_valid_depth_mask(
        self,
        depth: np.ndarray,
    ) -> np.ndarray:
        """Return True only for usable depth values."""

        return (
            np.isfinite(depth)
            & (
                depth
                >= self.config.min_detection_distance
            )
            & (
                depth
                <= self.config.max_detection_distance
            )
        )

    def _no_data(
        self,
        reason: str,
    ) -> ObstacleDetection:
        """Return a safe result when depth data is unusable."""

        return ObstacleDetection(
            detected=False,
            minimum_distance=float("inf"),
            danger_distance=self.config.danger_distance,
            reason=reason,
        )

    @staticmethod
    def _validate_frame(
        frame: CameraFrame,
    ) -> None:
        """Validate RGB, depth, and segmentation data."""

        if not isinstance(frame, CameraFrame):
            raise TypeError(
                "frame must be a CameraFrame instance."
            )

        if not isinstance(frame.rgb, np.ndarray):
            raise TypeError(
                "frame.rgb must be a numpy array."
            )

        if not isinstance(frame.depth, np.ndarray):
            raise TypeError(
                "frame.depth must be a numpy array."
            )

        if not isinstance(
            frame.segmentation,
            np.ndarray,
        ):
            raise TypeError(
                "frame.segmentation must be a numpy array."
            )

        if frame.rgb.ndim != 3:
            raise ValueError(
                "RGB image must be a 3D array."
            )

        if frame.rgb.shape[2] != 3:
            raise ValueError(
                "RGB image must have 3 channels."
            )

        if frame.depth.ndim != 2:
            raise ValueError(
                "Depth image must be a 2D array."
            )

        if frame.segmentation.ndim != 2:
            raise ValueError(
                "Segmentation image must be a 2D array."
            )

        rgb_shape = frame.rgb.shape[:2]
        depth_shape = frame.depth.shape
        segmentation_shape = frame.segmentation.shape

        if rgb_shape != depth_shape:
            raise ValueError(
                "RGB and depth dimensions do not match: "
                f"RGB={rgb_shape}, depth={depth_shape}."
            )

        if rgb_shape != segmentation_shape:
            raise ValueError(
                "RGB and segmentation dimensions do not match: "
                f"RGB={rgb_shape}, "
                f"segmentation={segmentation_shape}."
            )

        if frame.depth.size == 0:
            raise ValueError(
                "Depth image cannot be empty."
            )

        if not np.issubdtype(
            frame.depth.dtype,
            np.number,
        ):
            raise TypeError(
                "Depth image must contain numeric values."
            )

    def _validate_config(self) -> None:
        """Validate detector configuration."""

        values = [
            self.config.danger_distance,
            self.config.min_detection_distance,
            self.config.max_detection_distance,
            self.config.region_width_ratio,
            self.config.region_height_ratio,
            self.config.depth_percentile,
            self.config.min_valid_pixel_ratio,
            self.config.min_danger_pixel_ratio,
        ]

        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                "All detector configuration values must be finite."
            )

        if self.config.min_detection_distance <= 0:
            raise ValueError(
                "min_detection_distance must be greater than 0."
            )

        if (
            self.config.max_detection_distance
            <= self.config.min_detection_distance
        ):
            raise ValueError(
                "max_detection_distance must be greater "
                "than min_detection_distance."
            )

        if not (
            self.config.min_detection_distance
            < self.config.danger_distance
            <= self.config.max_detection_distance
        ):
            raise ValueError(
                "danger_distance must be inside the "
                "configured detection range."
            )

        if not (
            0 < self.config.region_width_ratio <= 1
        ):
            raise ValueError(
                "region_width_ratio must be between 0 and 1."
            )

        if not (
            0 < self.config.region_height_ratio <= 1
        ):
            raise ValueError(
                "region_height_ratio must be between 0 and 1."
            )

        if not (
            0 <= self.config.depth_percentile <= 100
        ):
            raise ValueError(
                "depth_percentile must be between 0 and 100."
            )

        if not (
            0 < self.config.min_valid_pixel_ratio <= 1
        ):
            raise ValueError(
                "min_valid_pixel_ratio must be between 0 and 1."
            )

        if not (
            0 < self.config.min_danger_pixel_ratio <= 1
        ):
            raise ValueError(
                "min_danger_pixel_ratio must be between 0 and 1."
            )


def main() -> None:
    """Run the detector against the PyBullet camera."""

    from robot_twin.simulation.robot_simulator import (
        RobotSimulator,
    )
    from robot_twin.vision.camera import RobotCamera

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    simulator = RobotSimulator()

    try:
        simulator.connect()
        simulator.load_world()
        simulator.load_robot()

        camera = RobotCamera(
            simulator=simulator
        )

        detector = ObstacleDetector()

        frame = camera.capture()

        result = detector.detect(frame)

        print()
        print("ROBOTWIN-AI OBSTACLE DETECTOR")
        print("=" * 50)
        print(
            f"Obstacle detected  : {result.detected}"
        )
        print(
            f"Minimum distance   : "
            f"{result.minimum_distance:.3f} m"
        )
        print(
            f"Danger distance    : "
            f"{result.danger_distance:.3f} m"
        )
        print(
            f"Valid depth ratio  : "
            f"{result.valid_pixel_ratio:.3f}"
        )
        print(
            f"Danger pixel ratio : "
            f"{result.danger_pixel_ratio:.3f}"
        )
        print(
            f"Analyzed pixels    : "
            f"{result.analyzed_pixel_count}"
        )
        print(
            f"Reason             : "
            f"{result.reason}"
        )
        print("=" * 50)

    finally:
        simulator.close()


if __name__ == "__main__":
    main()
