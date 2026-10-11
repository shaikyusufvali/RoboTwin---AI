
"""Automated tests for RoboTwin-AI obstacle detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from robot_twin.vision.camera import CameraFrame
from robot_twin.vision.obstacle_detector import (
    ObstacleDetection,
    ObstacleDetector,
    ObstacleDetectorConfig,
)


DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 480


@dataclass(frozen=True)
class DepthScene:
    """Synthetic depth scene used by detector unit tests."""

    depth: np.ndarray


def create_frame(
    depth: np.ndarray,
) -> CameraFrame:
    """Create a synthetic camera frame from a depth image."""

    height, width = depth.shape

    rgb = np.full(
        (height, width, 3),
        128,
        dtype=np.uint8,
    )

    segmentation = np.ones(
        (height, width),
        dtype=np.int32,
    )

    return CameraFrame(
        rgb=rgb,
        depth=depth.astype(
            np.float32,
            copy=False,
        ),
        segmentation=segmentation,
    )


def create_uniform_depth(
    distance: float,
    width: int = DEFAULT_IMAGE_WIDTH,
    height: int = DEFAULT_IMAGE_HEIGHT,
) -> np.ndarray:
    """Create a uniform synthetic depth image."""

    return np.full(
        (height, width),
        distance,
        dtype=np.float32,
    )


def create_central_obstacle_scene(
    background_distance: float = 3.5,
    obstacle_distance: float = 0.6,
    width: int = DEFAULT_IMAGE_WIDTH,
    height: int = DEFAULT_IMAGE_HEIGHT,
    obstacle_width_ratio: float = 0.20,
    obstacle_height_ratio: float = 0.20,
) -> np.ndarray:
    """Create a depth scene with a central obstacle."""

    depth = create_uniform_depth(
        distance=background_distance,
        width=width,
        height=height,
    )

    obstacle_width = int(
        width * obstacle_width_ratio
    )

    obstacle_height = int(
        height * obstacle_height_ratio
    )

    x_start = (
        width - obstacle_width
    ) // 2

    y_start = (
        height - obstacle_height
    ) // 2

    depth[
        y_start:y_start + obstacle_height,
        x_start:x_start + obstacle_width,
    ] = obstacle_distance

    return depth


@pytest.fixture
def detector() -> ObstacleDetector:
    """Create the default obstacle detector."""

    return ObstacleDetector()


@pytest.fixture
def default_config() -> ObstacleDetectorConfig:
    """Return the default detector configuration."""

    return ObstacleDetectorConfig()


def test_no_obstacle_is_detected(
    detector: ObstacleDetector,
) -> None:
    """A clear scene must not report an obstacle."""

    depth = create_uniform_depth(
        distance=3.5
    )

    frame = create_frame(
        depth
    )

    result = detector.detect(
        frame
    )

    assert result.detected is False
    assert result.minimum_distance == pytest.approx(
        3.5,
        abs=0.01,
    )
    assert result.danger_pixel_ratio == pytest.approx(
        0.0
    )
    assert result.valid_pixel_ratio == pytest.approx(
        1.0
    )


def test_near_central_obstacle_is_detected(
    detector: ObstacleDetector,
) -> None:
    """A sufficiently large nearby obstacle must be detected."""

    depth = create_central_obstacle_scene(
        background_distance=3.5,
        obstacle_distance=0.6,
    )

    frame = create_frame(
        depth
    )

    result = detector.detect(
        frame
    )

    assert result.detected is True
    assert result.minimum_distance < 1.0
    assert result.danger_pixel_ratio > 0.0
    assert result.valid_pixel_ratio == pytest.approx(
        1.0
    )


def test_obstacle_outside_danger_distance_is_not_detected(
    detector: ObstacleDetector,
) -> None:
    """An obstacle outside the danger threshold must remain clear."""

    depth = create_central_obstacle_scene(
        background_distance=4.0,
        obstacle_distance=1.5,
    )

    frame = create_frame(
        depth
    )

    result = detector.detect(
        frame
    )

    assert result.detected is False
    assert result.minimum_distance > 1.0
    assert result.danger_pixel_ratio == pytest.approx(
        0.0
    )


def test_invalid_depth_data_is_rejected(
    detector: ObstacleDetector,
) -> None:
    """NaN and infinity values must not become obstacle evidence."""

    depth = create_uniform_depth(
        distance=3.0
    )

    depth[
        100:200,
        100:200,
    ] = np.nan

    depth[
        200:300,
        200:300,
    ] = np.inf

    frame = create_frame(
        depth
    )

    result = detector.detect(
        frame
    )

    assert result.detected is False
    assert result.valid_pixel_ratio < 1.0
    assert result.valid_pixel_ratio > 0.0


def test_entirely_invalid_depth_returns_no_detection(
    detector: ObstacleDetector,
) -> None:
    """An entirely invalid depth image must not report an obstacle."""

    depth = np.full(
        (
            DEFAULT_IMAGE_HEIGHT,
            DEFAULT_IMAGE_WIDTH,
        ),
        np.nan,
        dtype=np.float32,
    )

    frame = create_frame(
        depth
    )

    result = detector.detect(
        frame
    )

    assert result.detected is False
    assert result.minimum_distance == float(
        "inf"
    )
    assert result.analyzed_pixel_count == 0
    assert result.valid_pixel_ratio == pytest.approx(
        0.0
    )


def test_depth_below_sensor_range_is_ignored(
    detector: ObstacleDetector,
) -> None:
    """Depth values below the configured minimum range are invalid."""

    depth = create_uniform_depth(
        distance=3.0
    )

    depth[
        200:280,
        250:390,
    ] = 0.01

    frame = create_frame(
        depth
    )

    result = detector.detect(
        frame
    )

    assert result.detected is False
    assert result.minimum_distance > 1.0


def test_depth_above_sensor_range_is_ignored(
    detector: ObstacleDetector,
) -> None:
    """Depth values beyond the configured maximum are invalid."""

    depth = create_uniform_depth(
        distance=3.0
    )

    depth[
        200:280,
        250:390,
    ] = 50.0

    frame = create_frame(
        depth
    )

    result = detector.detect(
        frame
    )

    assert result.detected is False
    assert result.valid_pixel_ratio < 1.0


def test_small_noise_does_not_create_obstacle(
    detector: ObstacleDetector,
) -> None:
    """A small isolated near-depth region should not trigger detection."""

    depth = create_uniform_depth(
        distance=3.5
    )

    depth[
        235:237,
        319:321,
    ] = 0.5

    frame = create_frame(
        depth
    )

    result = detector.detect(
        frame
    )

    assert result.detected is False
    assert result.danger_pixel_ratio < 0.02


@pytest.mark.parametrize(
    (
        "obstacle_distance",
        "expected_detection",
    ),
    [
        (0.20, True),
        (0.50, True),
        (0.90, True),
        (1.00, True),
        (1.10, False),
        (1.50, False),
        (2.00, False),
    ],
)
def test_detection_boundary(
    obstacle_distance: float,
    expected_detection: bool,
) -> None:
    """Verify behavior around the configured danger threshold."""

    detector = ObstacleDetector()

    depth = create_central_obstacle_scene(
        background_distance=4.0,
        obstacle_distance=obstacle_distance,
        obstacle_width_ratio=0.30,
        obstacle_height_ratio=0.30,
    )

    frame = create_frame(
        depth
    )

    result = detector.detect(
        frame
    )

    assert result.detected is expected_detection


@pytest.mark.parametrize(
    "region_width_ratio",
    [
        0.20,
        0.40,
        0.60,
        0.80,
        1.00,
    ],
)
def test_region_width_configuration(
    region_width_ratio: float,
) -> None:
    """Different horizontal detection regions must remain valid."""

    config = ObstacleDetectorConfig(
        region_width_ratio=region_width_ratio
    )

    detector = ObstacleDetector(
        config=config
    )

    depth = create_uniform_depth(
        distance=3.0
    )

    frame = create_frame(
        depth
    )

    result = detector.detect(
        frame
    )

    assert result.detected is False
    assert result.valid_pixel_ratio == pytest.approx(
        1.0
    )


def test_custom_danger_distance() -> None:
    """The detector must honor a custom safety distance."""

    config = ObstacleDetectorConfig(
        danger_distance=2.0
    )

    detector = ObstacleDetector(
        config=config
    )

    depth = create_central_obstacle_scene(
        background_distance=4.0,
        obstacle_distance=1.5,
    )

    frame = create_frame(
        depth
    )

    result = detector.detect(
        frame
    )

    assert result.detected is True
    assert result.danger_distance == pytest.approx(
        2.0
    )


@pytest.mark.parametrize(
    "config",
    [
        ObstacleDetectorConfig(
            danger_distance=0.0
        ),
        ObstacleDetectorConfig(
            danger_distance=-1.0
        ),
        ObstacleDetectorConfig(
            min_detection_distance=0.0
        ),
        ObstacleDetectorConfig(
            max_detection_distance=0.05
        ),
        ObstacleDetectorConfig(
            region_width_ratio=0.0
        ),
        ObstacleDetectorConfig(
            region_width_ratio=1.1
        ),
        ObstacleDetectorConfig(
            region_height_ratio=0.0
        ),
        ObstacleDetectorConfig(
            depth_percentile=-1.0
        ),
        ObstacleDetectorConfig(
            depth_percentile=101.0
        ),
        ObstacleDetectorConfig(
            min_valid_pixel_ratio=0.0
        ),
        ObstacleDetectorConfig(
            min_valid_pixel_ratio=1.1
        ),
        ObstacleDetectorConfig(
            min_danger_pixel_ratio=0.0
        ),
        ObstacleDetectorConfig(
            min_danger_pixel_ratio=1.1
        ),
    ],
)
def test_invalid_configuration_is_rejected(
    config: ObstacleDetectorConfig,
) -> None:
    """Invalid safety/perception configuration must fail fast."""

    with pytest.raises(
        ValueError
    ):
        ObstacleDetector(
            config=config
        )


def test_invalid_camera_frame_type_is_rejected(
    detector: ObstacleDetector,
) -> None:
    """The detector must reject objects that are not CameraFrame."""

    with pytest.raises(
        TypeError,
        match="CameraFrame",
    ):
        detector.detect(
            object()  # type: ignore[arg-type]
        )


def test_depth_shape_must_be_two_dimensional() -> None:
    """Depth input must be a two-dimensional array."""

    detector = ObstacleDetector()

    depth = np.zeros(
        (
            DEFAULT_IMAGE_HEIGHT,
            DEFAULT_IMAGE_WIDTH,
            1,
        ),
        dtype=np.float32,
    )

    rgb = np.zeros(
        (
            DEFAULT_IMAGE_HEIGHT,
            DEFAULT_IMAGE_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )

    segmentation = np.zeros(
        (
            DEFAULT_IMAGE_HEIGHT,
            DEFAULT_IMAGE_WIDTH,
        ),
        dtype=np.int32,
    )

    frame = CameraFrame(
        rgb=rgb,
        depth=depth,  # type: ignore[arg-type]
        segmentation=segmentation,
    )

    with pytest.raises(
        ValueError,
        match="2D",
    ):
        detector.detect(
            frame
        )


def test_rgb_and_depth_dimensions_must_match() -> None:
    """The detector must reject inconsistent camera dimensions."""

    detector = ObstacleDetector()

    rgb = np.zeros(
        (
            DEFAULT_IMAGE_HEIGHT,
            DEFAULT_IMAGE_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )

    depth = np.zeros(
        (
            DEFAULT_IMAGE_HEIGHT - 10,
            DEFAULT_IMAGE_WIDTH,
        ),
        dtype=np.float32,
    )

    segmentation = np.zeros(
        (
            DEFAULT_IMAGE_HEIGHT,
            DEFAULT_IMAGE_WIDTH,
        ),
        dtype=np.int32,
    )

    frame = CameraFrame(
        rgb=rgb,
        depth=depth,
        segmentation=segmentation,
    )

    with pytest.raises(
        ValueError,
        match="dimensions",
    ):
        detector.detect(
            frame
        )


def test_result_contains_observation_metadata(
    detector: ObstacleDetector,
) -> None:
    """Detection results must expose measurable evidence."""

    depth = create_central_obstacle_scene(
        background_distance=4.0,
        obstacle_distance=0.5,
        obstacle_width_ratio=0.30,
        obstacle_height_ratio=0.30,
    )

    frame = create_frame(
        depth
    )

    result = detector.detect(
        frame
    )

    assert isinstance(
        result,
        ObstacleDetection,
    )

    assert result.analyzed_pixel_count > 0
    assert 0.0 <= result.valid_pixel_ratio <= 1.0
    assert 0.0 <= result.danger_pixel_ratio <= 1.0
    assert result.danger_distance == pytest.approx(
        1.0
    )
    assert result.reason


def test_detector_is_deterministic(
    detector: ObstacleDetector,
) -> None:
    """Identical input frames must produce identical results."""

    depth = create_central_obstacle_scene(
        background_distance=4.0,
        obstacle_distance=0.6,
    )

    frame = create_frame(
        depth
    )

    first = detector.detect(
        frame
    )

    second = detector.detect(
        frame
    )

    assert first == second