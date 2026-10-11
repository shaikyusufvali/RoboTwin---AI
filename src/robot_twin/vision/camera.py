"""Camera interface for the RoboTwin-AI simulation."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import pybullet as p

from robot_twin.simulation.robot_simulator import RobotSimulator


LOGGER = logging.getLogger(__name__)


DEFAULT_CAMERA_POSITION: Final[
    tuple[float, float, float]
] = (
    2.5,
    -2.5,
    2.0,
)

DEFAULT_CAMERA_TARGET: Final[
    tuple[float, float, float]
] = (
    0.0,
    0.0,
    0.5,
)

DEFAULT_CAMERA_UP: Final[
    tuple[float, float, float]
] = (
    0.0,
    0.0,
    1.0,
)

DEFAULT_CAMERA_FOV: Final[float] = 60.0
DEFAULT_CAMERA_WIDTH: Final[int] = 640
DEFAULT_CAMERA_HEIGHT: Final[int] = 480
DEFAULT_CAMERA_NEAR: Final[float] = 0.1
DEFAULT_CAMERA_FAR: Final[float] = 10.0


@dataclass(frozen=True)
class CameraConfig:
    """Configuration for the simulated camera."""

    position: tuple[float, float, float] = (
        DEFAULT_CAMERA_POSITION
    )

    target: tuple[float, float, float] = (
        DEFAULT_CAMERA_TARGET
    )

    up_vector: tuple[float, float, float] = (
        DEFAULT_CAMERA_UP
    )

    fov: float = DEFAULT_CAMERA_FOV
    width: int = DEFAULT_CAMERA_WIDTH
    height: int = DEFAULT_CAMERA_HEIGHT
    near_plane: float = DEFAULT_CAMERA_NEAR
    far_plane: float = DEFAULT_CAMERA_FAR


@dataclass(frozen=True)
class CameraFrame:
    """Image data captured by the simulated camera."""

    rgb: np.ndarray
    depth: np.ndarray
    segmentation: np.ndarray


class CameraError(RuntimeError):
    """Raised when simulated camera capture fails."""


class RobotCamera:
    """Capture RGB, depth, and segmentation data from PyBullet."""

    def __init__(
        self,
        simulator: RobotSimulator,
        config: CameraConfig | None = None,
    ) -> None:
        self._simulator = simulator
        self._config = config or CameraConfig()

        self._validate_config()

    def capture(self) -> CameraFrame:
        """Capture one RGB, depth, and segmentation frame."""

        client_id = self._get_client_id()

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self._config.position,
            cameraTargetPosition=self._config.target,
            cameraUpVector=self._config.up_vector,
        )

        projection_matrix = (
            p.computeProjectionMatrixFOV(
                fov=self._config.fov,
                aspect=(
                    self._config.width
                    / self._config.height
                ),
                nearVal=self._config.near_plane,
                farVal=self._config.far_plane,
            )
        )

        try:
            image_data = p.getCameraImage(
                width=self._config.width,
                height=self._config.height,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                flags=(
                    p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
                ),
                physicsClientId=client_id,
            )
        except Exception as exc:
            LOGGER.exception(
                "PyBullet camera capture failed."
            )

            raise CameraError(
                "Failed to capture a frame from "
                "the PyBullet camera."
            ) from exc

        returned_width = int(image_data[0])
        returned_height = int(image_data[1])

        if (
            returned_width != self._config.width
            or returned_height != self._config.height
        ):
            raise CameraError(
                "Camera returned an unexpected image size: "
                f"{returned_width}x{returned_height}; "
                f"expected "
                f"{self._config.width}x"
                f"{self._config.height}."
            )

        try:
            rgb = self._convert_rgb(
                image_data[2],
                returned_width,
                returned_height,
            )

            depth_buffer = self._convert_depth_buffer(
                image_data[3],
                returned_width,
                returned_height,
            )

            segmentation = (
                self._convert_segmentation(
                    image_data[4],
                    returned_width,
                    returned_height,
                )
            )

        except (TypeError, ValueError) as exc:
            LOGGER.exception(
                "Invalid camera image data returned "
                "by PyBullet."
            )

            raise CameraError(
                "PyBullet returned invalid camera data."
            ) from exc

        depth = self._convert_depth(
            depth_buffer
        )

        return CameraFrame(
            rgb=rgb,
            depth=depth,
            segmentation=segmentation,
        )

    def _get_client_id(self) -> int:
        """Return the active PyBullet client ID."""

        try:
            return self._simulator.client_id
        except RuntimeError as exc:
            raise CameraError(
                "Simulation must be connected before "
                "capturing camera data."
            ) from exc

    @staticmethod
    def _convert_rgb(
        rgb_data: object,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Convert PyBullet RGBA data to an RGB image."""

        rgb = np.asarray(
            rgb_data,
            dtype=np.uint8,
        )

        expected_size = (
            width
            * height
            * 4
        )

        if rgb.size != expected_size:
            raise ValueError(
                "Unexpected RGB buffer size: "
                f"{rgb.size}; expected "
                f"{expected_size}."
            )

        return (
            rgb.reshape(
                height,
                width,
                4,
            )[:, :, :3]
            .copy()
        )

    @staticmethod
    def _convert_depth_buffer(
        depth_data: object,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Convert raw depth data into a 2D depth buffer."""

        depth_buffer = np.asarray(
            depth_data,
            dtype=np.float32,
        )

        expected_size = (
            width
            * height
        )

        if depth_buffer.size != expected_size:
            raise ValueError(
                "Unexpected depth buffer size: "
                f"{depth_buffer.size}; expected "
                f"{expected_size}."
            )

        return depth_buffer.reshape(
            height,
            width,
        )

    @staticmethod
    def _convert_segmentation(
        segmentation_data: object,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Convert segmentation data into a 2D integer mask."""

        segmentation = np.asarray(
            segmentation_data,
            dtype=np.int32,
        )

        expected_size = (
            width
            * height
        )

        if segmentation.size != expected_size:
            raise ValueError(
                "Unexpected segmentation buffer size: "
                f"{segmentation.size}; expected "
                f"{expected_size}."
            )

        return segmentation.reshape(
            height,
            width,
        ).copy()

    def _convert_depth(
        self,
        depth_buffer: np.ndarray,
    ) -> np.ndarray:
        """Convert normalized depth buffer values to meters."""

        near = self._config.near_plane
        far = self._config.far_plane

        denominator = (
            far
            - (far - near)
            * depth_buffer
        )

        depth = (
            far
            * near
            / denominator
        )

        return depth.astype(
            np.float32,
            copy=False,
        )

    def _validate_config(self) -> None:
        """Validate camera configuration."""

        if len(self._config.position) != 3:
            raise ValueError(
                "Camera position must contain three values."
            )

        if len(self._config.target) != 3:
            raise ValueError(
                "Camera target must contain three values."
            )

        if len(self._config.up_vector) != 3:
            raise ValueError(
                "Camera up_vector must contain three values."
            )

        for name, vector in (
            ("position", self._config.position),
            ("target", self._config.target),
            ("up_vector", self._config.up_vector),
        ):
            if not all(
                math.isfinite(value)
                for value in vector
            ):
                raise ValueError(
                    f"Camera {name} contains "
                    "non-finite values."
                )

        if not (
            0.0
            < self._config.fov
            < 180.0
        ):
            raise ValueError(
                "Camera fov must be between "
                "0 and 180 degrees."
            )

        if self._config.width <= 0:
            raise ValueError(
                "Camera width must be greater than zero."
            )

        if self._config.height <= 0:
            raise ValueError(
                "Camera height must be greater than zero."
            )

        if (
            self._config.near_plane <= 0.0
        ):
            raise ValueError(
                "Camera near_plane must be "
                "greater than zero."
            )

        if (
            self._config.far_plane
            <= self._config.near_plane
        ):
            raise ValueError(
                "Camera far_plane must be greater "
                "than near_plane."
            )


def main() -> None:
    """Capture and inspect a simulated camera frame."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    simulator = RobotSimulator()

    try:
        simulator.connect()
        simulator.load_world()
        simulator.load_robot()

        camera = RobotCamera(
            simulator=simulator
        )

        frame = camera.capture()

        print()
        print("ROBOTWIN-AI CAMERA TEST")
        print("=" * 60)
        print(
            f"RGB shape          : "
            f"{frame.rgb.shape}"
        )
        print(
            f"Depth shape        : "
            f"{frame.depth.shape}"
        )
        print(
            f"Segmentation shape : "
            f"{frame.segmentation.shape}"
        )
        print(
            f"RGB dtype          : "
            f"{frame.rgb.dtype}"
        )
        print(
            f"Depth dtype        : "
            f"{frame.depth.dtype}"
        )
        print(
            f"Segmentation dtype : "
            f"{frame.segmentation.dtype}"
        )
        print(
            f"Depth range        : "
            f"{frame.depth.min():.2f}m - "
            f"{frame.depth.max():.2f}m"
        )
        print(
            f"Segmentation IDs   : "
            f"{np.unique(frame.segmentation).size}"
        )

    except CameraError:
        LOGGER.exception(
            "RoboTwin-AI camera test failed."
        )
        raise

    finally:
        simulator.close()


if __name__ == "__main__":
    main()