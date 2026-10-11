"""Safety-aware motion control for RoboTwin-AI."""

from __future__ import annotations

import logging

from robot_twin.safety.safety_controller import SafetyAction, SafetyDecision
from robot_twin.simulation.motion_controller import JointMotionController


LOGGER = logging.getLogger(__name__)


class SafeMotionController:
    """Execute robot motion only when permitted by safety."""

    def __init__(self, motion_controller: JointMotionController) -> None:
        self._motion_controller = motion_controller

    def execute_joint_motion(
        self,
        joint_index: int,
        target_position: float,
        safety_decision: SafetyDecision,
    ) -> bool:
        """Execute or reject a joint motion command."""

        if safety_decision.action == SafetyAction.STOP:
            self._motion_controller.stop_joint(joint_index)
            LOGGER.warning(
                "Motion blocked by safety controller. Joint=%d",
                joint_index,
            )
            return False

        if safety_decision.action == SafetyAction.SLOW:
            LOGGER.info(
                "Safety warning active. Joint=%d motion permitted with caution.",
                joint_index,
            )

        self._motion_controller.move_joint(
            joint_index=joint_index,
            target_position=target_position,
        )

        return True

    def stop_joint(self, joint_index: int) -> None:
        """Stop a joint through the motion controller."""

        self._motion_controller.stop_joint(joint_index)
