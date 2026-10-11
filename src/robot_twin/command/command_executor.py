"""Execute structured robot commands safely."""

from __future__ import annotations

import logging
import math

from robot_twin.command.command_interpreter import CommandType, RobotCommand
from robot_twin.safety.safe_motion_controller import SafeMotionController
from robot_twin.safety.safety_controller import SafetyDecision, SafetyAction


LOGGER = logging.getLogger(__name__)


class CommandExecutor:
    """Execute validated robot commands through the safety layer."""

    def __init__(self, safe_motion: SafeMotionController) -> None:
        self._safe_motion = safe_motion

    def execute(
        self,
        command: RobotCommand,
        safety_decision: SafetyDecision,
    ) -> bool:
        """Execute a structured command when safety permits it."""

        if command.command_type == CommandType.STOP:
            return self._execute_stop(command, safety_decision)

        if command.command_type == CommandType.MOVE_JOINT:
            return self._execute_joint_motion(
                command,
                safety_decision,
            )

        LOGGER.warning(
            "Command type %s is not yet connected to motion planning.",
            command.command_type.value,
        )
        return False

    def _execute_stop(
        self,
        command: RobotCommand,
        safety_decision: SafetyDecision,
    ) -> bool:
        """Execute a stop command."""

        del command
        del safety_decision

        for joint_index in range(7):
            self._safe_motion.stop_joint(joint_index)

        LOGGER.info("Robot stop command executed.")
        return True

    def _execute_joint_motion(
        self,
        command: RobotCommand,
        safety_decision: SafetyDecision,
    ) -> bool:
        """Execute a joint movement command."""

        if command.joint_index is None:
            raise ValueError(
                "Joint command requires a joint index."
            )

        if command.target_position is None:
            raise ValueError(
                "Joint command requires a target position."
            )

        if safety_decision.action == SafetyAction.STOP:
            self._safe_motion.stop_joint(command.joint_index)
            LOGGER.warning(
                "Joint command blocked by safety. Joint=%d",
                command.joint_index,
            )
            return False

        target_radians = math.radians(command.target_position)

        return self._safe_motion.execute_joint_motion(
            joint_index=command.joint_index,
            target_position=target_radians,
            safety_decision=safety_decision,
        )
