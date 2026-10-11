"""Validate KUKA joint 6 position control."""

from __future__ import annotations

import logging

from robot_twin.simulation.motion_controller import (
    JointMotionController,
)
from robot_twin.simulation.robot_simulator import RobotSimulator


def main() -> None:
    """Test joint 6 movement."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    simulator = RobotSimulator()

    try:
        simulator.connect()
        simulator.load_world()
        simulator.load_robot()

        controller = JointMotionController(
            simulator=simulator,
        )

        initial_position = controller.get_joint_position(6)

        print()
        print("ROBOTWIN-AI JOINT 6 TEST")
        print("=" * 50)
        print(
            f"Initial joint 6 position : "
            f"{initial_position:.4f} rad"
        )

        success = controller.move_joint(
            joint_index=6,
            target_position=0.0,
        )

        final_position = controller.get_joint_position(6)

        print(
            f"Final joint 6 position   : "
            f"{final_position:.4f} rad"
        )
        print(
            f"Position error            : "
            f"{abs(final_position):.4f} rad"
        )
        print(
            f"Motion successful         : "
            f"{success}"
        )

    finally:
        simulator.close()


if __name__ == "__main__":
    main()