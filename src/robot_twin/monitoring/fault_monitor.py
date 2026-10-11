"""Live fault injection and temporal fault monitoring."""

from __future__ import annotations

import logging
from collections import Counter, deque
from typing import Any

from robot_twin.diagnostics.diagnostic_brain import (
    DiagnosticBrain,
    DiagnosticResult,
    print_diagnostic_report,
)
from robot_twin.simulation.fault_injector import (
    FaultScenario,
    FaultType,
)
from robot_twin.simulation.motion_controller import (
    JointMotionController,
)
from robot_twin.simulation.robot_simulator import (
    RobotSimulator,
)
from robot_twin.telemetry.pybullet_adapter import (
    PyBulletRobotAdapter,
)


LOGGER = logging.getLogger(__name__)


MONITORED_JOINT_INDEX = 2
TARGET_POSITION = 0.80

FAULT_TYPE = FaultType.MOTOR_RESISTANCE
FAULT_SEVERITY = 0.70

HEALTHY_SETTLE_SECONDS = 2.0
MONITORING_DURATION_SECONDS = 8.0

SAMPLE_INTERVAL_STEPS = 24

CONFIDENCE_THRESHOLD = 0.60
CONFIRMATION_WINDOW_SIZE = 10
MIN_CONFIRMATIONS = 7

CONFIRMABLE_FAULTS = {
    FaultType.MOTOR_RESISTANCE.value,
    FaultType.LOW_MOTOR_POWER.value,
    FaultType.POSITION_DRIFT.value,
}


class FaultMonitoringError(RuntimeError):
    """Raised when live fault monitoring fails."""


def normalize_fault_type(value: Any) -> str:
    """Normalize a fault label to its canonical lowercase value."""

    raw_value = getattr(
        value,
        "value",
        value,
    )

    normalized = (
        str(raw_value)
        .strip()
        .lower()
    )

    if "." in normalized:
        normalized = normalized.rsplit(
            ".",
            1,
        )[-1]

    return normalized


def confirm_fault(
    predictions: deque[str],
) -> str | None:
    """Confirm a persistent non-normal fault.

    A fault is confirmed only when the confirmation window
    contains enough consistent accepted predictions.
    """

    if len(predictions) < CONFIRMATION_WINDOW_SIZE:
        return None

    normalized_predictions = [
        normalize_fault_type(
            prediction
        )
        for prediction in predictions
    ]

    valid_predictions = [
        prediction
        for prediction in normalized_predictions
        if prediction in CONFIRMABLE_FAULTS
    ]

    if len(valid_predictions) < MIN_CONFIRMATIONS:
        return None

    counts = Counter(
        valid_predictions
    )

    fault_type, count = (
        counts.most_common(1)[0]
    )

    if count < MIN_CONFIRMATIONS:
        return None

    return fault_type


def print_live_sample(
    telemetry,
    diagnostic_result: DiagnosticResult,
    target_position: float,
    sample_number: int,
) -> None:
    """Display one live diagnostic sample."""

    position_error = (
        target_position
        - telemetry.position
    )

    print(
        f"[{sample_number:03d}] "
        f"joint={telemetry.joint_index} | "
        f"target={target_position:.3f} | "
        f"position={telemetry.position:.3f} | "
        f"error={position_error:.3f} | "
        f"velocity={telemetry.velocity:.3f} | "
        f"force={telemetry.force:.3f} | "
        f"current={telemetry.motor_current:.3f} | "
        f"temperature={telemetry.temperature:.2f} | "
        f"status={diagnostic_result.status} | "
        f"fault={diagnostic_result.fault_type} | "
        f"confidence={diagnostic_result.confidence:.3f}"
    )


def run_healthy_settling_phase(
    simulator: RobotSimulator,
    adapter: PyBulletRobotAdapter,
    motion_controller: JointMotionController,
    target_position: float,
) -> None:
    """Move the robot to the healthy target before fault injection."""

    LOGGER.info(
        "Starting healthy settling phase."
    )

    adapter.set_commanded_position(
        joint_index=MONITORED_JOINT_INDEX,
        target_position=target_position,
    )

    motion_controller.command_joint(
        joint_index=MONITORED_JOINT_INDEX,
        target_position=target_position,
    )

    physics_steps = int(
        HEALTHY_SETTLE_SECONDS
        / simulator.config.time_step
    )

    if physics_steps <= 0:
        raise FaultMonitoringError(
            "Healthy settling duration produced "
            "no simulation steps."
        )

    for _ in range(physics_steps):
        simulator.step()

    telemetry = (
        adapter.get_joint_telemetry(
            joint_index=MONITORED_JOINT_INDEX
        )
    )

    position_error = (
        target_position
        - telemetry.position
    )

    LOGGER.info(
        "Healthy settling complete | "
        "position=%.4f | "
        "error=%.4f | "
        "velocity=%.4f",
        telemetry.position,
        position_error,
        telemetry.velocity,
    )


def monitor(
    simulator: RobotSimulator,
    adapter: PyBulletRobotAdapter,
) -> DiagnosticResult | None:
    """Run the complete fault-monitoring experiment."""

    motion_controller = JointMotionController(
        simulator=simulator
    )

    diagnostic_brain = DiagnosticBrain()

    scenario = FaultScenario(
        fault_type=FAULT_TYPE,
        joint_index=MONITORED_JOINT_INDEX,
        severity=FAULT_SEVERITY,
    )

    predictions: deque[str] = deque(
        maxlen=CONFIRMATION_WINDOW_SIZE
    )

    diagnostic_history: deque[
        DiagnosticResult
    ] = deque(
        maxlen=CONFIRMATION_WINDOW_SIZE
    )

    latest_result: DiagnosticResult | None = None

    LOGGER.info(
        "Starting live fault-monitoring experiment."
    )

    LOGGER.info(
        "Fault scenario: %s",
        scenario.fault_type.value,
    )

    LOGGER.info(
        "Fault joint: %d",
        scenario.joint_index,
    )

    LOGGER.info(
        "Fault severity: %.2f",
        scenario.severity,
    )

    LOGGER.info(
        "Intended target position: %.3f rad",
        TARGET_POSITION,
    )

    run_healthy_settling_phase(
        simulator=simulator,
        adapter=adapter,
        motion_controller=motion_controller,
        target_position=TARGET_POSITION,
    )

    LOGGER.info(
        "Healthy phase complete."
    )

    LOGGER.info(
        "Activating fault injection: %s",
        scenario.fault_type.value,
    )

    # Apply the fault once.
    #
    # The simulator should then evolve naturally instead of
    # receiving the same motor command on every physics step.
    adapter.apply_fault(
        scenario
    )

    LOGGER.info(
        "Fault injection applied."
    )

    physics_steps = int(
        MONITORING_DURATION_SECONDS
        / simulator.config.time_step
    )

    if physics_steps <= 0:
        raise FaultMonitoringError(
            "Monitoring duration produced "
            "no simulation steps."
        )

    sample_number = 0

    for step_number in range(
        physics_steps
    ):
        simulator.step()

        if (
            step_number
            % SAMPLE_INTERVAL_STEPS
            != 0
        ):
            continue

        telemetry = (
            adapter.get_joint_telemetry(
                joint_index=MONITORED_JOINT_INDEX
            )
        )

        diagnostic_result = (
            diagnostic_brain.diagnose(
                telemetry=telemetry,
                target_position=TARGET_POSITION,
            )
        )

        latest_result = diagnostic_result

        diagnostic_history.append(
            diagnostic_result
        )

        sample_number += 1

        print_live_sample(
            telemetry=telemetry,
            diagnostic_result=diagnostic_result,
            target_position=TARGET_POSITION,
            sample_number=sample_number,
        )

        final_fault = normalize_fault_type(
            diagnostic_result.fault_type
        )

        if (
            final_fault in CONFIRMABLE_FAULTS
            and diagnostic_result.confidence
            >= CONFIDENCE_THRESHOLD
        ):
            predictions.append(
                final_fault
            )

            LOGGER.debug(
                "Accepted fault prediction: %s",
                final_fault,
            )

        else:
            LOGGER.debug(
                "Prediction rejected from "
                "confirmation window: %s",
                final_fault,
            )

        confirmed_fault = confirm_fault(
            predictions
        )

        if confirmed_fault is None:
            continue

        matching_results = [
            result
            for result in diagnostic_history
            if normalize_fault_type(
                result.fault_type
            )
            == confirmed_fault
        ]

        if not matching_results:
            LOGGER.warning(
                "Fault was confirmed as %s, "
                "but no matching diagnostic result "
                "exists in the confirmation history.",
                confirmed_fault,
            )
            continue

        confirmed_result = (
            matching_results[-1]
        )

        confirmation_count = sum(
            prediction == confirmed_fault
            for prediction in predictions
        )

        elapsed_time = (
            step_number
            * simulator.config.time_step
        )

        LOGGER.info(
            "Persistent fault confirmed: %s",
            confirmed_fault,
        )

        LOGGER.info(
            "Confirmation window: %d samples",
            len(predictions),
        )

        LOGGER.info(
            "Fault confirmations: %d/%d",
            confirmation_count,
            len(predictions),
        )

        LOGGER.info(
            "Fault confirmed after %.2f seconds.",
            elapsed_time,
        )

        LOGGER.info(
            "Final diagnostic result aligned "
            "with confirmed fault: %s",
            confirmed_result.fault_type,
        )

        return confirmed_result

    if latest_result is not None:
        LOGGER.warning(
            "No persistent fault confirmation "
            "was reached."
        )

        LOGGER.warning(
            "Latest diagnostic decision: %s",
            latest_result.fault_type,
        )

        return latest_result

    LOGGER.warning(
        "No diagnostic samples were collected."
    )

    return None


def main() -> None:
    """Run the live motor-resistance experiment."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    simulator = RobotSimulator()

    adapter = PyBulletRobotAdapter(
        simulator=simulator
    )

    try:
        adapter.connect()

        result = monitor(
            simulator=simulator,
            adapter=adapter,
        )

        if result is None:
            LOGGER.warning(
                "Fault-monitoring experiment produced "
                "no diagnostic result."
            )
            return

        print_diagnostic_report(
            result
        )

    except KeyboardInterrupt:
        LOGGER.info(
            "Fault monitoring stopped by user."
        )

    except Exception as exc:
        LOGGER.exception(
            "Fault-monitoring experiment failed: %s",
            exc,
        )

        raise FaultMonitoringError(
            "Live fault monitoring failed."
        ) from exc

    finally:
        adapter.disconnect()


if __name__ == "__main__":
    main()