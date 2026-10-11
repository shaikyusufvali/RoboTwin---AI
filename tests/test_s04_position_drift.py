
"""
Focused regression tests for the S04 POSITION_DRIFT scenario.

These tests protect the S04 diagnostic boundary without changing
the already validated S01/S02/S03 behavior.
"""

from robot_twin.diagnostics.diagnostic_brain import DiagnosticBrain
from robot_twin.telemetry.robot_adapter import JointTelemetry


def build_s04_telemetry() -> JointTelemetry:
    """Return the observed S04 telemetry signature."""
    return JointTelemetry(
        joint_index=2,
        position=0.750000,
        velocity=0.750000,
        force=0.903790,
        motor_current=0.532303,
        temperature=29.258425,
    )


def build_s01_telemetry() -> JointTelemetry:
    """Return the observed S01 NORMAL boundary telemetry."""
    return JointTelemetry(
        joint_index=2,
        position=0.750000,
        velocity=0.750000,
        force=0.903790,
        motor_current=0.462303,
        temperature=29.258425,
    )


def build_s03_telemetry() -> JointTelemetry:
    """Return the observed S03 LOW_MOTOR_POWER boundary telemetry."""
    return JointTelemetry(
        joint_index=2,
        position=0.750000,
        velocity=0.750000,
        force=0.903790,
        motor_current=1.162303,
        temperature=34.298425,
    )


def test_s04_telemetry_signature_is_position_drift():
    """S04 deterministic telemetry signature must be POSITION_DRIFT."""
    brain = DiagnosticBrain()
    telemetry = build_s04_telemetry()

    signature = brain._infer_fault_signature(
        telemetry=telemetry,
        target_position=0.80,
    )

    assert signature == "POSITION_DRIFT"


def test_s04_final_diagnosis_is_position_drift():
    """S04 public diagnosis must identify POSITION_DRIFT."""
    brain = DiagnosticBrain()
    telemetry = build_s04_telemetry()

    result = brain.diagnose(
        telemetry=telemetry,
        target_position=0.80,
    )

    assert result.fault_type == "POSITION_DRIFT"


def test_s04_final_status_is_anomaly():
    """A detected S04 fault must produce ANOMALY status."""
    brain = DiagnosticBrain()
    telemetry = build_s04_telemetry()

    result = brain.diagnose(
        telemetry=telemetry,
        target_position=0.80,
    )

    assert result.status == "ANOMALY"


def test_s04_must_not_regress_to_normal():
    """S04 must never fall back to NORMAL."""
    brain = DiagnosticBrain()
    telemetry = build_s04_telemetry()

    result = brain.diagnose(
        telemetry=telemetry,
        target_position=0.80,
    )

    assert result.fault_type != "NORMAL"


def test_s04_must_not_be_classified_as_low_motor_power():
    """S04 must remain distinct from LOW_MOTOR_POWER."""
    brain = DiagnosticBrain()
    telemetry = build_s04_telemetry()

    result = brain.diagnose(
        telemetry=telemetry,
        target_position=0.80,
    )

    assert result.fault_type != "LOW_MOTOR_POWER"


def test_s01_boundary_remains_normal():
    """S01 boundary must remain UNKNOWN at signature level."""
    brain = DiagnosticBrain()
    telemetry = build_s01_telemetry()

    signature = brain._infer_fault_signature(
        telemetry=telemetry,
        target_position=0.80,
    )

    assert signature == "UNKNOWN"


def test_s03_boundary_remains_low_motor_power():
    """S03 boundary must remain LOW_MOTOR_POWER."""
    brain = DiagnosticBrain()
    telemetry = build_s03_telemetry()

    signature = brain._infer_fault_signature(
        telemetry=telemetry,
        target_position=0.80,
    )

    assert signature == "LOW_MOTOR_POWER"
