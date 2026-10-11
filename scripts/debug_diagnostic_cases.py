
from __future__ import annotations

from robot_twin.diagnostics.diagnostic_brain import DiagnosticBrain
from robot_twin.diagnostics.fault_injector import (
    FaultInjector,
    FaultScenario,
    FaultType,
)
from robot_twin.simulation.robot_simulator import RobotSimulator
from robot_twin.telemetry.pybullet_adapter import PyBulletRobotAdapter


JOINT_INDEX = 2
SETTLING_STEPS = 240


# ----------------------------------------------------------------------
# Regression test cases
# ----------------------------------------------------------------------
CASES = [
    {
        "name": "S01_NORMAL",
        "fault_type": FaultType.NORMAL,
        "severity": 0.70,
        "expected_fault": "NORMAL",
        "expected_status": "NORMAL",
    },
    {
        "name": "S02_MOTOR_RESISTANCE",
        "fault_type": FaultType.MOTOR_RESISTANCE,
        "severity": 0.70,
        "expected_fault": "MOTOR_RESISTANCE",
        "expected_status": "ANOMALY",
    },
    {
        "name": "S03_LOW_MOTOR_POWER",
        "fault_type": FaultType.LOW_MOTOR_POWER,
        "severity": 0.70,
        "expected_fault": "LOW_MOTOR_POWER",
        "expected_status": "ANOMALY",
    },
    {
        "name": "S04_POSITION_DRIFT",
        "fault_type": FaultType.POSITION_DRIFT,
        "severity": 0.70,
        "expected_fault": "POSITION_DRIFT",
        "expected_status": "ANOMALY",
    },
]


def run_case(case: dict) -> bool:
    """Run one diagnostic regression case."""

    name = case["name"]
    fault_type = case["fault_type"]
    severity = case["severity"]
    expected_fault = case["expected_fault"]
    expected_status = case["expected_status"]

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    simulator = RobotSimulator()

    injector = FaultInjector(
        simulator,
    )

    adapter = PyBulletRobotAdapter(
        simulator=simulator,
        fault_injector=injector,
    )

    try:
        # --------------------------------------------------------------
        # 1. Connect digital twin
        # --------------------------------------------------------------
        adapter.connect()

        # --------------------------------------------------------------
        # 2. Create fault scenario
        # --------------------------------------------------------------
        scenario = FaultScenario(
            fault_type=fault_type,
            joint_index=JOINT_INDEX,
            severity=severity,
        )

        # Always start from a clean state.
        injector.clear(
            joint_index=JOINT_INDEX,
        )

        # Apply requested scenario.
        injector.apply(
            scenario,
        )

        # --------------------------------------------------------------
        # 3. Allow simulation to settle
        # --------------------------------------------------------------
        for _ in range(SETTLING_STEPS):
            simulator.step()

        # --------------------------------------------------------------
        # 4. Collect real robot telemetry
        # --------------------------------------------------------------
        telemetry = adapter.get_joint_telemetry(
            joint_index=JOINT_INDEX,
        )

        target_position = injector.get_target_position(
            joint_index=JOINT_INDEX,
        )

        # --------------------------------------------------------------
        # TEMP DEBUG
        #
        # Verify the exact values reaching DiagnosticBrain.
        # This is intentionally placed before the public diagnose()
        # call so we can isolate simulation/adapter differences.
        # --------------------------------------------------------------
        print("\nDEBUG S04 BOUNDARY")
        print(f"  Telemetry type : {type(telemetry)}")
        print(f"  Target type    : {type(target_position)}")
        print(f"  Position       : {repr(telemetry.position)}")
        print(f"  Force          : {repr(telemetry.force)}")
        print(f"  Current        : {repr(telemetry.motor_current)}")
        print(f"  Temperature    : {repr(telemetry.temperature)}")
        print(f"  Target         : {repr(target_position)}")

        debug_brain = DiagnosticBrain()

        debug_signature = debug_brain._infer_fault_signature(
            telemetry=telemetry,
            target_position=target_position,
        )

        print(f"  Signature      : {debug_signature}")

        # --------------------------------------------------------------
        # 5. Run real Diagnostic Brain
        # --------------------------------------------------------------
        brain = DiagnosticBrain()

        result = brain.diagnose(
            telemetry=telemetry,
            target_position=target_position,
        )

        # --------------------------------------------------------------
        # 6. Ground truth
        # --------------------------------------------------------------
        print("\nGROUND TRUTH")
        print(f"  Fault      : {fault_type.value}")
        print(f"  Severity   : {severity:.2f}")

        # --------------------------------------------------------------
        # 7. Telemetry
        # --------------------------------------------------------------
        print("\nTELEMETRY")
        print(f"  Joint      : {JOINT_INDEX}")
        print(f"  Position   : {telemetry.position:.6f}")
        print(f"  Velocity   : {telemetry.velocity:.6f}")
        print(f"  Force      : {telemetry.force:.6f}")
        print(f"  Current    : {telemetry.motor_current:.6f}")
        print(f"  Temperature: {telemetry.temperature:.6f}")

        # --------------------------------------------------------------
        # 8. Command
        # --------------------------------------------------------------
        print("\nCOMMAND")
        print(f"  Target     : {target_position:.6f}")

        # --------------------------------------------------------------
        # 9. Diagnostic result
        # --------------------------------------------------------------
        print("\nDIAGNOSTIC RESULT")
        print(f"  Status     : {result.status}")
        print(f"  Fault      : {result.fault_type}")
        print(f"  Confidence : {result.confidence:.4f}")
        print(f"  Anomaly    : {result.anomaly_score:.4f}")
        print(f"  ML Fault   : {result.ml_fault_type}")
        print(f"  ML Conf    : {result.ml_confidence:.4f}")

        # --------------------------------------------------------------
        # 10. Evidence
        # --------------------------------------------------------------
        print("\nEVIDENCE")

        if result.evidence:
            for item in result.evidence:
                print(f"  - {item}")
        else:
            print("  - No evidence returned.")

        # --------------------------------------------------------------
        # 11. Probable cause
        # --------------------------------------------------------------
        print("\nCAUSE")
        print(f"  {result.probable_cause}")

        # --------------------------------------------------------------
        # 12. Recommended action
        # --------------------------------------------------------------
        print("\nACTION")
        print(f"  {result.recommended_action}")

        # --------------------------------------------------------------
        # 13. Validation
        # --------------------------------------------------------------
        actual_fault = str(
            result.fault_type
        ).strip().upper()

        actual_status = str(
            result.status
        ).strip().upper()

        expected_fault_normalized = (
            expected_fault.upper()
        )

        expected_status_normalized = (
            expected_status.upper()
        )

        fault_pass = (
            actual_fault
            == expected_fault_normalized
        )

        status_pass = (
            actual_status
            == expected_status_normalized
        )

        print("\nVALIDATION")

        print(
            f"  Expected Fault  : "
            f"{expected_fault_normalized}"
        )

        print(
            f"  Actual Fault    : "
            f"{actual_fault}"
        )

        print(
            f"  Fault Check     : "
            f"{'PASS' if fault_pass else 'FAIL'}"
        )

        print(
            f"  Expected Status : "
            f"{expected_status_normalized}"
        )

        print(
            f"  Actual Status   : "
            f"{actual_status}"
        )

        print(
            f"  Status Check    : "
            f"{'PASS' if status_pass else 'FAIL'}"
        )

        case_passed = (
            fault_pass
            and status_pass
        )

        print()
        print("-" * 80)

        if case_passed:
            print(
                f"RESULT: PASS — {name}"
            )
        else:
            print(
                f"RESULT: FAIL — {name}"
            )

        print("-" * 80)

        return case_passed

    finally:
        adapter.disconnect()


def main() -> None:
    """Run the complete S01-S04 regression suite."""

    print()
    print("=" * 80)
    print("ROBOTWIN-AI DIAGNOSTIC REGRESSION TEST")
    print("=" * 80)

    print("\nConfiguration")
    print(
        f"  Joint index   : {JOINT_INDEX}"
    )
    print(
        f"  Settling steps: {SETTLING_STEPS}"
    )
    print(
        f"  Test cases    : {len(CASES)}"
    )

    results: list[bool] = []

    # ------------------------------------------------------------------
    # Execute all cases
    # ------------------------------------------------------------------
    for case in CASES:
        passed = run_case(case)
        results.append(passed)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    passed_count = sum(results)

    failed_count = (
        len(results)
        - passed_count
    )

    print()
    print("=" * 80)
    print("FINAL REGRESSION SUMMARY")
    print("=" * 80)

    for case, passed in zip(
        CASES,
        results,
    ):
        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"  {case['name']:<25} "
            f"{status}"
        )

    print()
    print(f"  Total : {len(results)}")
    print(f"  Passed: {passed_count}")
    print(f"  Failed: {failed_count}")

    print()

    if failed_count == 0:
        print("OVERALL RESULT: PASS")
        print(
            "S01-S04 diagnostic regression "
            "tests completed successfully."
        )
    else:
        print("OVERALL RESULT: FAIL")
        print(
            "One or more diagnostic cases "
            "failed validation."
        )

    print("=" * 80)


if __name__ == "__main__":
    main()

