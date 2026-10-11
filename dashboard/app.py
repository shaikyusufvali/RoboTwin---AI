
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.parser import parse_runtime_lines, latest_sample, to_frame
from dashboard.persistence import append_event, load_events, csv_bytes
from dashboard.styles import inject_styles


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RUNNER = (
    ROOT
    / "scripts"
    / "run_temporal_multimodal_diagnostic.py"
)

SCENARIOS = {
    "S01": "Normal operation",
    "S02": "Motor resistance",
    "S03": "Low motor power",
    "S04": "Position drift",
    "S05": "Motor resistance + obstacle",
    "S06": "Motor resistance + poor vision",
}


# ============================================================
# 2. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RoboTwin-AI | Fault Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()


# ============================================================
# 3. SESSION STATE
# ============================================================

def initialize_session():
    defaults = {
        "process": None,
        "scenario": "S01",
        "output_lines": [],
        "output_queue": queue.Queue(),
        "reader_thread": None,
        "last_error": None,
        "last_exit_code": None,
        "runtime_status": "IDLE",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def is_running():
    process = st.session_state.process

    return (
        process is not None
        and process.poll() is None
    )


# ============================================================
# 4. BACKGROUND OUTPUT READER
# ============================================================

def read_process_output(pipe, output_queue):
    """
    Read actual stdout from the existing diagnostic runtime.
    The dashboard does not generate fake sensor measurements.
    """

    try:
        for line in iter(pipe.readline, ""):
            output_queue.put(line.rstrip("\r\n"))

    except (OSError, ValueError):
        pass

    finally:
        try:
            pipe.close()
        except Exception:
            pass


def collect_runtime_output():
    output_queue = st.session_state.output_queue

    while True:
        try:
            line = output_queue.get_nowait()

        except queue.Empty:
            break

        st.session_state.output_lines.append(line)

    # Avoid unlimited in-memory output.
    st.session_state.output_lines = (
        st.session_state.output_lines[-4000:]
    )


# ============================================================
# 5. START EXISTING ROBO-TWIN RUNTIME
# ============================================================

def start_diagnosis(scenario):
    """
    Launch the project's existing simulation and diagnosis runner.
    """

    if not RUNNER.exists():
        st.session_state.last_error = (
            f"Runtime file not found: {RUNNER}"
        )

        st.session_state.runtime_status = "ERROR"
        return

    if is_running():
        st.session_state.last_error = (
            "A diagnostic run is already active."
        )
        return

    # Remove output left by a previous run.
    while True:
        try:
            st.session_state.output_queue.get_nowait()

        except queue.Empty:
            break

    st.session_state.output_lines = []
    st.session_state.last_error = None
    st.session_state.last_exit_code = None

    environment = os.environ.copy()

    source_path = str(ROOT / "src")

    existing_python_path = environment.get(
        "PYTHONPATH",
        "",
    )

    environment["PYTHONPATH"] = (
        source_path
        + os.pathsep
        + existing_python_path
    )

    environment["PYTHONUNBUFFERED"] = "1"

    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(RUNNER),
                "--scenario",
                scenario,
            ],
            cwd=str(ROOT),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        st.session_state.process = process
        st.session_state.scenario = scenario
        st.session_state.runtime_status = "RUNNING"

        # Each run gets a new reader thread.
        reader = threading.Thread(
            target=read_process_output,
            args=(
                process.stdout,
                st.session_state.output_queue,
            ),
            daemon=True,
        )

        reader.start()

        st.session_state.reader_thread = reader

        append_event(
            scenario,
            "RUN_STARTED",
            "Existing temporal/multimodal runtime started.",
        )

    except OSError as error:
        st.session_state.last_error = str(error)
        st.session_state.runtime_status = "ERROR"


# ============================================================
# 6. STOP SIMULATION
# ============================================================

def stop_diagnosis():
    process = st.session_state.process

    if process is None:
        return

    if process.poll() is not None:
        return

    process.terminate()

    try:
        process.wait(timeout=5)

    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)

    st.session_state.last_exit_code = (
        process.returncode
    )

    st.session_state.runtime_status = "STOPPED"

    append_event(
        st.session_state.scenario,
        "RUN_STOPPED",
        "Simulation process stopped from dashboard.",
    )


# ============================================================
# 7. HEADER
# ============================================================

def render_header():
    title_column, badge_column = st.columns([4, 1])

    with title_column:
        st.title("🤖 RoboTwin-AI")

        st.subheader(
            "Robot Fault Detection & Diagnostics"
        )

        st.caption(
            "PyBullet simulation connected to the existing "
            "temporal and multimodal diagnostic runtime."
        )

    with badge_column:
        st.markdown(
            """
            <div class="rt-badge">
                SIMULATION MODE
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# 8. SIDEBAR CONTROLS
# ============================================================

def render_sidebar():
    with st.sidebar:
        st.header("Simulation Control")

        selected_scenario = st.selectbox(
            "Select Scenario",
            options=list(SCENARIOS.keys()),
            format_func=lambda scenario: (
                f"{scenario} — {SCENARIOS[scenario]}"
            ),
            index=list(SCENARIOS.keys()).index(
                st.session_state.scenario
            ),
            disabled=is_running(),
        )

        st.divider()

        if st.button(
            "▶ Start Diagnosis",
            use_container_width=True,
            disabled=is_running(),
        ):
            start_diagnosis(selected_scenario)
            st.rerun()

        if st.button(
            "⏹ Stop Diagnosis",
            use_container_width=True,
            disabled=not is_running(),
        ):
            stop_diagnosis()
            st.rerun()

        if st.button(
            "🧹 Clear Output",
            use_container_width=True,
            disabled=is_running(),
        ):
            st.session_state.output_lines = []
            st.rerun()

        st.divider()

        st.markdown("### Runtime Status")

        if is_running():
            st.success("RUNNING")

        elif st.session_state.runtime_status == "ERROR":
            st.error("ERROR")

        else:
            st.info(
                st.session_state.runtime_status
            )

        st.write(
            "**Scenario:**",
            st.session_state.scenario,
        )

        st.caption(
            "The selected scenario is executed by the "
            "existing project runtime."
        )


# ============================================================
# 9. DIAGNOSIS METRICS
# ============================================================

def render_diagnosis_metrics(records):
    sample = latest_sample(records)

    st.markdown("## Diagnostic Overview")

    if not sample:
        columns = st.columns(5)

        labels = [
            "Detected Fault",
            "Confidence",
            "Persistence",
            "Ground Truth",
            "Samples",
        ]

        values = [
            "Waiting",
            "—",
            "—",
            "—",
            "0",
        ]

        for column, label, value in zip(
            columns,
            labels,
            values,
        ):
            column.metric(label, value)

        return

    detected_fault = (
        sample.get("temporal_fault")
        or sample.get("multimodal_fault")
        or sample.get("telemetry_fault")
        or "UNKNOWN"
    )

    confidence = sample.get(
        "temporal_confidence"
    )

    if confidence is None:
        confidence = sample.get(
            "multimodal_confidence"
        )

    if confidence is None:
        confidence = sample.get(
            "telemetry_confidence"
        )

    persistence = sample.get(
        "persistence_score"
    )

    ground_truth = sample.get(
        "ground_truth_fault",
        "NOT EMITTED",
    )

    columns = st.columns(5)

    columns[0].metric(
        "Detected Fault",
        str(detected_fault).upper(),
    )

    columns[1].metric(
        "Confidence",
        (
            f"{confidence:.1%}"
            if isinstance(confidence, (int, float))
            else "—"
        ),
    )

    columns[2].metric(
        "Persistence",
        (
            f"{persistence:.1%}"
            if isinstance(persistence, (int, float))
            else "—"
        ),
    )

    columns[3].metric(
        "Ground Truth",
        str(ground_truth).upper(),
    )

    columns[4].metric(
        "Parsed Samples",
        str(len(records)),
    )

    # Show disagreement instead of hiding it.
    predicted = str(detected_fault).upper()
    truth = str(ground_truth).upper()

    if truth not in (
        "NOT EMITTED",
        "UNKNOWN",
        "",
    ):
        if predicted == truth:
            st.success(
                "Latest diagnosis label matches the "
                "scenario ground truth."
            )

        else:
            st.warning(
                "Diagnosis differs from scenario ground truth. "
                "This is a diagnostic evaluation finding."
            )


# ============================================================
# 10. TELEMETRY CHARTS
# ============================================================

def render_telemetry(records):
    st.markdown("## 📊 Robot Telemetry")

    frame = to_frame(records)

    if frame.empty:
        st.info(
            "Waiting for telemetry from the runtime. "
            "Missing measurements are not replaced with dummy values."
        )
        return

    telemetry_signals = [
        "position",
        "target_position",
        "position_error",
        "velocity",
        "force",
        "motor_current",
        "temperature",
    ]

    available_signals = [
        signal
        for signal in telemetry_signals
        if (
            signal in frame.columns
            and frame[signal].notna().any()
        )
    ]

    if not available_signals:
        st.warning(
            "The runtime has not emitted recognized numeric telemetry fields."
        )

        st.dataframe(
            frame,
            use_container_width=True,
            hide_index=True,
        )

        return

    selected_signals = st.multiselect(
        "Select signals to plot",
        options=available_signals,
        default=available_signals[:3],
    )

    if selected_signals:
        chart_data = frame[
            ["sample_index"] + selected_signals
        ]

        long_data = chart_data.melt(
            id_vars=["sample_index"],
            var_name="Signal",
            value_name="Value",
        )

        figure = px.line(
            long_data,
            x="sample_index",
            y="Value",
            color="Signal",
            markers=True,
            title="Measurements emitted by the runtime",
        )

        figure.update_layout(
            height=380,
            margin=dict(
                l=10,
                r=10,
                t=50,
                b=10,
            ),
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

    st.markdown("### Latest Measurements")

    st.dataframe(
        frame.tail(15),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 11. TEMPORAL AND MULTIMODAL DIAGNOSIS
# ============================================================

def render_diagnosis_history(records):
    st.markdown("## 🩺 Diagnosis History")

    if not records:
        st.info(
            "No diagnostic observations have been parsed yet."
        )
        return

    frame = to_frame(records)

    preferred_columns = [
        "sample_index",
        "ground_truth_fault",
        "telemetry_fault",
        "telemetry_confidence",
        "multimodal_fault",
        "multimodal_confidence",
        "temporal_fault",
        "temporal_confidence",
        "temporal_state",
        "persistence_score",
        "confidence_trend",
        "obstacle_detected",
    ]

    visible_columns = [
        column
        for column in preferred_columns
        if column in frame.columns
    ]

    if visible_columns:
        st.dataframe(
            frame[visible_columns].tail(30),
            use_container_width=True,
            hide_index=True,
        )

    latest = latest_sample(records)

    left, right = st.columns(2)

    with left:
        st.markdown("### Temporal Analysis")

        st.write(
            "**State:**",
            latest.get(
                "temporal_state",
                "Not emitted by runtime",
            ),
        )

        st.write(
            "**Trend:**",
            latest.get(
                "confidence_trend",
                "Not emitted by runtime",
            ),
        )

        st.write(
            "**Persistence:**",
            latest.get(
                "persistence_score",
                "Not emitted by runtime",
            ),
        )

    with right:
        st.markdown("### Vision Context")

        obstacle = latest.get(
            "obstacle_detected"
        )

        st.write(
            "**Obstacle detected:**",
            (
                obstacle
                if obstacle is not None
                else "Not emitted by runtime"
            ),
        )

        st.caption(
            "Vision supplies contextual evidence. "
            "It must not independently declare an internal actuator fault."
        )


# ============================================================
# 12. RAW RUNTIME OUTPUT
# ============================================================

def render_raw_output(records):
    with st.expander(
        "Runtime Logs & Export",
        expanded=True,
    ):
        lines = st.session_state.output_lines

        if lines:
            st.code(
                "\n".join(lines[-250:]),
                language="text",
            )

        else:
            st.info(
                "Choose a scenario and press Start Diagnosis."
            )

        if records:
            st.download_button(
                label="⬇ Export Parsed Observations CSV",
                data=csv_bytes(records),
                file_name=(
                    f"robotwin_"
                    f"{st.session_state.scenario}_"
                    f"samples.csv"
                ),
                mime="text/csv",
            )


# ============================================================
# 13. EVENT HISTORY
# ============================================================

def render_event_history():
    with st.expander(
        "Dashboard Event History",
        expanded=False,
    ):
        events = load_events()

        if events:
            event_frame = pd.DataFrame(
                events[::-1]
            )

            st.dataframe(
                event_frame,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.caption(
                "No dashboard start/stop/finish events recorded yet."
            )


# ============================================================
# 14. MAIN APPLICATION
# ============================================================

def main():
    initialize_session()

    collect_runtime_output()

    render_header()
    render_sidebar()

    if st.session_state.last_error:
        st.error(
            st.session_state.last_error
        )

    process = st.session_state.process

    if (
        process is not None
        and process.poll() is not None
        and st.session_state.last_exit_code is None
    ):
        st.session_state.last_exit_code = (
            process.returncode
        )

        if st.session_state.runtime_status != "STOPPED":
            st.session_state.runtime_status = "FINISHED"

            append_event(
                st.session_state.scenario,
                "RUN_FINISHED",
                f"Process exit code: {process.returncode}",
            )

    records = parse_runtime_lines(
        st.session_state.output_lines
    )

    st.divider()

    render_diagnosis_metrics(records)

    st.divider()

    render_telemetry(records)

    st.divider()

    render_diagnosis_history(records)

    st.divider()

    render_raw_output(records)

    render_event_history()

    st.caption(
        "RoboTwin-AI | Existing PyBullet simulation and diagnostic "
        "runtime | Simulation only; not connected to physical hardware."
    )

    # Refresh the page while a run is active.
    if is_running():
        time.sleep(0.8)
        st.rerun()


if __name__ == "__main__":
    main()

