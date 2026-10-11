
from __future__ import annotations

import streamlit as st


def inject_styles() -> None:
    """Apply the RoboTwin-AI dashboard theme."""

    st.markdown(
        """
        <style>
        /* --------------------------------------------------
           GLOBAL THEME
        -------------------------------------------------- */

        .stApp {
            background:
                radial-gradient(
                    circle at 10% 0%,
                    rgba(0, 210, 255, 0.08),
                    transparent 35%
                ),
                #0b1020;
            color: #e6edf7;
        }

        [data-testid="stHeader"] {
            background: rgba(11, 16, 32, 0.85);
        }

        [data-testid="stSidebar"] {
            background: #10182b;
            border-right: 1px solid #25334d;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #e6edf7;
        }

        /* --------------------------------------------------
           HEADINGS
        -------------------------------------------------- */

        h1 {
            color: #f1f5ff !important;
            font-weight: 800 !important;
            letter-spacing: -1px;
        }

        h2 {
            color: #dbeafe !important;
            font-weight: 700 !important;
        }

        h3 {
            color: #c8d8f0 !important;
        }

        p,
        label,
        span {
            color: inherit;
        }

        /* --------------------------------------------------
           METRIC CARDS
        -------------------------------------------------- */

        [data-testid="stMetric"] {
            background: linear-gradient(
                145deg,
                #17233b,
                #111a2d
            );
            border: 1px solid #2a3b59;
            border-radius: 14px;
            padding: 18px 18px 14px 18px;
            min-height: 110px;
            box-shadow: 0 5px 18px rgba(0, 0, 0, 0.15);
        }

        [data-testid="stMetricLabel"] {
            color: #91a7c6 !important;
            font-size: 0.85rem !important;
        }

        [data-testid="stMetricValue"] {
            color: #36d9f7 !important;
            font-size: 1.55rem !important;
            font-weight: 800 !important;
        }

        [data-testid="stMetricDelta"] {
            color: #a7f3d0 !important;
        }

        /* --------------------------------------------------
           BUTTONS
        -------------------------------------------------- */

        .stButton > button {
            width: 100%;
            border-radius: 9px;
            border: 1px solid #31516f;
            background: #172944;
            color: #e8f5ff;
            font-weight: 650;
            min-height: 42px;
            transition: all 0.15s ease-in-out;
        }

        .stButton > button:hover {
            background: #1c4261;
            border-color: #36d9f7;
            color: #ffffff;
        }

        .stButton > button:disabled {
            opacity: 0.45;
        }

        /* --------------------------------------------------
           SELECT BOXES AND INPUTS
        -------------------------------------------------- */

        [data-baseweb="select"] > div {
            background: #141f34;
            border-color: #30435f;
            border-radius: 8px;
        }

        [data-baseweb="input"] > div {
            background: #141f34;
            border-color: #30435f;
        }

        /* --------------------------------------------------
           DATAFRAMES
        -------------------------------------------------- */

        [data-testid="stDataFrame"] {
            border: 1px solid #293a56;
            border-radius: 10px;
            overflow: hidden;
        }

        /* --------------------------------------------------
           ALERTS
        -------------------------------------------------- */

        [data-testid="stAlert"] {
            border-radius: 10px;
        }

        /* --------------------------------------------------
           EXPANDERS
        -------------------------------------------------- */

        [data-testid="stExpander"] {
            background: rgba(20, 31, 52, 0.75);
            border: 1px solid #293a56;
            border-radius: 10px;
        }

        /* --------------------------------------------------
           DIVIDERS AND CAPTIONS
        -------------------------------------------------- */

        hr {
            border-color: #273750 !important;
        }

        [data-testid="stCaptionContainer"] {
            color: #91a7c6;
        }

        /* --------------------------------------------------
           ROBO-TWIN HEADER BADGE
        -------------------------------------------------- */

        .rt-badge {
            display: inline-block;
            margin-top: 25px;
            padding: 10px 15px;
            border: 1px solid #176b78;
            border-radius: 8px;
            background: rgba(12, 74, 89, 0.25);
            color: #67e8f9;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 1px;
            text-align: center;
        }

        /* --------------------------------------------------
           PLOTLY CHART CONTAINER
        -------------------------------------------------- */

        [data-testid="stPlotlyChart"] {
            background: rgba(17, 26, 45, 0.7);
            border: 1px solid #293a56;
            border-radius: 12px;
            padding: 8px;
        }

        /* --------------------------------------------------
           RESPONSIVE SPACING
        -------------------------------------------------- */

        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2.5rem;
            max-width: 1600px;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


