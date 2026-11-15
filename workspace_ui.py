"""Small presentation helpers for the Streamlit workspace."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


CHART_TYPES = ["Trends", "Distribution", "Scatter", "Grouped scatter", "Rolling window", "Derivatives"]


def apply_workspace_style() -> None:
    st.markdown("""
    <style>
      .stApp { background: #f7f8fc; }
      .block-container { max-width: 1480px; padding-top: 2.4rem; padding-bottom: 3rem; }
      [data-testid="stSidebar"] { background: #fff; border-right: 1px solid #e4e7f0; }
      [data-testid="stSidebar"] .block-container { padding-top: 2rem; }
      h1 { font-size: 2.15rem !important; letter-spacing: -.055em; font-weight: 700 !important; }
      h2 { font-size: 1.35rem !important; letter-spacing: -.025em; }
      h3 { font-size: 1.1rem !important; letter-spacing: -.015em; }
      [data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: 12px; }
      [data-testid="stMetric"] { background: #fff; padding: 1rem 1.2rem; border: 1px solid #e4e7f0; border-radius: 12px; }
      [data-testid="stMetricLabel"] { color: #596579; font-size: .8rem; }
      [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 650; letter-spacing: -.04em; }
      [data-testid="stExpander"] { background: #fff; border-radius: 10px; }
      [data-testid="stPlotlyChart"] { border-radius: 10px; overflow: hidden; }
      [data-testid="stMain"] [data-testid="stRadio"] div[role="radiogroup"] { gap: .4rem; }
      [data-testid="stMain"] [data-testid="stRadio"] label[data-baseweb="radio"] {
        background: #fff; padding: .4rem .8rem; border: 1px solid #dce1ed; border-radius: 8px; margin: 0;
      }
      [data-testid="stMain"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
        background: #eeedff; border-color: #7665d8;
      }
      button:focus-visible, a:focus-visible { outline: 3px solid #7665d8 !important; outline-offset: 3px; }
      .brand { display: flex; align-items: center; gap: .7rem; margin-bottom: 1.8rem; }
      .brand-mark { display: grid; place-items: center; width: 36px; height: 36px; border-radius: 10px; background: #6251c6; color: white; font-weight: 750; font-size: .85rem; }
      .brand-name { color: #20273d; font-weight: 750; font-size: 1.05rem; letter-spacing: -.035em; }
      .eyebrow { color: #6251c6; font-size: .72rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; margin-bottom: .4rem; }
      .welcome-title { font-size: clamp(2rem, 4vw, 3.1rem); font-weight: 750; color: #20273d; line-height: 1.12; letter-spacing: -.06em; max-width: 650px; margin: .8rem 0 1rem; }
      .welcome-copy { color: #596579; max-width: 570px; line-height: 1.7; margin-bottom: 1.5rem; }
      .feature-number { color: #6251c6; font-size: .75rem; font-weight: 750; margin: .1rem 0 .8rem; }
      @media (max-width: 700px) {
        .block-container { padding: 1.6rem 1rem 2rem; }
        h1 { font-size: 1.7rem !important; }
        [data-testid="stMetric"] { padding: .7rem; }
      }
    </style>
    """, unsafe_allow_html=True)


def example_dataset() -> pd.DataFrame:
    """A repeatable, small dataset with numeric signals and grouping categories."""
    rng = np.random.default_rng(42)
    n = 720
    phase = np.arange(n) * 2 * np.pi / 144
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="10min"),
        "temperature": np.round(21 + 3 * np.sin(phase) + rng.normal(0, .35, n), 2),
        "humidity": np.round(55 - 8 * np.sin(phase) + rng.normal(0, 1.2, n), 2),
        "power_kw": np.round(2.5 + .7 * np.sin(phase - .7) + rng.normal(0, .15, n), 3),
        "shift": np.where(np.arange(n) % 144 < 72, "Day", "Night"),
    })


def use_example() -> None:
    st.session_state.data_source = "Example dataset"


def render_empty_state() -> None:
    with st.container(border=True):
        st.markdown('<div class="eyebrow">From dataset to discovery</div><div class="welcome-title">A clearer view of<br>your time series.</div><p class="welcome-copy">Find patterns, compare signals, and explore what changes over time. Start with your own file or take a look around with sample sensor data.</p>', unsafe_allow_html=True)
        st.button("Explore example data", type="primary", on_click=use_example, key="empty_example")
        st.caption("720 readings · 3 signals · no setup needed")
    st.write("")
    for col, number, title, body in zip(st.columns(3), ["01", "02", "03"],
        ["Bring your data", "Focus on what matters", "Explore & export"],
        ["Upload CSV, TSV, Excel, Parquet, JSON, or MATLAB v7.3 in the sidebar.",
         "Choose a time axis and narrow the dataset with clear, flexible filters.",
         "Compare six chart types and download the data behind each view."]):
        with col, st.container(border=True):
            st.markdown(f'<div class="feature-number">{number}</div>', unsafe_allow_html=True)
            st.subheader(title)
            st.caption(body)


def render_chart(fig, plotted_rows: int, source_rows: int, *, height: int = 450) -> None:
    fig.update_layout(
        height=height, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#37445c", size=12),
        colorway=["#6251c6", "#159b96", "#d18a29", "#ca5e7b", "#5185cc", "#7c9651"],
    )
    fig.update_xaxes(gridcolor="#edf0f6", zerolinecolor="#e0e5ef")
    fig.update_yaxes(gridcolor="#edf0f6", zerolinecolor="#e0e5ef")
    st.plotly_chart(fig, use_container_width=True, theme=None, config={
        "displaylogo": False, "scrollZoom": False,
        "toImageButtonOptions": {"format": "png", "filename": "tsplotter_chart", "scale": 2},
    })
    if plotted_rows < source_rows:
        st.caption(f"Showing {plotted_rows:,} of {source_rows:,} rows after chart preparation. Chart CSV exports contain the plotted data.")
