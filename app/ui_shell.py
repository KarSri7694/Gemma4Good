from __future__ import annotations

from html import escape

import streamlit as st


GEMMA_ANALYSIS_LABEL = "Gemma Analysis"

_GLOBAL_CSS = """
<style>
:root {
    --pp-bg: #f6f8fc;
    --pp-surface: #ffffff;
    --pp-surface-muted: #eef2ff;
    --pp-border: #dbe4ff;
    --pp-text: #1f2a44;
    --pp-text-muted: #5f6b85;
    --pp-primary: #3454d1;
    --pp-primary-strong: #223a9b;
    --pp-sidebar-top: #162447;
    --pp-sidebar-bottom: #21346b;
    --pp-success-bg: #dcfce7;
    --pp-success-text: #166534;
    --pp-shadow: 0 10px 30px rgba(26, 43, 92, 0.08);
}

html, body {
    font-family: "Segoe UI", "Inter", Arial, sans-serif;
    color: var(--pp-text);
}

.stApp {
    background:
        radial-gradient(circle at top right, rgba(52, 84, 209, 0.08), transparent 24%),
        linear-gradient(180deg, #fbfcff 0%, var(--pp-bg) 100%);
}

.main .block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px;
}

h1, h2, h3 {
    color: var(--pp-text);
    letter-spacing: -0.02em;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--pp-sidebar-top) 0%, var(--pp-sidebar-bottom) 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] .stCaption {
    color: #f8fbff !important;
}

section[data-testid="stSidebar"] [data-baseweb="input"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] textarea {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    color: #ffffff !important;
}

section[data-testid="stSidebar"] [data-baseweb="input"] input::placeholder,
section[data-testid="stSidebar"] textarea::placeholder {
    color: rgba(255, 255, 255, 0.62) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    padding: 0.35rem;
    border: 1px solid var(--pp-border);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.82);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    color: var(--pp-text-muted);
    min-height: 2.6rem;
    padding: 0.35rem 1rem;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--pp-primary) 0%, var(--pp-primary-strong) 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 6px 16px rgba(52, 84, 209, 0.24);
}

.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button {
    border-radius: 12px !important;
}

.stButton > button[kind="primary"],
.stFormSubmitButton > button {
    border: 1px solid transparent !important;
    background: linear-gradient(135deg, var(--pp-primary) 0%, var(--pp-primary-strong) 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 8px 20px rgba(52, 84, 209, 0.22) !important;
}

.stButton > button:not([kind="primary"]),
.stDownloadButton > button {
    border: 1px solid var(--pp-border) !important;
    background: rgba(255, 255, 255, 0.88) !important;
    color: var(--pp-text) !important;
}

[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid var(--pp-border);
    border-radius: 16px;
    padding: 0.9rem 1rem;
    box-shadow: var(--pp-shadow);
}

[data-testid="stMetricLabel"] {
    color: var(--pp-text-muted) !important;
    white-space: normal !important;
}

[data-testid="stMetricValue"] {
    color: var(--pp-text) !important;
}

.stTextInput input,
.stTextArea textarea,
div[data-baseweb="select"] > div,
.stDateInput input,
.stTimeInput input,
[data-testid="stFileUploader"] section {
    border-radius: 12px !important;
    border-color: var(--pp-border) !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: var(--pp-primary) !important;
    box-shadow: 0 0 0 0.2rem rgba(52, 84, 209, 0.12) !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px !important;
}

[data-testid="stNotification"] {
    border-radius: 14px !important;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--pp-primary), var(--pp-primary-strong)) !important;
}

.main .stCaption,
.main .stCaption p {
    color: var(--pp-text-muted) !important;
}
</style>
"""


def apply_global_ui_theme() -> None:
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def render_workspace_banner(
    *,
    class_label: str,
    subject_name: str,
    academic_year: str,
    medium: str,
    present_count: int,
    plan_completion: float | int,
    has_active_slot: bool,
) -> None:
    safe_class_label = escape(class_label or "Class")
    safe_subject_name = escape(subject_name or "No subject")
    safe_academic_year = escape(academic_year or "No year")
    safe_medium = escape(medium or "No medium")
    safe_present_count = escape(str(present_count))
    safe_plan_completion = escape(str(plan_completion))
    active_dot = (
        '<span style="display:inline-flex;align-items:center;gap:5px;background:#DCFCE7;'
        'color:#166534;padding:4px 12px;border-radius:999px;font-size:0.78rem;font-weight:600;">'
        '<span style="width:7px;height:7px;background:#22C55E;border-radius:50%;display:inline-block;"></span>'
        "Active period</span>"
        if has_active_slot
        else '<span style="display:inline-flex;align-items:center;gap:5px;background:#F3F4F6;'
        'color:#6B7280;padding:4px 12px;border-radius:999px;font-size:0.78rem;font-weight:500;">'
        "No active period</span>"
    )

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #3454D1 0%, #223A9B 100%);
            border-radius: 18px;
            padding: 1.4rem 1.75rem;
            margin-bottom: 1rem;
            box-shadow: 0 12px 28px rgba(26, 43, 92, 0.20);
        ">
            <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:0.75rem;">
                <div>
                    <div style="font-size:1.45rem; font-weight:700; color:white; letter-spacing:-0.02em; margin-bottom:0.2rem;">
                        {safe_class_label}
                    </div>
                    <div style="color:rgba(255,255,255,0.78); font-size:0.85rem;">
                        {safe_subject_name} &nbsp;&middot;&nbsp; {safe_academic_year}
                        &nbsp;&middot;&nbsp; {safe_medium}
                    </div>
                </div>
                <div>{active_dot}</div>
            </div>
            <div style="display:flex; gap:1.25rem; margin-top:1rem; flex-wrap:wrap;">
                <div style="background:rgba(255,255,255,0.14); border-radius:12px; padding:0.65rem 1rem; flex:1; min-width:110px;">
                    <div style="color:rgba(255,255,255,0.64); font-size:0.70rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:2px;">
                        Present Today
                    </div>
                    <div style="color:white; font-size:1.4rem; font-weight:700; line-height:1;">
                        {safe_present_count}
                    </div>
                </div>
                <div style="background:rgba(255,255,255,0.14); border-radius:12px; padding:0.65rem 1rem; flex:1; min-width:110px;">
                    <div style="color:rgba(255,255,255,0.64); font-size:0.70rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:2px;">
                        Plan Completion
                    </div>
                    <div style="color:white; font-size:1.4rem; font-weight:700; line-height:1;">
                        {safe_plan_completion}%
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(title: str, caption: str = "") -> None:
    safe_title = escape(title)
    safe_caption = escape(caption)
    caption_html = (
        f"<div style='color:#5F6B85;font-size:0.82rem;margin-top:0.2rem;padding-left:0.85rem;'>{safe_caption}</div>"
        if caption
        else ""
    )
    st.markdown(
        f"""
        <div style="margin-bottom:0.6rem;">
            <div style="font-size:1.12rem; font-weight:700; color:#1F2A44; border-left:4px solid #3454D1; padding-left:0.6rem; line-height:1.3;">
                {safe_title}
            </div>
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel(title: str, caption: str = ""):
    container = st.container(border=True)
    container.markdown(f"**{escape(title)}**")
    if caption:
        container.caption(caption)
    return container
