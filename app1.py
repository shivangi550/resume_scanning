import os
import tempfile
from collections import Counter
from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.extract_text import extract_text_from_pdf
from src.preprocess import preprocess_text
from src.information_extraction import extract_information
from src.matching import calculate_similarity
from src.skill_matching import calculate_skill_match
from src.job_description import extract_job_requirements
from src.explanation import generate_candidate_explanation

from src.scoring import (
    calculate_final_score,
    calculate_experience_match,
    calculate_education_match,
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ResumeIQ",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# DESIGN TOKENS
# =========================================================
# Palette:
#   --ink        near-navy, headings & primary text
#   --muted      slate gray, secondary text
#   --bg         cool light-slate app background
#   --surface    card surface (white)
#   --primary    indigo — brand / interactive accent
#   --strong     muted emerald — strong match
#   --moderate   deep amber — moderate match
#   --weak       deep rose — weak match
# Type:
#   Space Grotesk — display / headings
#   Inter         — body / UI text
#   JetBrains Mono— scores & numeric readouts (instrument feel)

TOKENS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700;800&display=swap');

:root {
    --ink: #0B1220;
    --muted: #64748B;
    --muted-2: #94A3B8;
    --bg: #EEF1F7;
    --surface: #FFFFFF;
    --border: #E2E6EF;
    --border-soft: #ECEFF5;

    --primary: #3538CD;
    --primary-dark: #262993;
    --primary-tint: #EEF0FD;

    --strong: #0F9D77;
    --strong-bg: #E4F7F0;
    --strong-ink: #0B6E54;

    --moderate: #B4770C;
    --moderate-bg: #FBF0DC;
    --moderate-ink: #8A5A08;

    --weak: #C6303E;
    --weak-bg: #FBE7E9;
    --weak-ink: #94212C;

    --radius-lg: 20px;
    --radius-md: 14px;
    --radius-sm: 10px;

    --shadow-sm: 0 2px 8px rgba(11, 18, 32, 0.04);
    --shadow-md: 0 10px 28px rgba(11, 18, 32, 0.07);
    --shadow-lg: 0 22px 48px rgba(11, 18, 32, 0.14);

    --font-display: 'Space Grotesk', 'Inter', sans-serif;
    --font-body: 'Inter', -apple-system, sans-serif;
    --font-mono: 'JetBrains Mono', ui-monospace, monospace;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

@media (prefers-reduced-motion: reduce) {
    * { animation: none !important; transition: none !important; }
}

/* ---------- GLOBAL ---------- */

html, body, [class*="css"] {
    font-family: var(--font-body);
}

.stApp {
    background:
        radial-gradient(circle at 1px 1px, rgba(11,18,32,0.05) 1px, transparent 0) 0 0/22px 22px,
        var(--bg);
}

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 1440px;
}

h1, h2, h3 {
    font-family: var(--font-display);
    letter-spacing: -0.02em;
    color: var(--ink);
}

hr, [data-testid="stDivider"] {
    border-color: var(--border) !important;
}

/* ---------- SIDEBAR ---------- */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1220 0%, #131B2E 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

[data-testid="stSidebar"] * {
    color: #E7EAF3;
    font-family: var(--font-body);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.4rem;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 2px;
}

.sidebar-mark {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    background: linear-gradient(135deg, #4548E5, #2225A0);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-display);
    font-weight: 800;
    font-size: 16px;
    color: white;
    box-shadow: 0 6px 16px rgba(53, 56, 205, 0.45);
    flex-shrink: 0;
}

.sidebar-logo {
    font-family: var(--font-display);
    font-size: 19px;
    font-weight: 700;
    letter-spacing: -0.01em;
}

.sidebar-subtitle {
    color: #8791A8 !important;
    font-size: 12.5px;
    margin: 2px 0 26px 44px;
    letter-spacing: 0.02em;
}

[data-testid="stSidebar"] .stRadio > label {
    display: none;
}

[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
    padding: 9px 12px;
    border-radius: 10px;
    margin-bottom: 2px;
    transition: background 0.15s ease;
}

[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.06);
}

.sidebar-caption {
    color: #6B7690 !important;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 4px 0 10px 2px;
}

.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 9px 13px;
    border-radius: 999px;
    font-size: 12.5px;
    font-weight: 600;
    width: 100%;
    box-sizing: border-box;
}

.status-chip .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}

.status-ready {
    background: rgba(15, 157, 119, 0.16);
    color: #5FE3B8;
}
.status-ready .dot { background: #34D399; box-shadow: 0 0 0 4px rgba(52,211,153,0.18); }

.status-waiting {
    background: rgba(148, 163, 184, 0.14);
    color: #A9B3C6;
}
.status-waiting .dot { background: #94A3B8; }

/* ---------- HERO ---------- */

.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(120deg, #0B1220 0%, #1B2340 55%, #262F52 100%);
    border-radius: var(--radius-lg);
    padding: 34px 38px;
    margin-bottom: 24px;
    color: white;
    box-shadow: var(--shadow-lg);
    animation: fadeUp 0.5s ease both;
}

.hero::after {
    content: "";
    position: absolute;
    top: -60%;
    right: -10%;
    width: 420px;
    height: 420px;
    background: radial-gradient(circle, rgba(83, 88, 235, 0.35) 0%, rgba(83,88,235,0) 70%);
    pointer-events: none;
}

.hero-eyebrow {
    font-size: 11.5px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #8D97F2;
    margin-bottom: 10px;
}

.hero-title {
    font-family: var(--font-display);
    font-size: 32px;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 8px;
    position: relative;
}

.hero-subtitle {
    color: #B7BEDA;
    font-size: 14.5px;
    max-width: 640px;
    line-height: 1.55;
    position: relative;
}

/* ---------- METRIC CARDS ---------- */

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 20px 22px;
    box-shadow: var(--shadow-sm);
    min-height: 108px;
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
    animation: fadeUp 0.5s ease both;
}

.metric-card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}

.metric-card::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: var(--accent-color, var(--primary));
    border-radius: 4px 0 0 4px;
}

.metric-icon {
    font-size: 17px;
    margin-bottom: 10px;
    display: inline-block;
}

.metric-label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

.metric-value {
    font-family: var(--font-mono);
    color: var(--ink);
    font-size: 27px;
    font-weight: 700;
    margin-top: 6px;
}

/* ---------- SECTION ---------- */

.section-title {
    font-family: var(--font-display);
    font-size: 19px;
    font-weight: 700;
    color: var(--ink);
    margin-top: 30px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 9px;
}

.section-title::before {
    content: "";
    width: 4px;
    height: 18px;
    border-radius: 3px;
    background: var(--primary);
    display: inline-block;
}

/* ---------- CANDIDATE CARD ---------- */

.candidate-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 24px 26px;
    margin-bottom: 14px;
    box-shadow: var(--shadow-sm);
    height: 100%;
    box-sizing: border-box;
}

.candidate-rank {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border-radius: 8px;
    background: var(--primary-tint);
    color: var(--primary);
    font-family: var(--font-mono);
    font-weight: 700;
    font-size: 12px;
    margin-right: 10px;
}

.candidate-name {
    font-family: var(--font-display);
    font-size: 19px;
    font-weight: 700;
    color: var(--ink);
    display: inline-flex;
    align-items: center;
}

.candidate-meta {
    color: var(--muted);
    font-size: 13px;
    margin-top: 6px;
    font-family: var(--font-mono);
}

/* ---------- BADGES ---------- */

.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 13px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.01em;
}

.badge .dot { width: 6px; height: 6px; border-radius: 50%; }

.badge-strong { background: var(--strong-bg); color: var(--strong-ink); }
.badge-strong .dot { background: var(--strong); }

.badge-medium { background: var(--moderate-bg); color: var(--moderate-ink); }
.badge-medium .dot { background: var(--moderate); }

.badge-weak { background: var(--weak-bg); color: var(--weak-ink); }
.badge-weak .dot { background: var(--weak); }

/* ---------- SCORE GAUGE (signature element) ---------- */

.score-gauge-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    height: 100%;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 22px 10px;
    box-shadow: var(--shadow-sm);
    box-sizing: border-box;
}

.score-gauge {
    width: 148px;
    height: 148px;
    border-radius: 50%;
    background: conic-gradient(var(--accent) calc(var(--pct) * 3.6deg), var(--border-soft) 0deg);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 9px;
    box-sizing: border-box;
}

.score-gauge-inner {
    width: 100%;
    height: 100%;
    border-radius: 50%;
    background: var(--surface);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    box-shadow: inset 0 0 0 1px var(--border-soft);
}

.score-gauge-number {
    font-family: var(--font-mono);
    font-size: 30px;
    font-weight: 700;
    color: var(--ink);
    line-height: 1;
}

.score-gauge-number span {
    font-size: 14px;
    color: var(--muted);
    font-weight: 600;
    margin-left: 1px;
}

.score-gauge-label {
    font-size: 11.5px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
}

/* ---------- SKILL PILLS ---------- */

.skill-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--strong-bg);
    color: var(--strong-ink);
    padding: 6px 12px;
    border-radius: 999px;
    margin: 3px;
    font-size: 12px;
    font-weight: 600;
}

.skill-pill::before { content: "✓"; font-weight: 800; }

.missing-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--weak-bg);
    color: var(--weak-ink);
    padding: 6px 12px;
    border-radius: 999px;
    margin: 3px;
    font-size: 12px;
    font-weight: 600;
}

.missing-pill::before { content: "✕"; font-weight: 800; }

.neutral-pill {
    display: inline-block;
    background: var(--primary-tint);
    color: var(--primary-dark);
    padding: 6px 12px;
    border-radius: 999px;
    margin: 3px;
    font-size: 12px;
    font-weight: 600;
}

/* ---------- BUTTONS ---------- */

.stButton > button {
    border-radius: var(--radius-sm);
    font-weight: 700;
    min-height: 46px;
    font-family: var(--font-body);
    border: 1px solid var(--border);
    transition: all 0.15s ease;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    border: none;
    box-shadow: 0 8px 20px rgba(53, 56, 205, 0.3);
}

.stButton > button[kind="primary"]:hover {
    box-shadow: 0 10px 26px rgba(53, 56, 205, 0.42);
    transform: translateY(-1px);
}

.stDownloadButton > button {
    border-radius: var(--radius-sm);
    font-weight: 700;
    border: 1px solid var(--border);
    background: var(--surface);
    min-height: 46px;
    transition: all 0.15s ease;
}

.stDownloadButton > button:hover {
    border-color: var(--primary);
    color: var(--primary);
}

/* ---------- FILE UPLOADER ---------- */

[data-testid="stFileUploader"] {
    background: var(--surface);
    border-radius: var(--radius-md);
    border: 1.5px dashed var(--border);
    padding: 6px;
}

[data-testid="stFileUploader"] section {
    background: transparent;
}

/* ---------- INPUTS ---------- */

.stTextArea textarea, .stSelectbox [data-baseweb="select"] {
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-body);
}

/* ---------- ALERTS / INFO BOXES ---------- */

[data-testid="stAlert"] {
    border-radius: var(--radius-sm);
    font-size: 13.5px;
}

/* ---------- DATAFRAME ---------- */

[data-testid="stDataFrame"] {
    border-radius: var(--radius-md);
    overflow: hidden;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
}

/* ---------- EXPLANATION CARD ---------- */

.explain-card {
    background: var(--primary-tint);
    border: 1px solid #D9DCFA;
    border-left: 3px solid var(--primary);
    border-radius: var(--radius-sm);
    padding: 13px 16px;
    margin-bottom: 9px;
    font-size: 13.5px;
    color: #2B2E7A;
    line-height: 1.5;
}

/* ---------- FOOTER ---------- */

.footer {
    text-align: center;
    color: var(--muted-2);
    font-size: 12px;
    letter-spacing: 0.02em;
    padding: 34px 0 8px 0;
}

</style>
"""

st.markdown(TOKENS, unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "results" not in st.session_state:
    st.session_state.results = []

if "job_requirements" not in st.session_state:
    st.session_state.job_requirements = None

if "screened" not in st.session_state:
    st.session_state.screened = False


results = st.session_state.results
job_requirements = st.session_state.job_requirements


# =========================================================
# HELPER FUNCTIONS
# =========================================================

STRONG, MODERATE, WEAK = "#0F9D77", "#B4770C", "#C6303E"


def get_status(score):

    if score >= 75:
        return "Strong", "badge-strong"

    if score >= 50:
        return "Moderate", "badge-medium"

    return "Weak", "badge-weak"


def get_accent(score):

    if score >= 75:
        return STRONG

    if score >= 50:
        return MODERATE

    return WEAK


def render_badge(score):

    status, css_class = get_status(score)

    st.markdown(
        f'<span class="badge {css_class}"><span class="dot"></span>{status} Match</span>',
        unsafe_allow_html=True,
    )


def render_metric(label, value, icon="◆", accent=None):

    accent = accent or "var(--primary)"

    st.markdown(
        f"""
        <div class="metric-card" style="--accent-color:{accent}">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_score_gauge(score, label="Overall Match"):

    accent = get_accent(score)

    st.markdown(
        f"""
        <div class="score-gauge-wrap">
            <div class="score-gauge" style="--pct:{score};--accent:{accent}">
                <div class="score-gauge-inner">
                    <div class="score-gauge-number">{score}<span>%</span></div>
                </div>
            </div>
            <div class="score-gauge-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_skills(skills, missing=False, empty_text="None detected."):

    if not skills:

        st.caption(empty_text)

        return

    css_class = "missing-pill" if missing else "skill-pill"

    html = ""

    for skill in skills:

        html += f'<span class="{css_class}">{skill}</span>'

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


PLOTLY_FONT = dict(family="Inter, sans-serif", color="#334155", size=13)


def style_score_chart(fig, values):

    colors = [get_accent(v) for v in values]

    fig.update_traces(
        marker_color=colors,
        marker_line_width=0,
        textfont=dict(family="JetBrains Mono, monospace", size=12, color="#0B1220"),
        texttemplate="%{text}%",
        textposition="outside",
        cliponaxis=False,
    )

    fig.update_layout(
        yaxis_range=[0, max(100, max(values) + 15) if values else 100],
        yaxis_title="Score (%)",
        xaxis_title=None,
        template="plotly_white",
        height=420,
        font=PLOTLY_FONT,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10),
    )

    fig.update_xaxes(showline=True, linecolor="#E2E6EF", tickfont=dict(size=12))
    fig.update_yaxes(gridcolor="#EEF1F7", zeroline=False)

    return fig


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-mark">◈</div>
            <div class="sidebar-logo">ResumeIQ</div>
        </div>
        <div class="sidebar-subtitle">Intelligent Resume Screening</div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Job Setup",
            "Candidates",
            "Reports",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown('<div class="sidebar-caption">System</div>', unsafe_allow_html=True)

    if st.session_state.screened:

        st.markdown(
            '<div class="status-chip status-ready"><span class="dot"></span>'
            f"Screening ready · {len(results)} candidates</div>",
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            '<div class="status-chip status-waiting"><span class="dot"></span>'
            "Waiting for screening</div>",
            unsafe_allow_html=True,
        )


# =========================================================
# JOB SETUP
# =========================================================

if page == "Job Setup":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">Step 1 of 1</div>
            <div class="hero-title">Job Setup</div>
            <div class="hero-subtitle">
                Define the role requirements and upload candidate resumes.
                ResumeIQ will parse, score, and rank every applicant automatically.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:

        st.markdown(
            '<div class="section-title">Job Description</div>', unsafe_allow_html=True
        )

        job_text = st.text_area(
            "Job Description",
            height=280,
            placeholder=("Paste your complete job description here..."),
            label_visibility="collapsed",
        )

    with col_right:

        st.markdown(
            '<div class="section-title">Candidate Resumes</div>', unsafe_allow_html=True
        )

        resume_files = st.file_uploader(
            "Upload Resume PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if resume_files:

            st.markdown(
                f'<span class="neutral-pill">{len(resume_files)} resume(s) selected</span>',
                unsafe_allow_html=True,
            )

    st.write("")

    if st.button(
        "Start Resume Screening",
        use_container_width=True,
        type="primary",
    ):

        if not job_text.strip():

            st.error("Please enter a job description.")

        elif not resume_files:

            st.error("Please upload at least one resume.")

        else:

            with st.spinner("Analyzing resumes..."):

                job_cleaned_text = preprocess_text(job_text)

                job_requirements = extract_job_requirements(job_text)

                job_skills = job_requirements["required_skills"]

                results = []

                progress = st.progress(0)

                total = len(resume_files)

                for index, resume_file in enumerate(resume_files):

                    temp_pdf_path = None

                    try:

                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".pdf",
                        ) as temp_file:

                            temp_file.write(resume_file.getbuffer())

                            temp_pdf_path = temp_file.name

                        raw_text = extract_text_from_pdf(temp_pdf_path)

                        cleaned_text = preprocess_text(raw_text)

                        information = extract_information(cleaned_text)

                        similarity = calculate_similarity(
                            job_cleaned_text,
                            cleaned_text,
                        )

                        tfidf_score = round(
                            similarity * 100,
                            2,
                        )

                        skill_result = calculate_skill_match(
                            job_skills,
                            information["skills"],
                        )

                        skill_score = skill_result["skill_score"]

                        experience_score = calculate_experience_match(
                            information["experience_years"],
                            job_requirements["experience_required"],
                        )

                        education_score = calculate_education_match(
                            information["education"],
                            job_requirements["required_education"],
                        )

                        final_score = calculate_final_score(
                            tfidf_score,
                            skill_score,
                            experience_score,
                            education_score,
                        )

                        explanation = generate_candidate_explanation(
                            {
                                "skills": information["skills"],
                                "experience": information["experience_years"],
                                "education_score": education_score,
                            },
                            job_requirements,
                        )

                        results.append(
                            {
                                "name": information["name"],
                                "resume": resume_file.name,
                                "email": information["email"],
                                "phone": information["phone"],
                                "skills": information["skills"],
                                "matched_skills": skill_result["matched_skills"],
                                "missing_skills": skill_result["missing_skills"],
                                "education": information["education"],
                                "experience": information["experience_years"],
                                "tfidf_score": tfidf_score,
                                "skill_match_score": skill_score,
                                "experience_score": experience_score,
                                "education_score": education_score,
                                "final_score": final_score,
                                "explanation": explanation,
                            }
                        )

                    except Exception as error:

                        st.warning(
                            f"Could not process " f"{resume_file.name}: " f"{error}"
                        )

                    finally:

                        if temp_pdf_path and os.path.exists(temp_pdf_path):

                            os.remove(temp_pdf_path)

                    progress.progress((index + 1) / total)

                results.sort(
                    key=lambda x: x["final_score"],
                    reverse=True,
                )

                for rank, candidate in enumerate(
                    results,
                    start=1,
                ):

                    candidate["rank"] = rank

                st.session_state.results = results

                st.session_state.job_requirements = job_requirements

                st.session_state.screened = True

            st.success(f"Screening completed for {len(results)} candidates.")

            st.info("Go to Dashboard to view the results.")


# =========================================================
# DASHBOARD
# =========================================================

elif page == "Dashboard":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">Overview</div>
            <div class="hero-title">Resume Intelligence</div>
            <div class="hero-subtitle">
                Screen, compare and rank candidates using AI-assisted resume analysis
                across skills, experience, and education fit.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not results:

        st.info("No screening results yet. Go to Job Setup to begin.")

    else:

        total = len(results)

        average = round(
            sum(candidate["final_score"] for candidate in results) / total,
            2,
        )

        strong = sum(candidate["final_score"] >= 75 for candidate in results)

        moderate = sum(50 <= candidate["final_score"] < 75 for candidate in results)

        top = results[0]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            render_metric(
                "Candidates Screened", total, icon="◧", accent="var(--primary)"
            )

        with col2:
            render_metric("Strong Matches", strong, icon="◆", accent=STRONG)

        with col3:
            render_metric("Moderate Matches", moderate, icon="◐", accent=MODERATE)

        with col4:
            render_metric(
                "Average Match", f"{average}%", icon="◎", accent="var(--primary)"
            )

        st.markdown(
            '<div class="section-title">Top Candidate</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([4, 1.3])

        with col1:

            st.markdown(
                f"""
                <div class="candidate-card">

                    <div>
                        <span class="candidate-rank">01</span>
                        <span class="candidate-name">{top["name"]}</span>
                    </div>

                    <div class="candidate-meta">{top["email"]}</div>

                    <br>
                """,
                unsafe_allow_html=True,
            )

            render_badge(top["final_score"])

            st.markdown("</div>", unsafe_allow_html=True)

        with col2:

            render_score_gauge(top["final_score"], "Overall Match")

        # Candidate score chart

        st.markdown(
            '<div class="section-title">Candidate Performance</div>',
            unsafe_allow_html=True,
        )

        chart_data = pd.DataFrame(
            {
                "Candidate": [c["name"] for c in results],
                "Score": [c["final_score"] for c in results],
            }
        )

        fig = px.bar(
            chart_data,
            x="Candidate",
            y="Score",
            text="Score",
        )

        fig = style_score_chart(fig, chart_data["Score"].tolist())

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# =========================================================
# CANDIDATES
# =========================================================

elif page == "Candidates":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">Deep Dive</div>
            <div class="hero-title">Candidate Intelligence</div>
            <div class="hero-subtitle">
                Explore individual candidate profiles and understand exactly
                why they ranked where they did.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not results:

        st.info("No candidates available.")

    else:

        candidate_labels = [
            (f"#{c['rank']}  {c['name']}  — {c['final_score']}%") for c in results
        ]

        selected = st.selectbox(
            "Select Candidate",
            candidate_labels,
        )

        index = candidate_labels.index(selected)

        candidate = results[index]

        col1, col2 = st.columns([3, 1.1])

        with col1:

            st.markdown(
                f"""
                <div class="candidate-card">

                    <div>
                        <span class="candidate-rank">{candidate["rank"]:02d}</span>
                        <span class="candidate-name">{candidate["name"]}</span>
                    </div>

                    <div class="candidate-meta">
                        {candidate["email"]} &nbsp;·&nbsp; {candidate["phone"]}
                    </div>

                    <br>
                """,
                unsafe_allow_html=True,
            )

            render_badge(candidate["final_score"])

            st.markdown("</div>", unsafe_allow_html=True)

        with col2:

            render_score_gauge(candidate["final_score"], "Match Score")

        # Skills

        st.markdown(
            '<div class="section-title">Skills</div>',
            unsafe_allow_html=True,
        )

        render_skills(candidate["skills"])

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("#### Matched Skills")

            render_skills(
                candidate["matched_skills"], empty_text="No skill overlap detected."
            )

        with col2:

            st.markdown("#### Missing Skills")

            render_skills(
                candidate["missing_skills"],
                missing=True,
                empty_text="No gaps — full coverage.",
            )

        # Education / experience

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                '<div class="section-title">Education</div>',
                unsafe_allow_html=True,
            )

            for education in candidate["education"] or ["Not detected"]:

                st.write(f"• {education}")

        with col2:

            st.markdown(
                '<div class="section-title">Experience</div>',
                unsafe_allow_html=True,
            )

            for experience in candidate["experience"] or ["Not detected"]:

                st.write(f"• {experience}")

        # Score breakdown

        st.markdown(
            '<div class="section-title">Score Breakdown</div>',
            unsafe_allow_html=True,
        )

        score_data = pd.DataFrame(
            {
                "Component": [
                    "TF-IDF",
                    "Skills",
                    "Experience",
                    "Education",
                ],
                "Score": [
                    candidate["tfidf_score"],
                    candidate["skill_match_score"],
                    candidate["experience_score"],
                    candidate["education_score"],
                ],
            }
        )

        fig = px.bar(
            score_data,
            x="Component",
            y="Score",
            text="Score",
        )

        fig = style_score_chart(fig, score_data["Score"].tolist())
        fig.update_layout(height=380)

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        # Explanation

        st.markdown(
            '<div class="section-title">Why This Candidate Received This Score</div>',
            unsafe_allow_html=True,
        )

        for reason in candidate["explanation"]:

            st.markdown(
                f'<div class="explain-card">{reason}</div>', unsafe_allow_html=True
            )


# =========================================================
# REPORTS
# =========================================================

elif page == "Reports":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">Export</div>
            <div class="hero-title">Reports &amp; Analytics</div>
            <div class="hero-subtitle">
                Export your screening results and analyze candidate performance
                and skill coverage across the applicant pool.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not results:

        st.info("No results available for reporting.")

    else:

        report_data = []

        for candidate in results:

            report_data.append(
                {
                    "Rank": candidate["rank"],
                    "Candidate": candidate["name"],
                    "Resume": candidate["resume"],
                    "Email": candidate["email"],
                    "Phone": candidate["phone"],
                    "Skills": ", ".join(
                        map(
                            str,
                            candidate["skills"],
                        )
                    ),
                    "Matched Skills": ", ".join(
                        map(
                            str,
                            candidate["matched_skills"],
                        )
                    ),
                    "Missing Skills": ", ".join(
                        map(
                            str,
                            candidate["missing_skills"],
                        )
                    ),
                    "Education": ", ".join(
                        map(
                            str,
                            candidate["education"],
                        )
                    ),
                    "Experience": ", ".join(
                        map(
                            str,
                            candidate["experience"],
                        )
                    ),
                    "TF-IDF": candidate["tfidf_score"],
                    "Skill Match": candidate["skill_match_score"],
                    "Experience Score": candidate["experience_score"],
                    "Education Score": candidate["education_score"],
                    "Final Score": candidate["final_score"],
                }
            )

        df = pd.DataFrame(report_data)

        st.markdown(
            '<div class="section-title">Ranking Table</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        col1, col2 = st.columns(2)

        # CSV

        csv_data = df.to_csv(index=False)

        with col1:

            st.download_button(
                "Download CSV",
                data=csv_data,
                file_name=("resume_screening_report.csv"),
                mime="text/csv",
                use_container_width=True,
            )

        # Excel

        excel_buffer = BytesIO()

        with pd.ExcelWriter(
            excel_buffer,
            engine="openpyxl",
        ) as writer:

            df.to_excel(
                writer,
                index=False,
                sheet_name="Candidate Ranking",
            )

        excel_buffer.seek(0)

        with col2:

            st.download_button(
                "Download Excel",
                data=excel_buffer,
                file_name=("resume_screening_report.xlsx"),
                mime=(
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet"
                ),
                use_container_width=True,
            )

        # Missing skills

        st.markdown(
            '<div class="section-title">Skill Gap Analysis</div>',
            unsafe_allow_html=True,
        )

        missing_counter = Counter()

        for candidate in results:

            for skill in candidate["missing_skills"]:

                missing_counter[skill] += 1

        if missing_counter:

            missing_df = pd.DataFrame(
                {
                    "Skill": list(missing_counter.keys()),
                    "Candidates": list(missing_counter.values()),
                }
            ).sort_values("Candidates", ascending=False)

            fig = px.bar(
                missing_df,
                x="Skill",
                y="Candidates",
                text="Candidates",
            )

            fig.update_traces(
                marker_color="#C6303E",
                marker_line_width=0,
                textfont=dict(
                    family="JetBrains Mono, monospace", size=12, color="#0B1220"
                ),
                textposition="outside",
                cliponaxis=False,
            )

            fig.update_layout(
                template="plotly_white",
                height=400,
                font=PLOTLY_FONT,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title=None,
                margin=dict(l=10, r=10, t=20, b=10),
            )

            fig.update_xaxes(showline=True, linecolor="#E2E6EF")
            fig.update_yaxes(gridcolor="#EEF1F7", zeroline=False)

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:

            st.success("No skill gaps detected across the candidate pool.")


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        ResumeIQ · Intelligent Resume Screening System
    </div>
    """,
    unsafe_allow_html=True,
)
