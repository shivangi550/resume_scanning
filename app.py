import os
import tempfile
from collections import Counter
from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st

from src.extract_text import extract_text_from_pdf
from src.preprocess import preprocess_text
from src.information_extraction import extract_information
from src.matching import calculate_similarity
from src.skill_matching import calculate_skill_match
from src.job_description import extract_job_requirements
from src.explanation import generate_candidate_explanation
from src.word2vec_matching import calculate_word2vec_similarity
from src.sbert_matching import calculate_sbert_similarity


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
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "screening_results" not in st.session_state:
    st.session_state["screening_results"] = []


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: #f6f8fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    h1, h2, h3 {
        letter-spacing: -0.5px;
    }


    /* ---------- SIDEBAR ---------- */

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1f2937;
    }

    [data-testid="stSidebar"] * {
        color: #f9fafb;
    }

    .sidebar-logo {
        font-size: 25px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .sidebar-subtitle {
        color: #9ca3af !important;
        font-size: 13px;
        margin-bottom: 30px;
    }


    /* ---------- HEADER ---------- */

    .hero {
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1f2937 100%
        );

        border-radius: 22px;
        padding: 32px 36px;
        margin-bottom: 26px;
        color: white;
        box-shadow: 0 12px 30px rgba(17, 24, 39, 0.12);
    }

    .hero-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #d1d5db;
        font-size: 15px;
    }


    /* ---------- METRIC CARDS ---------- */

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
        min-height: 120px;
    }

    .metric-label {
        color: #6b7280;
        font-size: 13px;
        font-weight: 600;
    }

    .metric-value {
        color: #111827;
        font-size: 28px;
        font-weight: 800;
        margin-top: 8px;
    }


    /* ---------- SECTION ---------- */

    .section-title {
        font-size: 21px;
        font-weight: 750;
        color: #111827;
        margin-top: 28px;
        margin-bottom: 15px;
    }


    /* ---------- CANDIDATE CARD ---------- */

    .candidate-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 14px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
    }

    .candidate-name {
        font-size: 18px;
        font-weight: 750;
        color: #111827;
    }

    .candidate-meta {
        color: #6b7280;
        font-size: 13px;
    }


    /* ---------- BADGES ---------- */

    .badge {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
    }

    .badge-strong {
        background: #dcfce7;
        color: #166534;
    }

    .badge-medium {
        background: #fef3c7;
        color: #92400e;
    }

    .badge-weak {
        background: #fee2e2;
        color: #991b1b;
    }


    /* ---------- SCORE ---------- */

    .score-box {
        background: linear-gradient(
            135deg,
            #111827,
            #374151
        );

        border-radius: 20px;
        padding: 25px;
        color: white;
        text-align: center;
    }

    .score-number {
        font-size: 44px;
        font-weight: 850;
    }

    .score-label {
        color: #d1d5db;
        font-size: 13px;
    }


    /* ---------- SKILL PILLS ---------- */

    .skill-pill {
        display: inline-block;
        background: #eef2ff;
        color: #3730a3;
        padding: 7px 12px;
        border-radius: 999px;
        margin: 3px;
        font-size: 12px;
        font-weight: 600;
    }

    .missing-pill {
        display: inline-block;
        background: #fef2f2;
        color: #b91c1c;
        padding: 7px 12px;
        border-radius: 999px;
        margin: 3px;
        font-size: 12px;
        font-weight: 600;
    }


    /* ---------- BUTTONS ---------- */

    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 44px;
    }


    /* ---------- FILE UPLOADER ---------- */

    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 15px;
    }


    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 12px;
        padding: 30px 0 10px 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


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


def get_status(score):

    if score >= 75:
        return "Strong", "badge-strong"

    if score >= 50:
        return "Moderate", "badge-medium"

    return "Weak", "badge-weak"


def render_metric(label, value):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_skills(skills, missing=False):

    if not skills:

        st.caption("None detected.")

        return

    css_class = "missing-pill" if missing else "skill-pill"

    html = ""

    for skill in skills:

        html += f'<span class="{css_class}">' f"{skill}" f"</span>"

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-logo">
            ✦ ResumeIQ
        </div>
        <div class="sidebar-subtitle">
            Intelligent Resume Screening
        </div>
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

    st.caption("SYSTEM")

    if st.session_state.screened:

        st.success("● Screening Ready")

    else:

        st.info("● Waiting for Screening")


# =========================================================
# JOB SETUP
# =========================================================

if page == "Job Setup":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Job Setup</div>
            <div class="hero-subtitle">
                Define the role and upload candidate resumes.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    job_text = st.text_area(
        "Job Description",
        height=280,
        placeholder=("Paste your complete job description here..."),
    )

    resume_files = st.file_uploader(
        "Upload Resume PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if resume_files:

        st.info(f"{len(resume_files)} resume(s) selected.")

    if st.button(
        " Start Resume Screening",
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

                        information = extract_information(raw_text)

                        similarity = calculate_similarity(
                            job_cleaned_text,
                            cleaned_text,
                        )

                        tfidf_score = round(
                            similarity * 100,
                            2,
                        )

                        word2vec_similarity = calculate_word2vec_similarity(
                            job_cleaned_text,
                            cleaned_text,
                        )

                        word2vec_score = round(word2vec_similarity * 100, 2)

                        sbert_similarity = calculate_sbert_similarity(
                            job_text,
                            raw_text,
                        )

                        sbert_score = round(sbert_similarity * 100, 2)

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
                                "word2vec_score": word2vec_score,
                                "sbert_score": sbert_score,
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

                st.session_state["screening_results"] = results

                st.session_state.results = results

                st.session_state.job_requirements = job_requirements

                st.session_state.screened = True

            st.success(f"Screening completed for " f"{len(results)} candidates.")

            st.info("Go to Dashboard to view the results.")


# =========================================================
# DASHBOARD
# =========================================================

elif page == "Dashboard":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                Resume Intelligence
            </div>
            <div class="hero-subtitle">
                Screen, compare and rank candidates
                using AI-assisted resume analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not results:

        st.info("No screening results yet. " "Go to Job Setup to begin.")

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
                "Candidates Screened",
                total,
            )

        with col2:
            render_metric(
                "Strong Matches",
                strong,
            )

        with col3:
            render_metric(
                "Moderate Matches",
                moderate,
            )

        with col4:
            render_metric(
                "Average Match",
                f"{average}%",
            )

        st.markdown(
            '<div class="section-title">' " Top Candidate" "</div>",
            unsafe_allow_html=True,
        )

        status, badge = get_status(top["final_score"])

        col1, col2 = st.columns([4, 1])

        with col1:

            st.markdown(
                f"""
                <div class="candidate-card">
                    <div class="candidate-name">
                        {top["name"]}
                    </div>
                    <div class="candidate-meta">
                        {top["email"]}
                    </div>
                    <br>
                    <span class="badge {badge}">
                        {status} Match
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:

            st.markdown(
                f"""
                <div class="score-box">
                    <div class="score-number">
                        {top["final_score"]}%
                    </div>
                    <div class="score-label">
                        Overall Match
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Candidate score chart

        st.markdown(
            '<div class="section-title">' " Candidate Performance" "</div>",
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

        fig.update_traces(
            texttemplate="%{text}%",
            textposition="outside",
        )

        fig.update_layout(
            yaxis_range=[0, 100],
            yaxis_title="Match Score (%)",
            xaxis_title=None,
            template="plotly_white",
            height=420,
        )

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
            <div class="hero-title">
                Candidate Intelligence
            </div>
            <div class="hero-subtitle">
                Explore candidate profiles and
                understand exactly why they ranked.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not results:

        st.info("No candidates available.")

    else:

        candidate_labels = [
            (f"#{c['rank']}  " f"{c['name']}  " f"— {c['final_score']}%")
            for c in results
        ]

        selected = st.selectbox(
            "Select Candidate",
            candidate_labels,
        )

        index = candidate_labels.index(selected)

        candidate = results[index]

        status, badge = get_status(candidate["final_score"])

        col1, col2 = st.columns([3, 1])

        with col1:

            st.markdown(
                f"""
                <div class="candidate-card">
                    <div class="candidate-name">
                        {candidate["name"]}
                    </div>
                    <div class="candidate-meta">
                        {candidate["email"]}
                        &nbsp; • &nbsp;
                        {candidate["phone"]}
                    </div>
                    <br>
                    <span class="badge {badge}">
                        {status} Match
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:

            st.markdown(
                f"""
                <div class="score-box">
                    <div class="score-number">
                        {candidate["final_score"]}%
                    </div>
                    <div class="score-label">
                        Match Score
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Skills

        st.markdown(
            '<div class="section-title">' " Skills" "</div>",
            unsafe_allow_html=True,
        )

        render_skills(candidate["skills"])

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("####  Matched Skills")

            render_skills(candidate["matched_skills"])

        with col2:

            st.markdown("####  Missing Skills")

            render_skills(
                candidate["missing_skills"],
                missing=True,
            )

        # Education / experience

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                '<div class="section-title">' " Education" "</div>",
                unsafe_allow_html=True,
            )

            for education in candidate["education"] or ["Not detected"]:

                st.write(f"• {education}")

        with col2:

            st.markdown(
                '<div class="section-title">' " Experience" "</div>",
                unsafe_allow_html=True,
            )

            for experience in candidate["experience"] or ["Not detected"]:

                st.write(f"• {experience}")

        # Score breakdown

        st.markdown(
            '<div class="section-title">' " Score Breakdown" "</div>",
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

        fig.update_traces(
            texttemplate="%{text}%",
            textposition="outside",
        )

        fig.update_layout(
            yaxis_range=[0, 100],
            yaxis_title="Score (%)",
            xaxis_title=None,
            template="plotly_white",
            height=400,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        # Explanation

        st.markdown(
            '<div class="section-title">'
            " Why This Candidate Received This Score"
            "</div>",
            unsafe_allow_html=True,
        )

        for reason in candidate["explanation"]:

            st.info(reason)


# =========================================================
# REPORTS
# =========================================================

elif page == "Reports":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                Reports & Analytics
            </div>
            <div class="hero-subtitle">
                Export your screening results and
                analyze candidate performance.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not results:

        st.info("No results available for reporting.")

    else:

        # =====================================================
        # NLP MODEL COMPARISON
        # =====================================================

        st.markdown(
            '<div class="section-title">NLP Model Comparison</div>',
            unsafe_allow_html=True,
        )

        comparison_rows = []

        for candidate in results:

            comparison_rows.append(
            {
                "Rank": candidate["rank"],
                "Candidate": candidate["name"],
                "TF-IDF Score": candidate["tfidf_score"],
                "Word2Vec Score": candidate["word2vec_score"],
                "SBERT Score": candidate["sbert_score"],
            }
        )

        comparison_df = pd.DataFrame(comparison_rows)

        st.dataframe(
            comparison_df.style.format(
                {
                    "TF-IDF Score": "{:.2f}%",
                    "Word2Vec Score": "{:.2f}%",
                    "SBERT Score": "{:.2f}%",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown(
            '<div class="section-title">Model Performance Visualization</div>',
            unsafe_allow_html=True,
        )

        chart_comparison_data = comparison_df.melt(
            id_vars=["Candidate"],
            value_vars=[
                "TF-IDF Score",
                "Word2Vec Score",
                "SBERT Score",
            ],
            var_name="Model",
            value_name="Similarity Score",
        )

        fig = px.bar(
            chart_comparison_data,
            x="Candidate",
            y="Similarity Score",
            color="Model",
            barmode="group",
            text="Similarity Score",
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
        )

        fig.update_layout(
            yaxis_range=[0, 100],
            yaxis_title="Similarity Score (%)",
            xaxis_title=None,
            template="plotly_white",
            height=450,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
     )

        # =====================================================
        # REPORT DATA
        # =====================================================

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
                    "Word2Vec": candidate["word2vec_score"],
                    "SBERT": candidate["sbert_score"],
                    "Skill Match": candidate["skill_match_score"],
                    "Experience Score": candidate["experience_score"],
                    "Education Score": candidate["education_score"],
                    "Final Score": candidate["final_score"],
                }
            )

        df = pd.DataFrame(report_data)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        # CSV

        csv_data = df.to_csv(index=False)

        st.download_button(
            "⬇ Download CSV",
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

        st.download_button(
            "⬇ Download Excel",
            data=excel_buffer,
            file_name=("resume_screening_report.xlsx"),
            mime=(
                "application/vnd.openxmlformats-" "officedocument.spreadsheetml.sheet"
            ),
            use_container_width=True,
        )

        # Missing skills

        st.markdown(
            '<div class="section-title">' " Skill Gap Analysis" "</div>",
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
            )

            fig = px.bar(
                missing_df,
                x="Skill",
                y="Candidates",
                text="Candidates",
            )

            fig.update_layout(
                template="plotly_white",
                height=400,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:

            st.success("No skill gaps detected.")


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        ResumeIQ • Intelligent Resume Screening System
    </div>
    """,
    unsafe_allow_html=True,
)
