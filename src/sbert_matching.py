from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_sbert_model():
    """
    Load the SBERT model only once.
    The first run downloads the model automatically.
    """
    return SentenceTransformer(MODEL_NAME)


def calculate_sbert_similarity(job_text, resume_text):
    """
    Compare a job description and resume using SBERT.

    Returns:
        A similarity score between 0 and 1.
    """
    if not job_text.strip() or not resume_text.strip():
        return 0.0

    model = get_sbert_model()

    embeddings = model.encode(
        [job_text, resume_text],
        normalize_embeddings=True,
    )

    similarity = float(np.dot(embeddings[0], embeddings[1]))

    return max(0.0, min(1.0, similarity))


if __name__ == "__main__":
    job_description = """
    We need a Python Data Analyst with experience in SQL,
    Pandas, machine learning, data visualization and Git.
    """

    candidate_resume = """
    Python developer with two years of data analysis experience.
    Skilled in SQL, Pandas, machine learning, Git and dashboards.
    """

    score = calculate_sbert_similarity(
        job_description,
        candidate_resume,
    )

    print(f"SBERT similarity score: {score * 100:.2f}%")
