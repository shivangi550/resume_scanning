import re

import numpy as np
from gensim.models import Word2Vec


def tokenize_text(text):
    """
    Convert text into a list of clean words.
    Example:
    'Python, SQL and Pandas!' -> ['python', 'sql', 'and', 'pandas']
    """
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def get_document_vector(tokens, model):
    """
    Create one vector for a whole document by averaging
    the vectors of all words found in the Word2Vec model.
    """
    word_vectors = [model.wv[word] for word in tokens if word in model.wv]

    if not word_vectors:
        return np.zeros(model.vector_size)

    return np.mean(word_vectors, axis=0)


def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two document vectors.
    """
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)

    if denominator == 0:
        return 0.0

    return float(np.dot(vector_a, vector_b) / denominator)


def calculate_word2vec_similarity(job_text, resume_text):
    """
    Compare a job description and a resume using Word2Vec.

    Returns:
        A similarity score between 0 and 1.
    """
    job_tokens = tokenize_text(job_text)
    resume_tokens = tokenize_text(resume_text)

    if not job_tokens or not resume_tokens:
        return 0.0

    # Train a small Word2Vec model using the job description
    # and resume as the current text corpus.
    sentences = [job_tokens, resume_tokens]

    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=1,
        seed=42,
        epochs=100,
    )

    job_vector = get_document_vector(job_tokens, model)
    resume_vector = get_document_vector(resume_tokens, model)

    similarity = cosine_similarity(job_vector, resume_vector)

    # Convert possible negative values to 0.
    return max(0.0, similarity)


if __name__ == "__main__":
    job_description = """
    Looking for a Python Data Analyst with SQL, Pandas,
    NumPy, machine learning, data visualization and Git skills.
    """

    candidate_resume = """
    Python developer with two years of experience in data analysis.
    Skilled in SQL, Pandas, NumPy, Git and machine learning.
    Created dashboards and data visualizations.
    """

    score = calculate_word2vec_similarity(
        job_description,
        candidate_resume,
    )

    print(f"Word2Vec similarity score: {score * 100:.2f}%")
