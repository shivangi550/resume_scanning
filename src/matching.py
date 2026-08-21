from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(job_description, resume_text):
    """
    Calculate similarity between a job description
    and a resume using TF-IDF and cosine similarity.
    """

    documents = [
        job_description,
        resume_text
    ]

    # Convert text into TF-IDF vectors
    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(documents)

    # Calculate cosine similarity
    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )

    score = similarity[0][0]

    return score

if __name__ == "__main__":

    job_description = """
    Python Developer with experience in Python, SQL,
    Pandas, NumPy, Machine Learning and Git.
    """

    resume = """
    Python developer with experience in Python, SQL,
    Pandas, Machine Learning and Git.
    """

    score = calculate_similarity(
        job_description,
        resume
    )

    print("Similarity Score:", score)
    print("Match Percentage:", round(score * 100, 2), "%")