def calculate_experience_match(candidate_experience, required_experience):
    """
    Compare candidate experience with the
    experience required by the job.
    """

    if required_experience <= 0:
        return 100

    if not candidate_experience:
        return 0

    try:
        candidate_years = max(float(year) for year in candidate_experience)
    except (ValueError, TypeError):
        return 0

    if candidate_years >= required_experience:
        return 100

    score = (candidate_years / required_experience) * 100

    return round(score, 2)


def calculate_education_match(candidate_education, required_education):
    """
    Compare candidate education with the
    education requirements of the job.
    """

    if not required_education:
        return 100

    if not candidate_education:
        return 0

    candidate_text = " ".join(candidate_education).lower()

    matches = 0

    for education in required_education:

        education = education.lower()

        if education in candidate_text:
            matches += 1

    score = (matches / len(required_education)) * 100

    return round(score, 2)


def calculate_final_score(tfidf_score, skill_score, experience_score, education_score):
    """
    Calculate the final candidate score.

    Weights:
    TF-IDF       = 30%
    Skills       = 50%
    Experience   = 15%
    Education    = 5%
    """

    final_score = (
        0.30 * tfidf_score
        + 0.50 * skill_score
        + 0.15 * experience_score
        + 0.05 * education_score
    )

    return round(final_score, 2)
