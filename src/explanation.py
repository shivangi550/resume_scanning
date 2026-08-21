def generate_candidate_explanation(candidate, job_requirements):
    """
    Generate a simple explanation for the candidate's ranking.
    """

    required_skills = set(
        skill.lower() for skill in job_requirements["required_skills"]
    )

    candidate_skills = set(skill.lower() for skill in candidate["skills"])

    matched_skills = candidate_skills.intersection(required_skills)

    missing_skills = required_skills.difference(candidate_skills)

    required_experience = job_requirements["experience_required"]

    candidate_experience = 0

    if candidate["experience"]:

        try:
            candidate_experience = max(float(year) for year in candidate["experience"])

        except (ValueError, TypeError):
            candidate_experience = 0

    # ------------------------------------------
    # Build explanation
    # ------------------------------------------

    explanation = []

    # Skills
    if matched_skills:

        explanation.append(
            f"Matched {len(matched_skills)} "
            f"of {len(required_skills)} required skills."
        )

    else:

        explanation.append("No required skills were matched.")

    # Missing skills
    if missing_skills:

        explanation.append("Missing skills: " + ", ".join(sorted(missing_skills)) + ".")

    else:

        explanation.append("The candidate has all required skills.")

    # Experience
    if required_experience > 0:

        if candidate_experience >= required_experience:

            explanation.append(
                f"Experience requirement met " f"({candidate_experience:g} years)."
            )

        else:

            explanation.append(
                f"Experience requirement not met "
                f"({candidate_experience:g} of "
                f"{required_experience} years)."
            )

    else:

        explanation.append("No specific experience requirement " "was detected.")

    # Education
    education_score = candidate["education_score"]

    if education_score >= 100:

        explanation.append("Education requirement matched.")

    elif education_score > 0:

        explanation.append("Education partially matches " "the job requirement.")

    else:

        explanation.append("Required education was not detected.")

    return explanation
