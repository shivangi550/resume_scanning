def calculate_skill_match(required_skills, candidate_skills):
    """
    Compare required job skills with candidate skills.
    """

    # Convert skills to lowercase
    required = set(
        skill.lower().strip()
        for skill in required_skills
    )

    candidate = set(
        skill.lower().strip()
        for skill in candidate_skills
    )

    # Skills present in both
    matched_skills = required.intersection(candidate)

    # Required skills missing from candidate
    missing_skills = required.difference(candidate)

    # Calculate percentage
    if len(required) == 0:
        skill_score = 0
    else:
        skill_score = (
            len(matched_skills) / len(required)
        ) * 100

    return {
        "matched_skills": sorted(
            list(matched_skills)
        ),

        "missing_skills": sorted(
            list(missing_skills)
        ),

        "skill_score": round(
            skill_score,
            2
        )
    }
if __name__ == "__main__":

    required_skills = [
        "python",
        "sql",
        "pandas",
        "numpy",
        "machine learning",
        "git",
        "django"
    ]

    candidate_skills = [
        "python",
        "sql",
        "pandas",
        "git"
    ]

    result = calculate_skill_match(
        required_skills,
        candidate_skills
    )

    print("Matched Skills:")
    print(result["matched_skills"])

    print("\nSkill Match Score:")
    print(result["skill_score"], "%")