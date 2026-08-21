import re


def extract_job_requirements(job_text):
    """
    Extract basic requirements from a job description.
    """

    text = job_text.lower()

    # -----------------------------------------
    # Skills
    # -----------------------------------------

    possible_skills = [
        "python",
        "java",
        "c++",
        "sql",
        "pandas",
        "numpy",
        "machine learning",
        "deep learning",
        "tensorflow",
        "pytorch",
        "scikit-learn",
        "git",
        "docker",
        "aws",
        "django",
        "flask",
        "rest api",
        "excel",
        "power bi",
        "tableau",
    ]

    required_skills = []

    for skill in possible_skills:

        if skill.lower() in text:
            required_skills.append(skill)

    # -----------------------------------------
    # Experience
    # -----------------------------------------

    experience_required = 0

    experience_patterns = [
        r"(\d+)\+?\s*years?\s*(?:of)?\s*experience",
        r"minimum\s*(?:of)?\s*(\d+)\s*years?",
        r"at least\s*(\d+)\s*years?",
    ]

    for pattern in experience_patterns:

        match = re.search(pattern, text)

        if match:

            experience_required = int(match.group(1))

            break

    # -----------------------------------------
    # Education
    # -----------------------------------------

    education_keywords = [
        "b.tech",
        "btech",
        "b.e",
        "bachelor",
        "b.sc",
        "bsc",
        "m.tech",
        "mtech",
        "master",
        "computer science",
        "engineering",
    ]

    required_education = []

    for education in education_keywords:

        if education in text:

            required_education.append(education)

    return {
        "required_skills": sorted(required_skills),
        "experience_required": experience_required,
        "required_education": sorted(required_education),
    }


if __name__ == "__main__":

    job_text = """
    We are looking for a Python Developer.

    Requirements:
    B.Tech in Computer Science.
    2+ years of experience.

    Skills required:
    Python, SQL, Pandas, Machine Learning and Git.
    """

    requirements = extract_job_requirements(job_text)

    print(requirements)
