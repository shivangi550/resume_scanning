import re

# List of skills that our system can recognize
SKILLS = [
    "python",
    "java",
    "c++",
    "sql",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "excel",
    "power bi",
    "tableau",
    "git",
    "github",
    "docker",
    "aws",
    "azure",
    "flask",
    "django",
    "html",
    "css",
    "javascript",
    "react",
]


def extract_email(text):
    """
    Extract an email address from the resume text.
    """

    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"

    match = re.search(pattern, text)

    if match:
        return match.group()

    return None


def extract_phone(text):
    """
    Extract an Indian phone number from the resume text.
    """

    pattern = r"(?:\+91[\s-]?)?[6-9]\d{9}"

    match = re.search(pattern, text)

    if match:
        return match.group()

    return None


def extract_skills(text):
    """
    Find recognized skills in the resume text.
    """

    text_lower = text.lower()

    found_skills = []

    for skill in SKILLS:

        if skill in text_lower:
            found_skills.append(skill)

    return found_skills


def extract_name(text):
    """
    Extract candidate name from the first few lines
    of the original resume text.
    """

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Check only the first 10 lines
    for line in lines[:10]:

        line_lower = line.lower()

        # Skip common headings and contact details
        skip_keywords = [
            "resume",
            "curriculum vitae",
            "cv",
            "email",
            "phone",
            "mobile",
            "linkedin",
            "github",
            "address",
            "@",
            "summary",
            "objective",
            "experience",
            "education",
            "skills",
        ]

        if any(keyword in line_lower for keyword in skip_keywords):
            continue

        # Remove unwanted characters
        clean_line = re.sub(
            r"[^A-Za-z\s]",
            "",
            line,
        ).strip()

        words = clean_line.split()

        # Candidate names usually have 2–4 words
        if 2 <= len(words) <= 4:

            if all(word.isalpha() for word in words):

                return clean_line.title()

    return "Unknown Candidate"


def extract_education(text):
    """
    Find education-related information in the resume.
    """

    education_keywords = [
        "b.tech",
        "btech",
        "b.e",
        "bachelor",
        "m.tech",
        "mtech",
        "master",
        "b.sc",
        "bsc",
        "m.sc",
        "msc",
        "mba",
        "phd",
        "computer science",
        "engineering",
    ]

    text_lower = text.lower()

    found_education = []

    for keyword in education_keywords:

        if keyword in text_lower:
            found_education.append(keyword)

    return found_education


def extract_experience(text):
    """
    Find years of experience mentioned in the resume.
    """

    pattern = r"\b(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\b"

    matches = re.findall(pattern, text.lower())

    return matches


def extract_information(text):
    """
    Extract all important candidate information.
    """

    information = {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text),
        "education": extract_education(text),
        "experience_years": extract_experience(text),
    }

    return information


if __name__ == "__main__":

    sample_text = """
Rahul Sharma

Email: rahul@gmail.com
Phone: +919876543210

Education:
B.Tech in Computer Science

Experience:
2 years of experience as a Python Developer

Skills:
Python
SQL
Pandas
Machine Learning
Git
"""

    result = extract_information(sample_text)

    print(result)
