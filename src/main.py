import os
import json

from extract_text import extract_text_from_pdf
from preprocess import preprocess_text
from information_extraction import extract_information
from job_description import process_job_description
from matching import calculate_similarity
from skill_matching import calculate_skill_match

from word2vec_matching import calculate_word2vec_similarity
from sbert_matching import calculate_sbert_similarity

# --------------------------------------------------
# PATHS
# --------------------------------------------------

resume_folder = "data/resumes"
job_description_path = "data/job_description/job_description.txt"
output_folder = "outputs"


# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


# -------------------------------------------------
# STEP 1: PROCESS JOB DESCRIPTION
# --------------------------------------------------

job_data = process_job_description(job_description_path)

job_cleaned_text = job_data["cleaned_text"]

print("\n" + "=" * 60)
print("JOB DESCRIPTION")
print("=" * 60)

print("\nRequired Skills:")
print(job_data["skills"])


# --------------------------------------------------
# STEP 2: PROCESS RESUMES
# --------------------------------------------------

all_results = []


for filename in os.listdir(resume_folder):

    if filename.lower().endswith(".pdf"):

        print("\n" + "=" * 60)
        print("PROCESSING:", filename)
        print("=" * 60)

        # PDF path
        pdf_path = os.path.join(resume_folder, filename)

        # Extract resume text
        raw_text = extract_text_from_pdf(pdf_path)

        # Preprocess resume
        cleaned_text = preprocess_text(raw_text)

        # Extract candidate information
        information = extract_information(cleaned_text)

        # Calculate TF-IDF similarity
        similarity_score = calculate_similarity(job_cleaned_text, cleaned_text)

        tfidf_percentage = round(similarity_score * 100, 2)

        # Calculate Word2Vec similarity
        word2vec_score = calculate_word2vec_similarity(
            job_cleaned_text,
            cleaned_text
        )

        word2vec_percentage = round(
            word2vec_score * 100,
            2
        )

        # Calculate SBERT similarity
        sbert_score = calculate_sbert_similarity(
            job_cleaned_text,
            cleaned_text
        )

        sbert_percentage = round(
            sbert_score * 100,
            2
        )

        # Calculate skill match
        skill_result = calculate_skill_match(job_data["skills"], information["skills"])

        skill_percentage = skill_result["skill_score"]
        matched_skills = skill_result["matched_skills"]

        # Calculate final score
        final_score = 0.4 * tfidf_percentage + 0.6 * skill_percentage

        final_score = round(final_score, 2)

        # Store result
        candidate_result = {
            "resume": filename,
            "name": information["name"],
            "email": information["email"],
            "phone": information["phone"],
            "skills": information["skills"],
            "matched_skills": matched_skills,
            "education": information["education"],
            "experience_years": information["experience_years"],
            "tfidf_score": tfidf_percentage,
            "word2vec_score": word2vec_percentage,
            "sbert_score": sbert_percentage,
            "skill_match_score": skill_percentage,
            "final_score": final_score,
        }

        all_results.append(candidate_result)

        # Display result
        print("Name:", information["name"])
        print("Email:", information["email"])
        print("Phone:", information["phone"])
        print("Skills:", information["skills"])
        print("Education:", information["education"])
        print("Experience:", information["experience_years"])
        print("Matched Skills:", matched_skills)
        print("TF-IDF Score:", tfidf_percentage, "%")
        print("Word2Vec Score:", word2vec_percentage, "%")
        print("SBERT Score:", sbert_percentage, "%")
        print("Skill Match Score:", skill_percentage, "%")
        print("Final Score:", final_score, "%")


# --------------------------------------------------
# STEP 3: RANK CANDIDATES
# --------------------------------------------------

all_results.sort(key=lambda candidate: candidate["final_score"], reverse=True)


# Add ranking
for rank, candidate in enumerate(all_results, start=1):

    candidate["rank"] = rank


# --------------------------------------------------
# STEP 4: SAVE RESULTS
# --------------------------------------------------

output_file = os.path.join(output_folder, "ranked_candidates.json")


with open(output_file, "w", encoding="utf-8") as file:

    json.dump(all_results, file, indent=4, ensure_ascii=False)


# --------------------------------------------------
# STEP 5: DISPLAY FINAL RANKING
# --------------------------------------------------

print("\n\n" + "=" * 60)
print("FINAL CANDIDATE RANKING")
print("=" * 60)

for candidate in all_results:

    print(
        f"Rank {candidate['rank']}: "
        f"{candidate['resume']} "
        f"| Final Score: {candidate['final_score']}%"
    )
