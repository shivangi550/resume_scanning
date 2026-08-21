import json
from pathlib import Path

# -----------------------------------------
# LOAD SCREENING RESULTS
# -----------------------------------------

result_file = Path("outputs/result.jsonl")

if not result_file.exists():

    print("result.jsonl not found.")
    print("Run the resume screening first.")
    exit()


results = []

with open(result_file, "r", encoding="utf-8") as file:

    for line in file:

        if line.strip():

            results.append(json.loads(line))


# -----------------------------------------
# SORT CANDIDATES
# -----------------------------------------

results.sort(key=lambda x: x.get("final_score", 0), reverse=True)


# -----------------------------------------
# DISPLAY RANKING
# -----------------------------------------

print("=" * 60)
print("RESUME SCREENING EVALUATION")
print("=" * 60)

for index, candidate in enumerate(results, start=1):

    name = candidate.get("name", "Unknown")

    score = candidate.get("final_score", 0)

    print(f"{index}. {name} " f"— {score}%")


# -----------------------------------------
# BASIC STATISTICS
# -----------------------------------------

if results:

    scores = [candidate.get("final_score", 0) for candidate in results]

    average_score = sum(scores) / len(scores)

    highest_score = max(scores)

    lowest_score = min(scores)

    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)

    print(f"Number of candidates: {len(results)}")

    print(f"Average score: {average_score:.2f}%")

    print(f"Highest score: {highest_score}%")

    print(f"Lowest score: {lowest_score}%")

else:

    print("No candidates found.")
