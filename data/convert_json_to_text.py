import json
from pathlib import Path

input_file = Path("./data/Finance_FAQ_Extended.json")
output_file = Path("./data/finance_faq.txt")

if not input_file.exists():
    print(f"Input file not found: {input_file}")
    exit(1)

with open(input_file, "r", encoding="utf-8") as f:
    try:
        data = json.load(f)
    except Exception as e:
        print(f"Failed to load JSON: {e}")
        exit(1)

with open(output_file, "w", encoding="utf-8") as out:
    count = 0
    for item in data:
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        if question and answer:
            out.write(f"Q: {question}\nA: {answer}\n\n")
            count += 1

print(f"Converted {count} question-answer pairs to {output_file}")
