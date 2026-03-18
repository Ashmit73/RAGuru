from datasets import load_dataset
import os

os.makedirs("data/rag", exist_ok=True)

print("=" * 50)
print("RAGuru — Dataset Loading Started!")
print("=" * 50)

print("\nLoading English QA dataset...")

try:
    english_data = load_dataset("squad", split="train[:200]")
    english_data.save_to_disk("data/rag/english_qa")
    print(f"English QA dataset loaded successfully!")
    print(f"Total examples: {len(english_data)}")
    print(f"Sample question: {english_data[0]['question']}")
    print(f"Sample answer: {english_data[0]['answers']['text'][0]}")
except Exception as e:
    print(f"Error loading English dataset: {e}")


print("\nLoading Hindi dataset...")

try:
    hindi_data = load_dataset(
        "Helsinki-NLP/opus-100",
        "en-hi",
        split="train[:200]"
    )
    hindi_data.save_to_disk("data/rag/hindi_qa")
    print(f"Hindi dataset loaded successfully!")
    print(f"Total examples: {len(hindi_data)}")
    print(f"Sample Hindi: {hindi_data[0]['translation']['hi']}")
    print(f"Sample English: {hindi_data[0]['translation']['en']}")
except Exception as e:
    print(f"Error loading Hindi dataset: {e}")

print("\n" + "=" * 50)
print("All RAGuru datasets saved successfully!")
print("Saved locations:")
print("  data/rag/english_qa  --> English QA dataset")
print("  data/rag/hindi_qa    --> Hindi dataset")
print("=" * 50)