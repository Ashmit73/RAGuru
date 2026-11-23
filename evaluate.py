import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "data/embeddings"
COLLECTION_NAME = "raguru_lectures"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
MAX_DISTANCE = 0.65


def load_db():
    try:
        model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    except Exception:
        model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    return model, collection


def evaluate_retrieval(test_cases):
    model, collection = load_db()

    total_in_scope = 0
    hit_at_1 = 0
    hit_at_3 = 0
    total_out_of_scope = 0
    rejected_correctly = 0

    print("\nRunning RAGuru Retrieval Evaluation...\n" + "-" * 62)

    for case in test_cases:
        query = case["question"]
        expected_lecture = case.get("expected_lecture")

        q_vec = model.encode([query]).tolist()
        results = collection.query(
            query_embeddings=q_vec,
            n_results=3,
            include=["metadatas", "distances"]
        )

        metas = results["metadatas"][0]
        dists = results["distances"][0]

        valid_lectures = [
            m["lecture"] for m, d in zip(metas, dists) if d <= MAX_DISTANCE
        ]

        if expected_lecture is None:
            total_out_of_scope += 1
            if not valid_lectures:
                rejected_correctly += 1
                status = "PASS (Rejected)"
            else:
                status = f"FAIL (Matched {valid_lectures[0]}, dist={dists[0]:.2f})"
            print(f"[OOS]  {query[:44]:<44} -> {status}")
        else:
            total_in_scope += 1
            is_h1 = len(valid_lectures) > 0 and expected_lecture == valid_lectures[0]
            is_h3 = expected_lecture in valid_lectures

            if is_h1:
                hit_at_1 += 1
            if is_h3:
                hit_at_3 += 1

            status = "HIT@1" if is_h1 else ("HIT@3" if is_h3 else f"MISS (got {valid_lectures})")
            print(f"[TEST] {query[:44]:<44} -> {status}")

    print("-" * 62)
    if total_in_scope > 0:
        print(f"In-Scope Queries   : {total_in_scope}")
        print(f"Top-1 Accuracy     : {(hit_at_1 / total_in_scope) * 100:.1f}% ({hit_at_1}/{total_in_scope})")
        print(f"Top-3 Hit Rate     : {(hit_at_3 / total_in_scope) * 100:.1f}% ({hit_at_3}/{total_in_scope})")
    if total_out_of_scope > 0:
        print(f"Out-of-Scope Guard : {(rejected_correctly / total_out_of_scope) * 100:.1f}% ({rejected_correctly}/{total_out_of_scope})")
    print("-" * 62 + "\n")


if __name__ == "__main__":
    EVAL_SET = [
        {"question": "100 days of code Python course mein kya sikhaya jayega?", "expected_lecture": "lecture1"},
        {"question": "What happened in the instructor's 2nd year of college?", "expected_lecture": "lecture2"},
        {"question": "Where is Python used in real world applications?", "expected_lecture": "lecture2"},
        {"question": "Python mein modules aur pip kya hote hain?", "expected_lecture": "lecture3"},
        {"question": "Difference between built-in and external modules", "expected_lecture": "lecture3"},
        {"question": "How do we install external packages using pip?", "expected_lecture": "lecture3"},
        {"question": "Python mein apna pehla program kaise likhte hain?", "expected_lecture": "lecture4"},
        {"question": "How does the print function work in our first program?", "expected_lecture": "lecture4"},
        {"question": "What are comments in Python and why are they used?", "expected_lecture": "lecture5"},
        {"question": "Python mein escape sequence characters jaise backslash n kya karte hain?", "expected_lecture": "lecture5"},
        {"question": "How do sep and end parameters work in print statement?", "expected_lecture": "lecture5"},
        {"question": "What is the capital of France?", "expected_lecture": None},
        {"question": "Explain Django REST framework authentication", "expected_lecture": None},
        {"question": "How to make paneer butter masala?", "expected_lecture": None},
    ]

    evaluate_retrieval(EVAL_SET)
