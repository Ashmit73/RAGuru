import os
import json
import shutil
import chromadb
from sentence_transformers import SentenceTransformer

TRANSCRIPT_DIR = "data/transcripts"
DB_PATH = "data/embeddings"
COLLECTION_NAME = "raguru_lectures"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Sliding window parameters (6 segments per chunk, 2 segments overlap)
WINDOW_SIZE = 6
STEP_SIZE = 4


def build_chunks(transcript_data, filename):
    segments = transcript_data.get("segments", [])
    lecture_name = transcript_data.get("lecture_name", os.path.splitext(filename)[0])

    chunks = []
    for i in range(0, len(segments), STEP_SIZE):
        window = segments[i: i + WINDOW_SIZE]
        if not window:
            continue

        combined_text = " ".join(seg["text"] for seg in window).strip()
        if len(combined_text) < 20:  # skip tiny noise segments
            continue

        chunks.append({
            "id": f"{lecture_name}_chunk_{i}",
            "text": combined_text,
            "metadata": {
                "lecture": lecture_name,
                "start_time": window[0]["start"],
                "end_time": window[-1]["end"],
                "source": filename
            }
        })

        # Stop if we've reached the end of segments
        if i + WINDOW_SIZE >= len(segments):
            break

    return chunks


def main():
    if not os.path.exists(TRANSCRIPT_DIR):
        print(f"Folder '{TRANSCRIPT_DIR}' not found. Run transcribe.py first.")
        return

    json_files = [f for f in os.listdir(TRANSCRIPT_DIR) if f.endswith(".json")]
    if not json_files:
        print("No transcript JSON files found.")
        return

    all_chunks = []
    for file in json_files:
        path = os.path.join(TRANSCRIPT_DIR, file)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_chunks = build_chunks(data, file)
        all_chunks.extend(file_chunks)
        print(f"{file}: {len(file_chunks)} chunks prepared")

    if not all_chunks:
        print("No valid text chunks generated.")
        return

    print(f"\nTotal chunks ready: {len(all_chunks)}")
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in all_chunks]
    ids = [c["id"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # Reset old database folder cleanly to avoid stale/mismatched vectors
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # cosine distance works best for text similarity
    )

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"\nDone! Stored {collection.count()} chunks in '{DB_PATH}'.")


if __name__ == "__main__":
    main()