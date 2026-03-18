import json
import os
import chromadb
from sentence_transformers import SentenceTransformer

# Create embeddings output folder
os.makedirs("data/embeddings", exist_ok=True)

# Load sentence transformer model for embeddings
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready!")

# Initialize ChromaDB
print("\nInitializing ChromaDB...")
client = chromadb.PersistentClient(path="data/embeddings")
collection = client.get_or_create_collection(name="raguru_lectures")
print("ChromaDB ready!")

# Load all transcript JSON files
transcript_folder = "data/transcripts"
transcript_files = [f for f in os.listdir(transcript_folder)
                    if f.endswith(".json")]

print(f"\nTotal transcripts found: {len(transcript_files)}")
print("=" * 50)

all_chunks = []
all_ids = []
all_metadata = []

# Process each transcript file
for transcript_file in transcript_files:
    transcript_path = f"{transcript_folder}/{transcript_file}"
    lecture_name = transcript_file.replace(".json", "")

    print(f"\nProcessing: {transcript_file}")

    # Load transcript JSON
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    # Split transcript into chunks using segments
    segments = transcript_data.get("segments", [])

    # Group every 5 segments into one chunk
    chunk_size = 5
    for i in range(0, len(segments), chunk_size):
        chunk_segments = segments[i:i + chunk_size]

        # Combine text from segments
        chunk_text = " ".join([seg["text"] for seg in chunk_segments])
        start_time = chunk_segments[0]["start"]
        end_time = chunk_segments[-1]["end"]

        chunk_id = f"{lecture_name}_chunk_{i}"

        all_chunks.append(chunk_text)
        all_ids.append(chunk_id)
        all_metadata.append({
            "lecture": lecture_name,
            "start_time": start_time,
            "end_time": end_time,
            "source": transcript_file
        })

    print(f"Total chunks created: {len(segments) // chunk_size + 1}")

print(f"\nTotal chunks to embed: {len(all_chunks)}")
print("Creating embeddings... (this may take a few minutes)")
embeddings = model.encode(all_chunks, show_progress_bar=True)

# Store in ChromaDB
print("\nStoring embeddings in ChromaDB...")
collection.add(
    documents=all_chunks,
    embeddings=embeddings.tolist(),
    ids=all_ids,
    metadatas=all_metadata
)

print("\n" + "=" * 50)
print("All embeddings stored successfully!")
print(f"Total chunks in ChromaDB: {collection.count()}")
print("RAGuru knowledge base is ready!")
print("=" * 50)