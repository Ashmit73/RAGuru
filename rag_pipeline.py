import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
print("Loading RAGuru...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="data/embeddings")
collection = client.get_collection(name="raguru_lectures")

print(f"Knowledge base loaded! Total chunks: {collection.count()}")
print("=" * 50)


def search_knowledge_base(question, top_k=3):
    """Search relevant chunks from ChromaDB"""

    # Convert question to embedding
    question_embedding = model.encode([question]).tolist()

    # Search in ChromaDB
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )

    return results


def ask_raguru(question):
    """Main RAGuru function — ask anything!"""

    print(f"\nQuestion: {question}")
    print("-" * 50)

    # Find relevant chunks
    results = search_knowledge_base(question)

    # Show top matching chunks
    print("Top matching content from lectures:\n")
    for i, (doc, metadata) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0]
    )):
        print(f"Match {i + 1}:")
        print(f"  Lecture: {metadata['lecture']}")
        print(f"  Time: {metadata['start_time']}s - {metadata['end_time']}s")
        print(f"  Content: {doc[:200]}...")
        print()


print("\nRAGuru is ready! Testing with sample questions...")
print("=" * 50)

ask_raguru("What is Python?")
ask_raguru("How to use pip?")
ask_raguru("What are modules in Python?")