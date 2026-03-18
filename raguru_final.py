import chromadb
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load .env file
load_dotenv()

print("Loading RAGuru Final Pipeline...")

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="data/embeddings")
collection = client.get_collection(name="raguru_lectures")

print(f"Knowledge base ready! Total chunks: {collection.count()}")

# Groq client setup - API key from .env file
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_relevant_chunks(question, top_k=3):
    """Find most relevant chunks from ChromaDB"""
    question_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )
    return results

def generate_answer(question, context_chunks):
    """Generate answer using Groq LLaMA"""

    context = "\n\n".join(context_chunks)

    prompt = f"""You are RAGuru, an AI teaching assistant.
Use the following lecture content to answer the student's question.
If the answer is not in the context, say "This topic was not covered in the lectures."

Lecture Context:
{context}

Student Question: {question}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": "You are RAGuru, a helpful AI teaching assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=500
    )

    return response.choices[0].message.content

def ask_raguru(question):
    """Complete RAGuru pipeline"""
    print(f"\nQuestion: {question}")
    print("-" * 50)

    results = get_relevant_chunks(question)
    context_chunks = results['documents'][0]
    metadatas = results['metadatas'][0]

    print("Sources found:")
    for i, meta in enumerate(metadatas):
        print(f"  {i+1}. {meta['lecture']} at {meta['start_time']}s")

    print("\nGenerating answer...")
    answer = generate_answer(question, context_chunks)

    print(f"\nRAGuru Answer:\n{answer}")
    print("=" * 50)


print("\n" + "=" * 50)
print("RAGuru is ready! Ask anything about your lectures.")
print("Type 'exit' to quit.")
print("=" * 50)

while True:
    user_question = input("\nAsk RAGuru: ")

    if user_question.lower() == "exit":
        print("Goodbye! Keep learning!")
        break

    if user_question.strip() == "":
        print("Please enter a question!")
        continue

    ask_raguru(user_question)