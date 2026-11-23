import os
import json
from io import BytesIO
from datetime import datetime
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit

load_dotenv()

DB_PATH = "data/embeddings"
COLLECTION_NAME = "raguru_lectures"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
MAX_DISTANCE = 0.65

st.set_page_config(
    page_title="RAGuru - AI Teaching Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @media (max-width: 768px) {
        .main .block-container { padding: 1rem; }
        h1 { font-size: 1.5rem !important; }
    }
    .main { background-color: #0e1117; }
    .stChatMessage { border-radius: 12px; padding: 10px; margin: 5px 0; }
    .stButton>button {
        background: linear-gradient(135deg, #1f538d, #2d7dd2);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 20px;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2d7dd2, #1f538d);
    }
    .quiz-card {
        background-color: #161b22;
        border-radius: 10px;
        padding: 16px;
        margin: 10px 0;
        border-left: 4px solid #2d7dd2;
    }
    [data-testid="stMetric"] {
        background-color: #161b22;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_resources():
    try:
        embedder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    except Exception:
        embedder = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return embedder, collection, groq_client


embedder, collection, groq_client = load_resources()

defaults = {
    "messages": [{
        "role": "assistant",
        "content": "Hi! I'm RAGuru, your AI study assistant. Type or record your doubt from the lectures in English or Hinglish."
    }],
    "quiz_score": 0,
    "total_questions": 0,
    "doubt_history": [],
    "topics_covered": [],
    "flashcards": [],
    "current_quiz": None,
    "quiz_submitted": False,
    "quiz_answers": {},
    "quiz_topic": "",
    "quiz_result": None,
    "last_audio_bytes": None
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


def format_timestamp(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def transcribe_voice(audio_bytes):
    transcription = groq_client.audio.transcriptions.create(
        file=("doubt.wav", audio_bytes),
        model="whisper-large-v3",
        response_format="text"
    )
    return transcription.strip()


def retrieve_context(query, top_k=4):
    query_vec = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_vec,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    valid_docs, valid_metas = [], []
    for doc, meta, dist in zip(docs, metas, dists):
        if dist <= MAX_DISTANCE:
            valid_docs.append(doc)
            valid_metas.append(meta)

    return valid_docs, valid_metas


def stream_tutor_response(question, context_chunks, chat_history):
    if not context_chunks:
        yield (
            "I checked the lecture notes, and this specific topic doesn't appear to be covered in the uploaded material. "
            "Try rephrasing your question or ask about a topic from your lectures!"
        )
        return

    context_text = "\n\n---\n\n".join(context_chunks)
    recent_chat = ""
    for msg in chat_history[-4:]:
        speaker = "Student" if msg["role"] == "user" else "RAGuru"
        recent_chat += f"{speaker}: {msg['content']}\n"

    prompt = f"""You are RAGuru, a supportive and clear AI teaching assistant.

Guidelines:
- Answer the student's question primarily using the Lecture Context provided below.
- Use a warm, encouraging, and easy-to-understand tone with simple examples.
- Reply in the same language style the student used (English or Hinglish).
- If the context only partially answers the question, explain what is in the lecture clearly and avoid making up facts not supported by the material.

Lecture Context:
{context_text}

Recent Conversation:
{recent_chat}

Student Question: {question}

Response:"""

    stream = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are RAGuru, a helpful AI teaching assistant grounded in lecture transcripts."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=700,
        temperature=0.4,
        stream=True
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def generate_quiz(topic, num_questions=5):
    docs, _ = retrieve_context(topic, top_k=5)
    if not docs:
        return None

    context = "\n\n".join(docs)
    prompt = f"""Create {num_questions} multiple-choice questions on "{topic}" strictly based on the lecture content below.

Lecture Content:
{context}

Return ONLY a valid JSON array in this exact format:
[
  {{
    "question": "Clear question text?",
    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
    "correct": "A",
    "explanation": "Brief explanation referencing the lecture"
  }}
]"""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "Output only valid JSON. No markdown fences or extra text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        temperature=0.3
    )

    try:
        raw = response.choices[0].message.content
        start, end = raw.find("["), raw.rfind("]") + 1
        return json.loads(raw[start:end])
    except Exception:
        return None


def generate_summary(topic):
    docs, _ = retrieve_context(topic, top_k=5)
    if not docs:
        return "No relevant lecture content found for this topic. Try searching a keyword covered in your lectures."

    context = "\n\n".join(docs)
    prompt = f"""Summarize the topic "{topic}" for a student using the lecture content below.

Content:
{context}

Structure:
- **Overview:** 1-2 sentences
- **Key Takeaways:** 3-5 bullet points
- **Practical Example:** 1 relatable example
Keep it concise (under 250 words) and clear."""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.4
    )
    return response.choices[0].message.content


def generate_flashcards(topic, num_cards=5):
    docs, _ = retrieve_context(topic, top_k=4)
    if not docs:
        return None

    context = "\n\n".join(docs)
    prompt = f"""Create {num_cards} study flashcards on "{topic}" from the lecture content below.

Content:
{context}

Return ONLY a valid JSON array:
[{{"front": "Concept or Question", "back": "Concise answer"}}]"""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "Output only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.3
    )

    try:
        raw = response.choices[0].message.content
        start, end = raw.find("["), raw.rfind("]") + 1
        return json.loads(raw[start:end])
    except Exception:
        return None


def build_pdf_bytes(title, content):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, f"RAGuru Notes: {title}")
    c.setFont("Helvetica", 9)
    c.drawString(50, height - 68, f"Generated on {datetime.now().strftime('%d %b %Y, %I:%M %p')}")
    c.line(50, height - 76, width - 50, height - 76)

    c.setFont("Helvetica", 11)
    y = height - 100

    safe_text = content.encode("latin-1", "replace").decode("latin-1")

    for paragraph in safe_text.split("\n"):
        wrapped_lines = simpleSplit(paragraph, "Helvetica", 11, width - 100) or [""]
        for line in wrapped_lines:
            if y < 60:
                c.showPage()
                c.setFont("Helvetica", 11)
                y = height - 50
            c.drawString(50, y, line)
            y -= 16

    c.save()
    buffer.seek(0)
    return buffer


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🎓 RAGuru")
    st.caption("AI Teaching Assistant")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "💬 Ask Doubt",
            "📝 Quiz Generator",
            "🃏 Flashcards",
            "📖 Topic Summary",
            "📊 Progress",
            "📚 Doubt History"
        ]
    )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Score", f"{st.session_state.quiz_score}")
    with col2:
        acc = int((st.session_state.quiz_score / st.session_state.total_questions) * 100) if st.session_state.total_questions else 0
        st.metric("Accuracy", f"{acc}%")

    st.divider()
    st.markdown(f"**Indexed Chunks:** {collection.count()}")
    st.markdown(f"**Doubts Asked:** {len(st.session_state.doubt_history)}")
    st.markdown(f"**Topics Practiced:** {len(st.session_state.topics_covered)}")
    st.divider()

    if st.button("Clear Chat"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Chat cleared! What would you like to study next?"
        }]
        st.session_state.last_audio_bytes = None
        st.rerun()


# ---------------- PAGES ----------------
if page == "💬 Ask Doubt":
    st.title("💬 Ask RAGuru")
    st.caption("Ask by typing or recording your voice — grounded in your lecture transcripts")
    st.divider()

    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    audio_data = st.audio_input("🎤 Record your doubt (English / Hinglish)")
    typed_q = st.chat_input("Ask a question from your lectures...")

    user_q = None
    if typed_q:
        user_q = typed_q
    elif audio_data is not None:
        audio_bytes = audio_data.getvalue()
        if st.session_state.last_audio_bytes != audio_bytes:
            st.session_state.last_audio_bytes = audio_bytes
            with st.spinner("Transcribing voice..."):
                user_q = transcribe_voice(audio_bytes)

    if user_q:
        with st.chat_message("user"):
            st.markdown(user_q)

        st.session_state.messages.append({"role": "user", "content": user_q})
        st.session_state.doubt_history.append({
            "question": user_q,
            "time": datetime.now().strftime("%d %b %Y, %I:%M %p")
        })

        with st.chat_message("assistant"):
            docs, metas = retrieve_context(user_q, top_k=3)
            reply = st.write_stream(
                stream_tutor_response(user_q, docs, st.session_state.messages)
            )

            if metas:
                with st.expander("📍 Referenced Lecture Timestamps"):
                    for i, m in enumerate(metas, start=1):
                        start_fmt = format_timestamp(m["start_time"])
                        end_fmt = format_timestamp(m["end_time"])
                        st.write(f"**{i}. {m['lecture']}** (`{start_fmt}` – `{end_fmt}`)")

            pdf_bytes = build_pdf_bytes("Doubt Note", f"Question:\n{user_q}\n\nAnswer:\n{reply}")
            st.download_button(
                label="📥 Download Note (PDF)",
                data=pdf_bytes,
                file_name=f"raguru_note_{len(st.session_state.messages)}.pdf",
                mime="application/pdf"
            )

        st.session_state.messages.append({"role": "assistant", "content": reply})


elif page == "📝 Quiz Generator":
    st.title("📝 Practice Quiz")
    st.caption("Test your understanding based on lecture content")
    st.divider()

    if st.session_state.current_quiz is None:
        col1, col2 = st.columns([2, 1])
        with col1:
            topic_input = st.text_input("Topic", placeholder="e.g., Variables, Loops, Functions...")
        with col2:
            num_q = st.slider("Questions", 3, 10, 5)

        if st.button("Generate Quiz"):
            if topic_input.strip():
                with st.spinner("Generating questions from lectures..."):
                    quiz_data = generate_quiz(topic_input.strip(), num_q)
                    if quiz_data:
                        st.session_state.current_quiz = quiz_data
                        st.session_state.quiz_topic = topic_input.strip()
                        st.session_state.quiz_answers = {}
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_result = None
                        st.rerun()
                    else:
                        st.error("Couldn't find enough lecture material on this topic. Try another keyword.")
            else:
                st.warning("Please enter a topic first.")

    elif not st.session_state.quiz_submitted:
        st.subheader(f"Topic: {st.session_state.quiz_topic}")

        for i, q in enumerate(st.session_state.current_quiz):
            st.markdown(f"<div class='quiz-card'><b>Q{i+1}. {q['question']}</b></div>", unsafe_allow_html=True)
            choice = st.radio(
                f"q_{i}",
                q["options"],
                key=f"radio_q_{i}",
                label_visibility="collapsed"
            )
            st.session_state.quiz_answers[i] = choice[0]

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Submit Answers"):
                score = 0
                breakdown = []
                for i, q in enumerate(st.session_state.current_quiz):
                    picked = st.session_state.quiz_answers.get(i, "")
                    is_right = picked == q["correct"]
                    if is_right:
                        score += 1
                    breakdown.append({
                        "num": i + 1,
                        "is_right": is_right,
                        "correct": q["correct"],
                        "explanation": q["explanation"]
                    })

                total = len(st.session_state.current_quiz)
                st.session_state.quiz_score += score
                st.session_state.total_questions += total
                st.session_state.quiz_submitted = True
                st.session_state.quiz_result = {
                    "score": score,
                    "total": total,
                    "pct": int((score / total) * 100),
                    "breakdown": breakdown
                }
                if st.session_state.quiz_topic not in st.session_state.topics_covered:
                    st.session_state.topics_covered.append(st.session_state.quiz_topic)
                st.rerun()

        with col2:
            if st.button("Cancel"):
                st.session_state.current_quiz = None
                st.rerun()

    else:
        res = st.session_state.quiz_result
        st.subheader(f"Result: {res['score']}/{res['total']} ({res['pct']}%)")

        for item in res["breakdown"]:
            if item["is_right"]:
                st.success(f"**Q{item['num']}: Correct** — {item['explanation']}")
            else:
                st.error(f"**Q{item['num']}: Incorrect** (Correct: {item['correct']}) — {item['explanation']}")

        if st.button("Take Another Quiz"):
            st.session_state.current_quiz = None
            st.session_state.quiz_submitted = False
            st.rerun()


elif page == "🃏 Flashcards":
    st.title("🃏 Study Flashcards")
    st.caption("Active recall cards built from your lectures")
    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        fc_topic = st.text_input("Topic", placeholder="e.g., Data Types, Recursion...")
    with col2:
        num_fc = st.slider("Cards", 3, 10, 5)

    if st.button("Create Flashcards"):
        if fc_topic.strip():
            with st.spinner("Building flashcards..."):
                cards = generate_flashcards(fc_topic.strip(), num_fc)
                if cards:
                    st.session_state.flashcards = cards
                else:
                    st.error("No matching lecture content found for this topic.")
        else:
            st.warning("Enter a topic first.")

    if st.session_state.flashcards:
        st.divider()
        for i, card in enumerate(st.session_state.flashcards, start=1):
            with st.expander(f"Card {i}: {card['front']}"):
                st.write(card["back"])

        fc_text = "\n\n".join([f"Card {i}: {c['front']}\nAnswer: {c['back']}" for i, c in enumerate(st.session_state.flashcards, 1)])
        st.download_button(
            "📥 Download Flashcards (PDF)",
            data=build_pdf_bytes("Flashcards", fc_text),
            file_name="raguru_flashcards.pdf",
            mime="application/pdf"
        )


elif page == "📖 Topic Summary":
    st.title("📖 Lecture Summary")
    st.caption("Quick revision notes generated from lecture segments")
    st.divider()

    sum_topic = st.text_input("Topic to summarize", placeholder="e.g., Introduction to Python...")
    if st.button("Generate Summary"):
        if sum_topic.strip():
            with st.spinner("Summarizing..."):
                summary_text = generate_summary(sum_topic.strip())
                st.markdown(summary_text)
                st.download_button(
                    "📥 Download Summary (PDF)",
                    data=build_pdf_bytes(sum_topic.strip(), summary_text),
                    file_name=f"summary_{sum_topic.strip().replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("Enter a topic first.")


elif page == "📊 Progress":
    st.title("📊 Learning Progress")
    st.divider()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Points", st.session_state.quiz_score)
    c2.metric("Questions Attempted", st.session_state.total_questions)
    acc = int((st.session_state.quiz_score / st.session_state.total_questions) * 100) if st.session_state.total_questions else 0
    c3.metric("Overall Accuracy", f"{acc}%")

    if st.session_state.topics_covered:
        st.subheader("Practiced Topics")
        for t in st.session_state.topics_covered:
            st.write(f"- {t}")

    if st.session_state.total_questions > 0 and st.button("Reset Stats"):
        st.session_state.quiz_score = 0
        st.session_state.total_questions = 0
        st.session_state.topics_covered = []
        st.rerun()


elif page == "📚 Doubt History":
    st.title("📚 Doubt History")
    st.divider()

    if st.session_state.doubt_history:
        for i, item in enumerate(reversed(st.session_state.doubt_history), 1):
            st.markdown(f"**{i}. {item['question']}**  \n*Asked: {item['time']}*")
            st.divider()

        hist_text = "\n\n".join([f"Q{i}: {d['question']} ({d['time']})" for i, d in enumerate(st.session_state.doubt_history, 1)])
        st.download_button(
            "📥 Export History (PDF)",
            data=build_pdf_bytes("Doubt History", hist_text),
            file_name="doubt_history.pdf",
            mime="application/pdf"
        )
    else:
        st.info("No questions asked yet.")