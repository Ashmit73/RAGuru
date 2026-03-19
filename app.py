import streamlit as st
import chromadb
import os
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

load_dotenv()


st.set_page_config(
    page_title="RAGuru - AI Teaching Assistant",
    page_icon="🧠",
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
    .stChatMessage { border-radius: 15px; padding: 10px; margin: 5px 0; }
    .stButton>button {
        background: linear-gradient(135deg, #1f538d, #2d7dd2);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2d7dd2, #1f538d);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(29,83,141,0.4);
    }
    .quiz-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #2d7dd2;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #2d7dd2;
    }
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="data/embeddings")
    collection = client.get_collection(name="raguru_lectures")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return embedding_model, collection, groq_client


embedding_model, collection, groq_client = load_models()


if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hey! I am RAGuru — your personal teacher! Ask me anything from the lectures. Any language works — English, Hindi, whatever!"
    }]

if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0

if "total_questions" not in st.session_state:
    st.session_state.total_questions = 0

if "doubt_history" not in st.session_state:
    st.session_state.doubt_history = []

if "topics_covered" not in st.session_state:
    st.session_state.topics_covered = []

if "flashcards" not in st.session_state:
    st.session_state.flashcards = []

if "current_quiz" not in st.session_state:
    st.session_state.current_quiz = None

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}

if "quiz_topic" not in st.session_state:
    st.session_state.quiz_topic = ""

if "quiz_result" not in st.session_state:
    st.session_state.quiz_result = None



def get_relevant_chunks(question, top_k=3):
    question_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )
    return results


def humanized_answer(question, context_chunks, chat_history):
    context = "\n\n".join(context_chunks)
    history_text = ""
    for msg in chat_history[-4:]:
        role = "Student" if msg["role"] == "user" else "RAGuru"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""You are RAGuru — a real human teacher and mentor, NOT an AI.

Your personality:
- Friendly, experienced teacher who genuinely cares
- Natural conversational tone — warm and approachable
- Say things like "Look, it is quite simple", "Think of it this way", "Great question!"
- Real world examples students can relate to
- Encouraging — "You are getting it!", "Almost there!"
- NEVER say you are an AI, chatbot or assistant
- NEVER start with "Certainly!" or "Of course!" or "As an AI..."
- Answer in WHATEVER language the student uses
- Keep answers clear and focused

Rules:
1. Answer from lecture context if available
2. General study questions — use your teaching knowledge
3. Off-topic — warmly say "That is outside our course right now!"
4. Always make the student feel smart and capable

Lecture Content:
{context}

Previous Chat:
{history_text}

Student: {question}

Your response as a real human teacher:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are RAGuru, a real human teacher. Never reveal you are an AI. Be natural and warm."
            },
            {"role": "user", "content": prompt}
        ],
        max_tokens=700,
        temperature=0.8
    )
    return response.choices[0].message.content


def generate_quiz(topic, num_questions=5):
    results = get_relevant_chunks(topic)
    context = "\n\n".join(results['documents'][0])

    prompt = f"""Create {num_questions} multiple choice questions about: {topic}

Based on:
{context}

Return ONLY this JSON — nothing else:
[
  {{
    "question": "Question here?",
    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
    "correct": "A",
    "explanation": "Why this is correct"
  }}
]"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Return only valid JSON array."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        temperature=0.5
    )

    try:
        text = response.choices[0].message.content
        start = text.find('[')
        end = text.rfind(']') + 1
        return json.loads(text[start:end])
    except:
        return None


def generate_summary(topic):
    results = get_relevant_chunks(topic, top_k=5)
    context = "\n\n".join(results['documents'][0])

    prompt = f"""Create a clear student-friendly summary of: {topic}

Content: {context}

Format:
- One sentence overview
- Key points as bullets
- One real world example
- Why this matters
Under 300 words. Simple language."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content


def generate_flashcards(topic, num_cards=5):
    results = get_relevant_chunks(topic)
    context = "\n\n".join(results['documents'][0])

    prompt = f"""Create {num_cards} flashcards for: {topic}
Content: {context}
Return ONLY JSON:
[{{"front": "Question", "back": "Answer"}}]"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Return only valid JSON array."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )

    try:
        text = response.choices[0].message.content
        start = text.find('[')
        end = text.rfind(']') + 1
        return json.loads(text[start:end])
    except:
        return None


def save_as_pdf(content, filename):
    os.makedirs("data/notes", exist_ok=True)
    pdf_path = f"data/notes/{filename}.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 770, "RAGuru Study Notes")
    c.setFont("Helvetica", 10)
    c.drawString(50, 750, f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}")
    c.line(50, 740, 550, 740)
    c.setFont("Helvetica", 11)
    y = 720
    for line in content.split('\n'):
        if y < 60:
            c.showPage()
            c.setFont("Helvetica", 11)
            y = 750
        c.drawString(50, y, line[:85])
        y -= 18
    c.save()
    return pdf_path



with st.sidebar:
    st.markdown("## 🧠 RAGuru")
    st.markdown("*Your Personal AI Teacher*")
    st.markdown("---")



    page = st.radio(
        "Navigate To",
        [
            "💬 Chat with RAGuru",
            "📝 Quiz Generator",
            "🃏 Flashcards",
            "📊 My Progress",
            "📖 Summary Generator",
            "📚 Doubt History"
        ]
    )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Score", f"{st.session_state.quiz_score} pts")
    with col2:
        sidebar_accuracy = 0
        if st.session_state.total_questions > 0:
            sidebar_accuracy = int(
                (st.session_state.quiz_score / st.session_state.total_questions) * 100
            )
        st.metric("Accuracy", f"{sidebar_accuracy}%")

    st.markdown("---")
    st.markdown(f"📚 **Knowledge Base:** {collection.count()} chunks")
    st.markdown(f"🎯 **Questions Attempted:** {st.session_state.total_questions}")
    st.markdown(f"💡 **Doubts Asked:** {len(st.session_state.doubt_history)}")
    st.markdown(f"📖 **Topics Covered:** {len(st.session_state.topics_covered)}")
    st.markdown("---")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hey! I am RAGuru — your personal teacher! Ask me anything!"
        }]
        st.rerun()

if page == "💬 Chat with RAGuru":
    st.title("💬 Chat with RAGuru")
    st.caption("Ask anything — any language works!")
    st.divider()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("Type your question here..."):
        with st.chat_message("user"):
            st.markdown(user_question)

        st.session_state.messages.append({"role": "user", "content": user_question})
        st.session_state.doubt_history.append({
            "question": user_question,
            "time": datetime.now().strftime("%d %b %Y, %I:%M %p")
        })

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                results = get_relevant_chunks(user_question)
                context_chunks = results['documents'][0]
                metadatas = results['metadatas'][0]

                answer = humanized_answer(
                    user_question,
                    context_chunks,
                    st.session_state.messages
                )

                st.markdown(answer)

                if st.button("📥 Save as PDF", key=f"pdf_{len(st.session_state.messages)}"):
                    path = save_as_pdf(
                        f"Q: {user_question}\n\nA: {answer}",
                        f"answer_{len(st.session_state.messages)}"
                    )
                    st.success("Saved!")

                with st.expander("📍 Lecture Sources"):
                    for i, meta in enumerate(metadatas):
                        st.write(f"**{i + 1}.** {meta['lecture']} at {meta['start_time']}s")

        st.session_state.messages.append({"role": "assistant", "content": answer})


elif page == "📝 Quiz Generator":
    st.title("📝 Quiz Generator")
    st.caption("Test your knowledge from the lectures!")
    st.divider()

    if st.session_state.current_quiz is None:
        col1, col2 = st.columns(2)
        with col1:
            quiz_topic = st.text_input(
                "Enter Topic",
                placeholder="e.g., Python basics, Variables, Loops..."
            )
        with col2:
            num_q = st.slider("Number of Questions", 3, 10, 5)

        if st.button("🎯 Generate Quiz!"):
            if quiz_topic:
                with st.spinner("Creating your quiz..."):
                    questions = generate_quiz(quiz_topic, num_q)
                    if questions:
                        st.session_state.current_quiz = questions
                        st.session_state.quiz_topic = quiz_topic
                        st.session_state.quiz_answers = {}
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_result = None
                        st.rerun()
                    else:
                        st.error("Could not generate quiz — try a different topic!")
            else:
                st.warning("Please enter a topic first!")

    elif not st.session_state.quiz_submitted:
        st.markdown(f"### Topic: {st.session_state.quiz_topic}")
        st.markdown(f"*{len(st.session_state.current_quiz)} questions — answer all below*")
        st.markdown("---")

        for i, q in enumerate(st.session_state.current_quiz):
            st.markdown(f"""
<div class="quiz-card">
<b>Q{i + 1}. {q['question']}</b>
</div>
""", unsafe_allow_html=True)
            selected = st.radio(
                f"Q{i + 1}",
                q['options'],
                key=f"quiz_q_{i}",
                label_visibility="collapsed"
            )
            st.session_state.quiz_answers[i] = selected[0]

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Submit Quiz"):
                score = 0
                results_list = []

                for i, q in enumerate(st.session_state.current_quiz):
                    user_ans = st.session_state.quiz_answers.get(i, "")
                    correct = q['correct']
                    is_correct = user_ans == correct
                    if is_correct:
                        score += 1
                    results_list.append({
                        "question_num": i + 1,
                        "correct": is_correct,
                        "user_ans": user_ans,
                        "correct_ans": correct,
                        "explanation": q['explanation']
                    })

                total = len(st.session_state.current_quiz)
                percentage = int((score / total) * 100)

                st.session_state.quiz_score += score
                st.session_state.total_questions += total
                st.session_state.quiz_submitted = True
                st.session_state.quiz_result = {
                    "score": score,
                    "total": total,
                    "percentage": percentage,
                    "results": results_list
                }

                topic = st.session_state.quiz_topic
                if topic not in st.session_state.topics_covered:
                    st.session_state.topics_covered.append(topic)

                st.rerun()

        with col2:
            if st.button("❌ Cancel Quiz"):
                st.session_state.current_quiz = None
                st.session_state.quiz_submitted = False
                st.session_state.quiz_result = None
                st.rerun()

    elif st.session_state.quiz_submitted and st.session_state.quiz_result:
        result = st.session_state.quiz_result
        score = result['score']
        total = result['total']
        percentage = result['percentage']

        st.markdown("### Your Results")
        st.markdown("---")

        for r in result['results']:
            if r['correct']:
                st.success(f"**Q{r['question_num']}: Correct! ✅** — {r['explanation']}")
            else:
                st.error(f"**Q{r['question_num']}: Wrong ❌** — Correct: {r['correct_ans']}) — {r['explanation']}")

        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Your Score", f"{score}/{total}")
        with col2:
            st.metric("Percentage", f"{percentage}%")
        with col3:
            overall_acc = 0
            if st.session_state.total_questions > 0:
                overall_acc = int(
                    (st.session_state.quiz_score / st.session_state.total_questions) * 100
                )
            st.metric("Overall Accuracy", f"{overall_acc}%")

        st.markdown("---")

        if percentage >= 80:
            st.balloons()
            st.success(f"🌟 Excellent! {score}/{total} — {percentage}%! You really know this topic!")
        elif percentage >= 50:
            st.warning(f"👍 Good effort! {score}/{total} — {percentage}%. Keep practicing!")
        else:
            st.error(f"💪 {score}/{total} — {percentage}%. Review the lecture and try again!")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Take Another Quiz"):
                st.session_state.current_quiz = None
                st.session_state.quiz_submitted = False
                st.session_state.quiz_result = None
                st.rerun()
        with col2:
            if st.button("📊 View My Progress"):
                st.session_state.current_quiz = None
                st.session_state.quiz_submitted = False
                st.session_state.quiz_result = None
                st.rerun()

elif page == "🃏 Flashcards":
    st.title("🃏 Flashcards")
    st.caption("Quick revision made easy!")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        flash_topic = st.text_input("Enter Topic", placeholder="e.g., Python functions, OOP...")
    with col2:
        num_cards = st.slider("Number of Cards", 3, 10, 5)

    if st.button("🃏 Create Flashcards"):
        if flash_topic:
            with st.spinner("Creating flashcards..."):
                cards = generate_flashcards(flash_topic, num_cards)
                if cards:
                    st.session_state.flashcards = cards
                    st.success(f"{len(cards)} flashcards ready!")
                else:
                    st.error("Could not create flashcards — try again!")
        else:
            st.warning("Please enter a topic first!")

    if st.session_state.flashcards:
        st.markdown("---")
        st.markdown("### Click each card to see the answer:")

        for i, card in enumerate(st.session_state.flashcards):
            with st.expander(f"📌 Card {i + 1}: {card['front']}"):
                st.info(f"**Answer:** {card['back']}")

        if st.button("📥 Download as PDF"):
            content = "RAGuru Flashcards\n\n"
            for i, card in enumerate(st.session_state.flashcards):
                content += f"Card {i + 1}\nQ: {card['front']}\nA: {card['back']}\n\n"
            path = save_as_pdf(content, "flashcards")
            st.success("Saved!")


elif page == "📊 My Progress":
    st.title("📊 My Progress")
    st.caption("Track your learning journey!")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Score", st.session_state.quiz_score)
    with col2:
        st.metric("Questions Done", st.session_state.total_questions)
    with col3:
        acc = 0
        if st.session_state.total_questions > 0:
            acc = int((st.session_state.quiz_score / st.session_state.total_questions) * 100)
        st.metric("Accuracy", f"{acc}%")
    with col4:
        st.metric("Topics Covered", len(st.session_state.topics_covered))

    st.markdown("---")

    if st.session_state.topics_covered:
        st.markdown("### Topics Covered")
        for topic in st.session_state.topics_covered:
            st.success(f"✅ {topic}")
    else:
        st.info("No topics yet — take a quiz!")

    st.markdown("---")

    if acc >= 80:
        st.success("🌟 Excellent! You are mastering the material!")
    elif acc >= 60:
        st.warning("👍 Good progress! Keep practicing!")
    elif acc > 0:
        st.error("💪 Keep going — review lectures and try again!")
    else:
        st.info("Take a quiz to start tracking your progress!")

    st.markdown("---")
    if st.session_state.total_questions > 0:
        if st.button("🔄 Reset Progress"):
            st.session_state.quiz_score = 0
            st.session_state.total_questions = 0
            st.session_state.topics_covered = []
            st.success("Progress reset!")
            st.rerun()


elif page == "📖 Summary Generator":
    st.title("📖 Summary Generator")
    st.caption("Get a clear summary of any topic!")
    st.divider()

    summary_topic = st.text_input("Enter Topic", placeholder="e.g., Python, Variables...")

    if st.button("📖 Generate Summary"):
        if summary_topic:
            with st.spinner("Creating summary..."):
                summary = generate_summary(summary_topic)
                st.markdown("### Summary")
                st.markdown(summary)
                st.markdown("---")

                if st.button("📥 Save as PDF"):
                    path = save_as_pdf(f"Topic: {summary_topic}\n\n{summary}", f"summary_{summary_topic}")
                    st.success("Saved!")
        else:
            st.warning("Please enter a topic first!")


elif page == "📚 Doubt History":
    st.title("📚 Doubt History")
    st.caption("All your previous questions!")
    st.divider()

    if st.session_state.doubt_history:
        st.markdown(f"**Total questions: {len(st.session_state.doubt_history)}**")
        st.markdown("---")

        for i, doubt in enumerate(reversed(st.session_state.doubt_history)):
            with st.expander(f"Q{len(st.session_state.doubt_history) - i}: {doubt['question'][:60]}..."):
                st.write(f"**Question:** {doubt['question']}")
                st.caption(f"Asked: {doubt['time']}")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Download as PDF"):
                content = "My Doubt History\n\n"
                for i, d in enumerate(st.session_state.doubt_history):
                    content += f"Q{i + 1}: {d['question']}\nAsked: {d['time']}\n\n"
                path = save_as_pdf(content, "doubt_history")
                st.success("Saved!")
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.doubt_history = []
                st.success("Cleared!")
                st.rerun()
    else:
        st.info("No questions yet — go to Chat and ask something!")