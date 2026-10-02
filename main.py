import streamlit as st
import numpy as np
import re

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="IntelliMind AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(99,102,241,0.12), transparent 25%),
        radial-gradient(circle at 90% 20%, rgba(14,165,233,0.10), transparent 25%),
        #f8fafc;
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

/* Header */
.hero {
    padding: 30px;
    border-radius: 24px;
    background: linear-gradient(135deg, #111827, #1e293b);
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 15px 40px rgba(15,23,42,0.18);
}

.hero h1 {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.hero p {
    font-size: 16px;
    color: #cbd5e1;
    margin-bottom: 0;
}

/* Cards */
.info-card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 5px 20px rgba(15,23,42,0.05);
    min-height: 110px;
}

.info-title {
    color: #64748b;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
}

.info-value {
    font-size: 28px;
    font-weight: 800;
    color: #0f172a;
    margin-top: 5px;
}

/* Chat */
[data-testid="stChatMessage"] {
    border-radius: 18px;
    padding: 8px;
    margin-bottom: 10px;
}

/* Input */
[data-testid="stChatInput"] {
    border-radius: 18px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f172a;
}

section[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

.sidebar-title {
    font-size: 24px;
    font-weight: 800;
    color: white;
    margin-bottom: 5px;
}

.sidebar-subtitle {
    font-size: 13px;
    color: #94a3b8;
}

/* Dataset status */
.status-box {
    padding: 15px;
    border-radius: 15px;
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #065f46;
    margin-top: 15px;
}

/* Question suggestions */
.suggestion {
    padding: 12px 15px;
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    margin-bottom: 8px;
    color: #334155;
}

/* Hide default menu/footer */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "dataset" not in st.session_state:
    st.session_state.dataset = []

if "questions" not in st.session_state:
    st.session_state.questions = []

if "answers" not in st.session_state:
    st.session_state.answers = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None

if "tfidf_vectorizer" not in st.session_state:
    st.session_state.tfidf_vectorizer = None


# =========================================================
# LOAD SENTENCE TRANSFORMER
# =========================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================
# PARSE DATASET
# =========================================================

def parse_dataset(file):

    content = file.read().decode("utf-8", errors="ignore")

    questions = []
    answers = []

    for line in content.splitlines():

        line = line.strip()

        if not line:
            continue

        if "|" not in line:
            continue

        question, answer = line.split("|", 1)

        question = question.strip()
        answer = answer.strip()

        if question and answer:
            questions.append(question)
            answers.append(answer)

    return questions, answers


# =========================================================
# CREATE DATASET INDEX
# =========================================================

@st.cache_data(show_spinner=False)
def create_embeddings(questions):

    cleaned_questions = [
        clean_text(q)
        for q in questions
    ]

    embeddings = embedding_model.encode(
        cleaned_questions,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return np.array(embeddings)


@st.cache_data(show_spinner=False)
def create_tfidf(questions):

    cleaned_questions = [
        clean_text(q)
        for q in questions
    ]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    matrix = vectorizer.fit_transform(cleaned_questions)

    return vectorizer, matrix


# =========================================================
# SMART SEARCH
# =========================================================

def find_best_matches(user_question, top_k=5):

    if not st.session_state.questions:
        return []

    query = clean_text(user_question)

    # Semantic similarity
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    semantic_scores = cosine_similarity(
        query_embedding,
        st.session_state.embeddings
    )[0]

    # TF-IDF similarity
    query_tfidf = st.session_state.tfidf_vectorizer.transform(
        [query]
    )

    lexical_scores = cosine_similarity(
        query_tfidf,
        st.session_state.tfidf_matrix
    )[0]

    # Hybrid score
    final_scores = (
        0.75 * semantic_scores
        +
        0.25 * lexical_scores
    )

    top_indices = np.argsort(final_scores)[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "question": st.session_state.questions[index],
            "answer": st.session_state.answers[index],
            "score": float(final_scores[index]),
            "semantic_score": float(semantic_scores[index]),
            "lexical_score": float(lexical_scores[index])
        })

    return results


# =========================================================
# GEMINI CLIENT
# =========================================================

def get_gemini_client():

    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        api_key = ""

    if not api_key:
        return None

    return genai.Client(api_key=api_key)


# =========================================================
# GEMINI ANSWER
# =========================================================

def generate_ai_answer(question, matches):

    client = get_gemini_client()

    if client is None:
        return (
            "⚠️ Gemini API key is not configured.\n\n"
            "Please add `GEMINI_API_KEY` to Streamlit Secrets."
        )

    context_parts = []

    for item in matches[:3]:

        context_parts.append(
            f"Question: {item['question']}\n"
            f"Answer: {item['answer']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are IntelliMind AI, an intelligent educational assistant.

User Question:
{question}

Relevant knowledge from the uploaded dataset:
{context}

Instructions:

1. Understand the user's question carefully.
2. Use the dataset information whenever it is relevant.
3. If the dataset answer is relevant, explain it naturally.
4. Do not blindly copy the dataset.
5. If the dataset information is insufficient, answer using your general knowledge.
6. Keep the answer simple and easy to understand.
7. Use examples when useful.
8. Do not mention similarity score, dataset retrieval, embeddings, or internal system details.
9. If the question is very short, understand its likely meaning from context.
10. Give a direct answer first.

Answer:
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if response and response.text:
            return response.text.strip()

        return "Sorry, I could not generate an answer."

    except Exception as e:

        error_message = str(e)

        if "503" in error_message:
            return (
                "⚠️ Gemini is currently experiencing high demand.\n\n"
                "Please try again in a few seconds."
            )

        return f"⚠️ AI generation error:\n\n{error_message}"


# =========================================================
# SMART ANSWER ENGINE
# =========================================================

def get_answer(user_question):

    matches = find_best_matches(user_question, top_k=5)

    if not matches:
        return (
            "I couldn't find any information in the uploaded dataset.",
            [],
            0
        )

    best = matches[0]
    score = best["score"]

    # -----------------------------------------------------
    # HIGH CONFIDENCE
    # -----------------------------------------------------

    if score >= 0.62:

        answer = best["answer"]

        return answer, matches, score

    # -----------------------------------------------------
    # MEDIUM CONFIDENCE
    # -----------------------------------------------------

    elif score >= 0.38:

        answer = generate_ai_answer(
            user_question,
            matches
        )

        return answer, matches, score

    # -----------------------------------------------------
    # LOW CONFIDENCE
    # -----------------------------------------------------

    else:

        answer = generate_ai_answer(
            user_question,
            matches
        )

        return answer, matches, score


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🤖 IntelliMind AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'Smart NLP + Semantic Search + Generative AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 📂 Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload your TXT dataset",
        type=["txt"],
        help="Format: Question | Answer"
    )

    if uploaded_file is not None:

        try:

            questions, answers = parse_dataset(
                uploaded_file
            )

            if questions:

                # Update only when dataset changes
                dataset_key = (
                    uploaded_file.name,
                    uploaded_file.size
                )

                if st.session_state.get(
                    "dataset_key"
                ) != dataset_key:

                    with st.spinner(
                        "🧠 Learning your dataset..."
                    ):

                        embeddings = create_embeddings(
                            tuple(questions)
                        )

                        vectorizer, matrix = create_tfidf(
                            tuple(questions)
                        )

                    st.session_state.questions = questions
                    st.session_state.answers = answers
                    st.session_state.embeddings = embeddings
                    st.session_state.tfidf_vectorizer = vectorizer
                    st.session_state.tfidf_matrix = matrix
                    st.session_state.dataset_key = dataset_key

                    st.session_state.messages = []

                st.markdown(
                    f"""
                    <div class="status-box">
                    ✅ Dataset loaded successfully<br>
                    📚 <b>{len(questions)}</b> Q&A pairs ready
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.error(
                    "No valid Question | Answer data found."
                )

        except Exception as e:

            st.error(
                f"Dataset error: {e}"
            )

    else:

        st.info(
            "Upload your .txt file to activate the AI knowledge base."
        )

    st.divider()

    st.markdown("### ⚙️ AI Settings")

    show_sources = st.checkbox(
        "Show matched dataset",
        value=False
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.markdown(
        """
        **Technology**

        🧠 Sentence Transformers  
        🔎 Semantic Search  
        📊 TF-IDF  
        🤖 Gemini AI  
        💬 Streamlit
        """
    )


# =========================================================
# HERO HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>🤖 IntelliMind AI</h1>
        <p>
            Intelligent Question Answering using NLP,
            Semantic Search and Generative AI
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# INFORMATION CARDS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-title">Knowledge Base</div>
            <div class="info-value">
                {len(st.session_state.questions)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">Search Engine</div>
            <div class="info-value">NLP</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">AI Model</div>
            <div class="info-value">Gemini</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:

    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">Mode</div>
            <div class="info-value">Smart</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")


# =========================================================
# WELCOME MESSAGE
# =========================================================

if not st.session_state.messages:

    st.markdown("### 👋 Welcome to IntelliMind AI")

    st.write(
        "Ask me anything related to your uploaded knowledge base. "
        "I can understand different ways of asking the same question."
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="suggestion">
            💡 <b>Example</b><br>
            What is Artificial Intelligence?
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="suggestion">
            🧠 <b>Example</b><br>
            What is NLP?
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="suggestion">
            🔎 <b>Example</b><br>
            Explain machine learning
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"],
        avatar="🤖" if message["role"] == "assistant" else "👤"
    ):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and show_sources
            and message.get("matches")
        ):

            with st.expander("🔎 View matched knowledge"):

                for i, item in enumerate(
                    message["matches"][:3],
                    start=1
                ):

                    st.markdown(
                        f"""
                        **{i}. {item['question']}**

                        Similarity:
                        `{item['score']:.3f}`

                        {item['answer']}
                        """
                    )


# =========================================================
# CHAT INPUT
# =========================================================

user_question = st.chat_input(
    "Ask IntelliMind AI anything..."
)


if user_question:

    # -----------------------------------------------------
    # Check dataset
    # -----------------------------------------------------

    if not st.session_state.questions:

        st.warning(
            "📂 Please upload your `.txt` dataset first."
        )

        st.stop()

    # -----------------------------------------------------
    # Add user message
    # -----------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message(
        "user",
        avatar="👤"
    ):

        st.markdown(user_question)

    # -----------------------------------------------------
    # Generate answer
    # -----------------------------------------------------

    with st.chat_message(
        "assistant",
        avatar="🤖"
    ):

        with st.spinner(
            "🧠 IntelliMind is thinking..."
        ):

            answer, matches, score = get_answer(
                user_question
            )

        st.markdown(answer)

        if show_sources and matches:

            with st.expander(
                f"🔎 Matched knowledge • {score:.2f}"
            ):

                for i, item in enumerate(
                    matches[:3],
                    start=1
                ):

                    st.markdown(
                        f"""
                        **{i}. {item['question']}**

                        Similarity: `{item['score']:.3f}`

                        {item['answer']}
                        """
                    )

    # -----------------------------------------------------
    # Save assistant message
    # -----------------------------------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "matches": matches,
        "score": score
    })
