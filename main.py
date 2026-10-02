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
# BLUE AI INTERFACE
# =========================================================

st.markdown("""
<style>

/* =========================
   GLOBAL
========================= */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(37, 99, 235, 0.14),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(14, 165, 233, 0.12),
            transparent 28%
        ),
        linear-gradient(
            135deg,
            #f8fbff 0%,
            #eef6ff 50%,
            #f8fbff 100%
        );
}

/* =========================
   MAIN CONTAINER
========================= */

.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 120px;
}


/* =========================
   AI HEADER
========================= */

.ai-header {
    position: relative;
    overflow: hidden;

    padding: 32px;
    border-radius: 28px;

    background:
        linear-gradient(
            135deg,
            #071a3d 0%,
            #0b3b91 48%,
            #0879d9 100%
        );

    box-shadow:
        0 20px 50px rgba(15, 80, 180, 0.25);

    color: white;
    margin-bottom: 24px;
}

.ai-header:before {
    content: "";
    position: absolute;

    width: 260px;
    height: 260px;

    border-radius: 50%;

    background: rgba(255,255,255,0.08);

    right: -80px;
    top: -100px;
}

.ai-header:after {
    content: "";
    position: absolute;

    width: 180px;
    height: 180px;

    border-radius: 50%;

    background: rgba(56,189,248,0.12);

    left: 45%;
    bottom: -120px;
}

.ai-logo {
    font-size: 46px;
    margin-bottom: 5px;
}

.ai-title {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0;
}

.ai-subtitle {
    font-size: 15px;
    color: #dbeafe;
    margin-top: 8px;
}

.ai-status {
    display: inline-flex;

    margin-top: 18px;
    padding: 7px 13px;

    border-radius: 30px;

    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.18);

    font-size: 12px;
    font-weight: 600;
}


/* =========================
   STAT CARDS
========================= */

.stat-card {
    background: rgba(255,255,255,0.85);

    border: 1px solid #dbeafe;

    border-radius: 20px;

    padding: 20px;

    box-shadow:
        0 8px 30px rgba(30,64,175,0.07);

    transition: 0.25s;
}

.stat-card:hover {
    transform: translateY(-3px);

    box-shadow:
        0 14px 35px rgba(30,64,175,0.13);
}

.stat-icon {
    font-size: 23px;
}

.stat-label {
    font-size: 11px;
    color: #64748b;

    text-transform: uppercase;
    letter-spacing: 0.7px;

    margin-top: 8px;
}

.stat-value {
    font-size: 25px;
    font-weight: 800;

    color: #0f3c88;

    margin-top: 3px;
}


/* =========================
   WELCOME CARD
========================= */

.welcome-card {
    margin-top: 25px;

    padding: 25px;

    border-radius: 22px;

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.95),
            rgba(239,246,255,0.9)
        );

    border: 1px solid #bfdbfe;

    box-shadow:
        0 10px 35px rgba(30,64,175,0.07);
}

.welcome-title {
    font-size: 24px;
    font-weight: 750;

    color: #0f3c88;
}

.welcome-text {
    color: #64748b;
    font-size: 14px;
}


/* =========================
   FEATURE CARDS
========================= */

.feature-card {
    background: white;

    border-radius: 18px;

    padding: 18px;

    border: 1px solid #e0ecff;

    min-height: 115px;

    box-shadow:
        0 6px 20px rgba(15, 60, 140, 0.05);
}

.feature-icon {
    font-size: 25px;
}

.feature-title {
    font-weight: 700;

    color: #123b7a;

    margin-top: 6px;
}

.feature-text {
    font-size: 12px;
    color: #64748b;

    margin-top: 4px;
}


/* =========================
   CHAT
========================= */

[data-testid="stChatMessage"] {

    border-radius: 20px;

    margin-top: 8px;
    margin-bottom: 8px;

    border: 1px solid #e5eefc;
}

[data-testid="stChatMessage"]:has(
    [data-testid="chatAvatarIcon-assistant"]
) {
    background:
        linear-gradient(
            135deg,
            #f0f7ff,
            #ffffff
        );
}


/* =========================
   CHAT INPUT
========================= */

[data-testid="stChatInput"] {
    border-radius: 20px;

    border: 1px solid #93c5fd;

    box-shadow:
        0 5px 25px rgba(37,99,235,0.12);
}


/* =========================
   SIDEBAR
========================= */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #061735 0%,
            #0b2859 55%,
            #08204a 100%
        );
}

section[data-testid="stSidebar"] * {
    color: #e0efff;
}

.sidebar-brand {

    padding: 10px 5px 20px;

    border-bottom:
        1px solid rgba(255,255,255,0.10);

    margin-bottom: 20px;
}

.sidebar-logo {
    font-size: 30px;
}

.sidebar-title {
    font-size: 22px;
    font-weight: 800;

    color: white;
}

.sidebar-desc {
    font-size: 11px;
    color: #93c5fd;
}


/* =========================
   SIDEBAR STATUS
========================= */

.sidebar-status {

    padding: 15px;

    border-radius: 16px;

    background:
        rgba(37,99,235,0.18);

    border:
        1px solid rgba(147,197,253,0.18);

    margin-top: 15px;
}

.sidebar-status-title {
    font-weight: 700;
    color: #bfdbfe;
}

.sidebar-status-text {
    font-size: 12px;
    color: #93c5fd;
    margin-top: 5px;
}


/* =========================
   BUTTON
========================= */

.stButton > button {

    border-radius: 12px !important;

    border: 1px solid #93c5fd !important;

    transition: 0.2s;
}

.stButton > button:hover {

    border-color: #2563eb !important;

    box-shadow:
        0 5px 18px rgba(37,99,235,0.18);
}


/* =========================
   FOOTER
========================= */

.ai-footer {

    text-align: center;

    margin-top: 45px;

    color: #64748b;

    font-size: 11px;
}


/* =========================
   MOBILE RESPONSIVE
========================= */

@media (max-width: 768px) {

    .block-container {
        padding-left: 15px;
        padding-right: 15px;
    }

    .ai-header {
        padding: 23px;
        border-radius: 22px;
    }

    .ai-title {
        font-size: 30px;
    }

    .ai-logo {
        font-size: 35px;
    }

    .ai-subtitle {
        font-size: 13px;
    }

    .stat-card {
        margin-bottom: 10px;
    }

}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

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

if "dataset_key" not in st.session_state:
    st.session_state.dataset_key = None


# =========================================================
# ABBREVIATION / SHORT FORM SUPPORT
# =========================================================

SHORT_FORMS = {

    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "llm": "large language model",
    "genai": "generative artificial intelligence",
    "gen ai": "generative artificial intelligence",
    "rag": "retrieval augmented generation",
    "cnn": "convolutional neural network",
    "rnn": "recurrent neural network",
    "lstm": "long short term memory",
    "gru": "gated recurrent unit",
    "bert": "bidirectional encoder representations from transformers",
    "api": "application programming interface",
    "eda": "exploratory data analysis",
    "pca": "principal component analysis",
    "svm": "support vector machine",
    "knn": "k nearest neighbors",
    "rf": "random forest",
    "dt": "decision tree",
    "ann": "artificial neural network",
    "gan": "generative adversarial network",
    "vae": "variational autoencoder",
    "ocr": "optical character recognition",
    "asr": "automatic speech recognition",
    "tts": "text to speech",
    "stt": "speech to text"
}


def expand_short_forms(text):

    text = text.lower()

    # Normalize special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    words = text.split()

    expanded = []

    for word in words:

        if word in SHORT_FORMS:

            expanded.append(word)
            expanded.append(SHORT_FORMS[word])

        else:

            expanded.append(word)

    return " ".join(expanded)


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_model()


# =========================================================
# PARSE DATASET
# =========================================================

def parse_dataset(uploaded_file):

    raw_text = uploaded_file.read()

    text = raw_text.decode(
        "utf-8",
        errors="ignore"
    )

    questions = []
    answers = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if "|" not in line:
            continue

        question, answer = line.split(
            "|",
            1
        )

        question = question.strip()
        answer = answer.strip()

        if question and answer:

            questions.append(question)
            answers.append(answer)

    return questions, answers


# =========================================================
# CREATE SEMANTIC EMBEDDINGS
# =========================================================

@st.cache_data(show_spinner=False)
def create_embeddings(question_tuple):

    processed_questions = []

    for question in question_tuple:

        # Add expanded meaning
        expanded = expand_short_forms(
            question
        )

        processed_questions.append(
            expanded
        )

    embeddings = embedding_model.encode(
        processed_questions,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return np.array(embeddings)


# =========================================================
# CREATE TF-IDF
# =========================================================

@st.cache_data(show_spinner=False)
def create_tfidf(question_tuple):

    processed_questions = []

    for question in question_tuple:

        expanded = expand_short_forms(
            question
        )

        processed_questions.append(
            expanded
        )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    matrix = vectorizer.fit_transform(
        processed_questions
    )

    return vectorizer, matrix


# =========================================================
# SMART SEARCH
# =========================================================

def search_dataset(
    user_question,
    top_k=5
):

    if not st.session_state.questions:

        return []

    # Original question
    original_query = clean_text(
        user_question
    )

    # Expanded question
    expanded_query = expand_short_forms(
        original_query
    )

    # -------------------------
    # Semantic search
    # -------------------------

    query_embedding = embedding_model.encode(
        [expanded_query],
        normalize_embeddings=True
    )

    semantic_scores = cosine_similarity(
        query_embedding,
        st.session_state.embeddings
    )[0]

    # -------------------------
    # TF-IDF search
    # -------------------------

    query_tfidf = (
        st.session_state
        .tfidf_vectorizer
        .transform(
            [expanded_query]
        )
    )

    lexical_scores = cosine_similarity(
        query_tfidf,
        st.session_state.tfidf_matrix
    )[0]

    # -------------------------
    # Hybrid intelligence
    # -------------------------

    final_scores = (
        semantic_scores * 0.75
        +
        lexical_scores * 0.25
    )

    indices = np.argsort(
        final_scores
    )[::-1][:top_k]

    results = []

    for index in indices:

        results.append({

            "question":
                st.session_state
                .questions[index],

            "answer":
                st.session_state
                .answers[index],

            "score":
                float(
                    final_scores[index]
                ),

            "semantic":
                float(
                    semantic_scores[index]
                ),

            "lexical":
                float(
                    lexical_scores[index]
                )
        })

    return results


# =========================================================
# GEMINI CLIENT
# =========================================================

def get_gemini_client():

    try:

        api_key = st.secrets.get(
            "GEMINI_API_KEY",
            ""
        )

    except Exception:

        api_key = ""

    if not api_key:

        return None

    return genai.Client(
        api_key=api_key
    )


# =========================================================
# AI ANSWER
# =========================================================

def generate_ai_answer(
    question,
    matches
):

    client = get_gemini_client()

    if client is None:

        return (
            "⚠️ Gemini API key is not configured.\n\n"
            "Please add `GEMINI_API_KEY` "
            "to Streamlit Secrets."
        )

    # Create context
    context = ""

    for i, item in enumerate(
        matches[:4],
        start=1
    ):

        context += f"""
Knowledge {i}:

Question:
{item["question"]}

Answer:
{item["answer"]}

"""

    prompt = f"""
You are IntelliMind AI, a professional educational AI assistant.

USER QUESTION:
{question}

RELEVANT KNOWLEDGE:
{context}

IMPORTANT RULES:

1. Understand abbreviations and short forms.

Examples:
ML = Machine Learning
DL = Deep Learning
CV = Computer Vision
AI = Artificial Intelligence
NLP = Natural Language Processing
LLM = Large Language Model
CNN = Convolutional Neural Network
RNN = Recurrent Neural Network

2. If the user asks:
"what is ml?"
understand it as:
"What is Machine Learning?"

3. If the user asks:
"dl vs ml"
understand it as:
"Deep Learning vs Machine Learning"

4. If the user asks:
"cv ki?"
understand it as:
"What is Computer Vision?"

5. Give simple, clear and natural answers.

6. Use the uploaded knowledge when relevant.

7. Do not mention:
- similarity score
- embeddings
- TF-IDF
- dataset retrieval
- internal system
- prompt

8. If the knowledge base contains the answer,
use it as the main source.

9. If the knowledge base is insufficient,
use your general AI knowledge.

10. For comparison questions, use a small table
when useful.

11. For technical questions, give a simple example
when useful.

12. Keep answers concise unless the user asks
for detailed explanation.

ANSWER:
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

        error = str(e)

        if "503" in error:

            return (
                "⚠️ Gemini is currently busy.\n\n"
                "Please try again in a few seconds."
            )

        return (
            "⚠️ AI generation error:\n\n"
            + error
        )


# =========================================================
# SMART ANSWER ENGINE
# =========================================================

def answer_question(question):

    matches = search_dataset(
        question,
        top_k=5
    )

    if not matches:

        return (
            "I couldn't find any relevant information.",
            [],
            0
        )

    best = matches[0]

    score = best["score"]

    # ==========================================
    # VERY HIGH CONFIDENCE
    # ==========================================

    if score >= 0.70:

        return (
            best["answer"],
            matches,
            score
        )

    # ==========================================
    # GOOD CONFIDENCE
    # ==========================================

    elif score >= 0.48:

        ai_answer = generate_ai_answer(
            question,
            matches
        )

        return (
            ai_answer,
            matches,
            score
        )

    # ==========================================
    # LOW CONFIDENCE
    # ==========================================

    else:

        ai_answer = generate_ai_answer(
            question,
            matches
        )

        return (
            ai_answer,
            matches,
            score
        )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">

            <div class="sidebar-logo">
                🤖
            </div>

            <div class="sidebar-title">
                IntelliMind AI
            </div>

            <div class="sidebar-desc">
                Intelligent NLP Knowledge Assistant
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "### 📚 Knowledge Base"
    )

    uploaded_file = st.file_uploader(
        "Upload TXT Dataset",
        type=["txt"],
        help="Format: Question | Answer"
    )

    if uploaded_file:

        try:

            questions, answers = parse_dataset(
                uploaded_file
            )

            if questions:

                current_key = (
                    uploaded_file.name,
                    uploaded_file.size
                )

                if (
                    st.session_state.dataset_key
                    != current_key
                ):

                    with st.spinner(
                        "🧠 Training knowledge index..."
                    ):

                        embeddings = (
                            create_embeddings(
                                tuple(questions)
                            )
                        )

                        vectorizer, matrix = (
                            create_tfidf(
                                tuple(questions)
                            )
                        )

                    st.session_state.questions = (
                        questions
                    )

                    st.session_state.answers = (
                        answers
                    )

                    st.session_state.embeddings = (
                        embeddings
                    )

                    st.session_state.tfidf_vectorizer = (
                        vectorizer
                    )

                    st.session_state.tfidf_matrix = (
                        matrix
                    )

                    st.session_state.dataset_key = (
                        current_key
                    )

                    st.session_state.messages = []

                st.markdown(
                    f"""
                    <div class="sidebar-status">

                        <div class="sidebar-status-title">
                            🟢 Knowledge Base Active
                        </div>

                        <div class="sidebar-status-text">
                            {len(questions)}
                            Q&A entries loaded
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.error(
                    "No valid Q&A found."
                )

        except Exception as e:

            st.error(
                f"Dataset error: {e}"
            )

    else:

        st.markdown(
            """
            <div class="sidebar-status">

                <div class="sidebar-status-title">
                    🔵 Waiting for Dataset
                </div>

                <div class="sidebar-status-text">
                    Upload your .txt knowledge
                    base to start.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    st.markdown(
        "### ⚙️ Settings"
    )

    show_sources = st.checkbox(
        "🔎 Show matched knowledge",
        value=False
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.markdown(
        """
        **AI Engine**

        🧠 Sentence Transformer  
        🔍 Semantic Search  
        📊 TF-IDF  
        🤖 Gemini  
        ⚡ Hybrid Retrieval
        """
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    """
    <div class="ai-header">

        <div class="ai-logo">
            🤖
        </div>

        <div class="ai-title">
            IntelliMind AI
        </div>

        <div class="ai-subtitle">
            Your intelligent knowledge assistant powered by
            NLP, semantic search and Generative AI.
        </div>

        <div class="ai-status">
            ● AI SYSTEM ONLINE
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# STATS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.markdown(
        f"""
        <div class="stat-card">

            <div class="stat-icon">📚</div>

            <div class="stat-label">
                Knowledge
            </div>

            <div class="stat-value">
                {len(st.session_state.questions)}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        """
        <div class="stat-card">

            <div class="stat-icon">🧠</div>

            <div class="stat-label">
                NLP Engine
            </div>

            <div class="stat-value">
                Active
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c3:

    st.markdown(
        """
        <div class="stat-card">

            <div class="stat-icon">🔎</div>

            <div class="stat-label">
                Search
            </div>

            <div class="stat-value">
                Semantic
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c4:

    st.markdown(
        """
        <div class="stat-card">

            <div class="stat-icon">⚡</div>

            <div class="stat-label">
                AI Model
            </div>

            <div class="stat-value">
                Gemini
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# WELCOME
# =========================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div class="welcome-card">

            <div class="welcome-title">
                👋 Welcome to IntelliMind
            </div>

            <div class="welcome-text">
                Ask questions naturally. IntelliMind can
                understand technical short forms such as
                AI, ML, DL, CV, NLP and LLM.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    f1, f2, f3 = st.columns(3)

    with f1:

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-icon">
                    🧠
                </div>

                <div class="feature-title">
                    Smart Understanding
                </div>

                <div class="feature-text">
                    Understands natural questions
                    and technical abbreviations.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with f2:

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-icon">
                    🔍
                </div>

                <div class="feature-title">
                    Semantic Search
                </div>

                <div class="feature-text">
                    Finds related questions even
                    when wording is different.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with f3:

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-icon">
                    ⚡
                </div>

                <div class="feature-title">
                    Generative AI
                </div>

                <div class="feature-text">
                    Gemini creates natural answers
                    when more explanation is needed.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    role = message["role"]

    avatar = (
        "🤖"
        if role == "assistant"
        else "👤"
    )

    with st.chat_message(
        role,
        avatar=avatar
    ):

        st.markdown(
            message["content"]
        )

        if (
            role == "assistant"
            and show_sources
            and message.get("matches")
        ):

            with st.expander(
                "🔎 Knowledge Match"
            ):

                for i, item in enumerate(
                    message["matches"][:3],
                    start=1
                ):

                    st.markdown(
                        f"""
                        **{i}. {item["question"]}**

                        Similarity:
                        `{item["score"]:.3f}`

                        {item["answer"]}
                        """
                    )


# =========================================================
# USER INPUT
# =========================================================

user_question = st.chat_input(
    "Ask anything... e.g. What is ML?"
)


if user_question:

    # ---------------------------------------------
    # Dataset check
    # ---------------------------------------------

    if not st.session_state.questions:

        st.warning(
            "📚 Please upload your TXT dataset first."
        )

        st.stop()

    # ---------------------------------------------
    # User message
    # ---------------------------------------------

    st.session_state.messages.append({

        "role": "user",

        "content": user_question
    })

    with st.chat_message(
        "user",
        avatar="👤"
    ):

        st.markdown(
            user_question
        )

    # ---------------------------------------------
    # AI response
    # ---------------------------------------------

    with st.chat_message(
        "assistant",
        avatar="🤖"
    ):

        with st.spinner(
            "🧠 Thinking..."
        ):

            answer, matches, score = (
                answer_question(
                    user_question
                )
            )

        st.markdown(
            answer
        )

        if show_sources and matches:

            with st.expander(
                f"🔎 Knowledge Match • {score:.2f}"
            ):

                for i, item in enumerate(
                    matches[:3],
                    start=1
                ):

                    st.markdown(
                        f"""
                        **{i}. {item["question"]}**

                        Score:
                        `{item["score"]:.3f}`

                        {item["answer"]}
                        """
                    )

    # ---------------------------------------------
    # Save response
    # ---------------------------------------------

    st.session_state.messages.append({

        "role": "assistant",

        "content": answer,

        "matches": matches,

        "score": score
    })


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="ai-footer">

        IntelliMind AI • NLP • Semantic Search •
        Machine Learning • Generative AI

    </div>
    """,
    unsafe_allow_html=True
)
