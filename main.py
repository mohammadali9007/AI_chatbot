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
# PREMIUM BLUE AI UI
# =========================================================

st.markdown("""
<style>

/* =====================================================
   GLOBAL
===================================================== */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        linear-gradient(
            135deg,
            #f8fbff 0%,
            #eef5ff 50%,
            #f8fbff 100%
        );
}

/* Main width */
.block-container {
    max-width: 1100px;
    padding-top: 25px;
    padding-bottom: 120px;
}


/* =====================================================
   SIDEBAR
===================================================== */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #061735 0%,
            #082657 50%,
            #061a3d 100%
        );

    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: #e7f1ff;
}

.sidebar-logo {

    display: flex;
    align-items: center;
    gap: 12px;

    padding: 8px 5px 22px;

    border-bottom:
        1px solid rgba(255,255,255,0.10);
}

.logo-circle {

    width: 44px;
    height: 44px;

    border-radius: 14px;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 23px;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #06b6d4
        );

    box-shadow:
        0 8px 25px rgba(37,99,235,0.35);
}

.logo-name {
    font-size: 18px;
    font-weight: 700;
}

.logo-sub {
    font-size: 10px;
    color: #8fb5e8 !important;
    margin-top: 2px;
}

.sidebar-section {
    margin-top: 24px;
    margin-bottom: 10px;

    font-size: 11px;
    font-weight: 600;

    color: #7fb0ed !important;

    text-transform: uppercase;
    letter-spacing: 1px;
}


/* =====================================================
   HERO
===================================================== */

.hero {

    text-align: center;

    padding-top: 15px;
    padding-bottom: 25px;
}

.hero-icon {

    width: 70px;
    height: 70px;

    margin: auto;

    border-radius: 22px;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 35px;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #06b6d4
        );

    box-shadow:
        0 12px 40px rgba(37,99,235,0.28);
}

.hero-title {

    margin-top: 18px;

    font-size: 34px;

    font-weight: 700;

    letter-spacing: -1px;

    color: #0f2f63;
}

.hero-subtitle {

    margin-top: 7px;

    font-size: 13px;

    color: #64748b;
}


/* =====================================================
   ONLINE STATUS
===================================================== */

.online {

    display: inline-flex;

    align-items: center;

    gap: 6px;

    margin-top: 13px;

    padding: 5px 12px;

    border-radius: 30px;

    background: #eff6ff;

    border: 1px solid #bfdbfe;

    color: #2563eb;

    font-size: 11px;

    font-weight: 600;
}

.online-dot {

    width: 7px;
    height: 7px;

    border-radius: 50%;

    background: #22c55e;

    box-shadow:
        0 0 10px #22c55e;
}


/* =====================================================
   CHAT AREA
===================================================== */

.chat-container {

    margin-top: 10px;
}


/* Assistant message */

[data-testid="stChatMessage"] {

    border: none !important;

    background: transparent !important;

    padding-top: 5px;
    padding-bottom: 5px;
}


/* User bubble */

[data-testid="stChatMessage"]:has(
    [data-testid="chatAvatarIcon-user"]
) {

    background: transparent !important;
}


/* Chat text */

[data-testid="stChatMessageContent"] {

    font-size: 14px;

    line-height: 1.7;
}


/* =====================================================
   CHAT INPUT
===================================================== */

[data-testid="stChatInput"] {

    position: fixed !important;

    bottom: 20px !important;

    left: 50% !important;

    transform: translateX(-50%);

    width: min(850px, 85%) !important;

    z-index: 999;

    border-radius: 20px !important;

    border: 1px solid #bfdbfe !important;

    background: rgba(255,255,255,0.96) !important;

    box-shadow:
        0 10px 35px rgba(30,64,175,0.16);

    backdrop-filter: blur(12px);
}


/* =====================================================
   WELCOME SCREEN
===================================================== */

.welcome {

    max-width: 760px;

    margin: 25px auto 10px;

    text-align: center;
}

.welcome-title {

    font-size: 22px;

    font-weight: 600;

    color: #163d78;
}

.welcome-text {

    margin-top: 8px;

    color: #64748b;

    font-size: 13px;
}


/* =====================================================
   EXAMPLE PROMPTS
===================================================== */

.prompt-card {

    background: rgba(255,255,255,0.75);

    border: 1px solid #dbeafe;

    border-radius: 15px;

    padding: 15px;

    margin-top: 10px;

    text-align: left;

    transition: 0.2s;
}

.prompt-card:hover {

    border-color: #93c5fd;

    box-shadow:
        0 8px 25px rgba(37,99,235,0.08);
}

.prompt-icon {
    font-size: 18px;
}

.prompt-title {

    margin-top: 5px;

    font-size: 12px;

    font-weight: 600;

    color: #1e40af;
}

.prompt-text {

    margin-top: 4px;

    font-size: 11px;

    color: #64748b;
}


/* =====================================================
   DATASET BOX
===================================================== */

.dataset-box {

    padding: 13px;

    margin-top: 10px;

    border-radius: 13px;

    background:
        rgba(37,99,235,0.12);

    border:
        1px solid rgba(147,197,253,0.20);
}

.dataset-title {

    font-size: 12px;

    font-weight: 600;

    color: #bfdbfe !important;
}

.dataset-text {

    margin-top: 4px;

    font-size: 10px;

    color: #8fb5e8 !important;
}


/* =====================================================
   BUTTONS
===================================================== */

.stButton > button {

    border-radius: 10px !important;

    border: 1px solid #dbeafe !important;

    background: white !important;

    color: #1e40af !important;

    font-size: 12px !important;

    transition: 0.2s;
}

.stButton > button:hover {

    border-color: #60a5fa !important;

    background: #eff6ff !important;

}


/* =====================================================
   EXPANDER
===================================================== */

[data-testid="stExpander"] {

    border: 1px solid #dbeafe !important;

    border-radius: 12px !important;

    background: rgba(239,246,255,0.5);
}


/* =====================================================
   MOBILE
===================================================== */

@media (max-width: 768px) {

    .block-container {

        padding-left: 15px;
        padding-right: 15px;

    }

    .hero-title {

        font-size: 27px;

    }

    .hero-icon {

        width: 60px;
        height: 60px;

        font-size: 29px;

    }

    [data-testid="stChatInput"] {

        width: 92% !important;

        bottom: 12px !important;

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

if "tfidf_vectorizer" not in st.session_state:
    st.session_state.tfidf_vectorizer = None

if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None

if "dataset_key" not in st.session_state:
    st.session_state.dataset_key = None


# =========================================================
# SHORT FORM DICTIONARY
# =========================================================

SHORT_FORMS = {

    "ai": "artificial intelligence",

    "ml": "machine learning",

    "dl": "deep learning",

    "nlp": "natural language processing",

    "cv": "computer vision",

    "llm": "large language model",

    "rag": "retrieval augmented generation",

    "cnn": "convolutional neural network",

    "rnn": "recurrent neural network",

    "lstm": "long short term memory",

    "gru": "gated recurrent unit",

    "bert":
        "bidirectional encoder representations from transformers",

    "api":
        "application programming interface",

    "svm":
        "support vector machine",

    "knn":
        "k nearest neighbors",

    "rf":
        "random forest",

    "dt":
        "decision tree",

    "ann":
        "artificial neural network",

    "gan":
        "generative adversarial network",

    "ocr":
        "optical character recognition",

    "tts":
        "text to speech",

    "stt":
        "speech to text",

    "eda":
        "exploratory data analysis"
}


# =========================================================
# EXPAND SHORT FORMS
# =========================================================

def expand_short_forms(text):

    text = text.lower()

    # Keep useful characters
    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    words = text.split()

    result = []

    for word in words:

        result.append(word)

        if word in SHORT_FORMS:

            result.append(
                SHORT_FORMS[word]
            )

    return " ".join(result)


# =========================================================
# CLEAN TEXT
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
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


model = load_model()


# =========================================================
# PARSE TXT DATASET
# =========================================================

def parse_dataset(file):

    content = file.read().decode(
        "utf-8",
        errors="ignore"
    )

    questions = []
    answers = []

    for line in content.splitlines():

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
# EMBEDDINGS
# =========================================================

@st.cache_data(show_spinner=False)
def build_embeddings(question_tuple):

    processed = [
        expand_short_forms(q)
        for q in question_tuple
    ]

    return model.encode(
        processed,
        normalize_embeddings=True
    )


# =========================================================
# TF-IDF
# =========================================================

@st.cache_data(show_spinner=False)
def build_tfidf(question_tuple):

    processed = [
        expand_short_forms(q)
        for q in question_tuple
    ]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    matrix = vectorizer.fit_transform(
        processed
    )

    return vectorizer, matrix


# =========================================================
# SMART SEARCH
# =========================================================

def find_matches(
    question,
    top_k=5
):

    if not st.session_state.questions:

        return []

    query = expand_short_forms(
        question
    )

    # Semantic
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    semantic_scores = cosine_similarity(
        query_embedding,
        st.session_state.embeddings
    )[0]

    # TF-IDF
    query_tfidf = (
        st.session_state
        .tfidf_vectorizer
        .transform([query])
    )

    lexical_scores = cosine_similarity(
        query_tfidf,
        st.session_state.tfidf_matrix
    )[0]

    # Hybrid score
    final_scores = (
        semantic_scores * 0.75
        +
        lexical_scores * 0.25
    )

    indices = np.argsort(
        final_scores
    )[::-1][:top_k]

    results = []

    for i in indices:

        results.append({

            "question":
                st.session_state.questions[i],

            "answer":
                st.session_state.answers[i],

            "score":
                float(final_scores[i])
        })

    return results


# =========================================================
# GEMINI CLIENT
# =========================================================

def get_client():

    try:

        api_key = st.secrets.get(
            "GEMINI_API_KEY",
            ""
        )

    except:

        api_key = ""

    if not api_key:

        return None

    return genai.Client(
        api_key=api_key
    )


# =========================================================
# GEMINI
# =========================================================

def generate_ai_answer(
    question,
    matches
):

    client = get_client()

    if client is None:

        return (
            "⚠️ Gemini API key is not configured."
        )

    context = ""

    for item in matches[:4]:

        context += f"""

Question:
{item["question"]}

Answer:
{item["answer"]}

"""

    prompt = f"""
You are IntelliMind AI.

You are a smart educational assistant.

User Question:
{question}

Relevant Knowledge:
{context}

Rules:

1. Understand short forms and abbreviations.

ML = Machine Learning
DL = Deep Learning
CV = Computer Vision
AI = Artificial Intelligence
NLP = Natural Language Processing
LLM = Large Language Model
CNN = Convolutional Neural Network
RNN = Recurrent Neural Network
RAG = Retrieval Augmented Generation

2. If user writes "ML ki?"
understand it as:
"What is Machine Learning?"

3. If user writes "DL vs ML"
understand it as:
"Deep Learning vs Machine Learning."

4. Answer in simple English.

5. If user asks in Bangla,
you may answer in simple Bangla.

6. Use the provided knowledge whenever relevant.

7. If knowledge is insufficient,
use your general knowledge.

8. Do not mention the dataset,
similarity score, embeddings,
TF-IDF or internal system.

9. Be concise but useful.

10. For comparison questions,
use a simple table when useful.

Answer the user's question directly.
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if response and response.text:

            return response.text.strip()

        return "I couldn't generate an answer."

    except Exception as e:

        if "503" in str(e):

            return (
                "⚠️ AI server is currently busy. "
                "Please try again in a few seconds."
            )

        return (
            "⚠️ AI generation error: "
            + str(e)
        )


# =========================================================
# ANSWER ENGINE
# =========================================================

def get_answer(question):

    matches = find_matches(
        question,
        top_k=5
    )

    if not matches:

        return (
            "I couldn't find a relevant answer.",
            [],
            0
        )

    best = matches[0]

    score = best["score"]

    # High confidence
    if score >= 0.68:

        return (
            best["answer"],
            matches,
            score
        )

    # Medium / low confidence
    answer = generate_ai_answer(
        question,
        matches
    )

    return (
        answer,
        matches,
        score
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-logo">

            <div class="logo-circle">
                🤖
            </div>

            <div>
                <div class="logo-name">
                    IntelliMind AI
                </div>

                <div class="logo-sub">
                    Smart Knowledge Assistant
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-section">Knowledge Base</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload your TXT dataset",
        type=["txt"],
        label_visibility="visible"
    )

    if uploaded_file:

        try:

            questions, answers = parse_dataset(
                uploaded_file
            )

            if questions:

                dataset_key = (
                    uploaded_file.name,
                    uploaded_file.size
                )

                if (
                    st.session_state.dataset_key
                    != dataset_key
                ):

                    with st.spinner(
                        "Learning dataset..."
                    ):

                        embeddings = build_embeddings(
                            tuple(questions)
                        )

                        vectorizer, matrix = (
                            build_tfidf(
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
                        dataset_key
                    )

                    st.session_state.messages = []

                st.markdown(
                    f"""
                    <div class="dataset-box">

                        <div class="dataset-title">
                            🟢 Knowledge Base Ready
                        </div>

                        <div class="dataset-text">
                            {len(questions)}
                            questions loaded
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
                f"Error: {e}"
            )

    else:

        st.markdown(
            """
            <div class="dataset-box">

                <div class="dataset-title">
                    🔵 No Dataset
                </div>

                <div class="dataset-text">
                    Upload a Question | Answer
                    TXT file.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="sidebar-section">Options</div>',
        unsafe_allow_html=True
    )

    show_matches = st.checkbox(
        "Show knowledge matches",
        value=False
    )

    st.write("")

    if st.button(
        "🗑️ New Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.markdown(
        """
        <div style="
            margin-top:25px;
            padding-top:15px;
            border-top:1px solid rgba(255,255,255,0.08);
            font-size:10px;
            color:#6f9bd2;
        ">
        IntelliMind AI<br>
        NLP • Semantic Search • Gemini
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HERO
# =========================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div class="hero">

            <div class="hero-icon">
                🤖
            </div>

            <div class="hero-title">
                IntelliMind AI
            </div>

            <div class="hero-subtitle">
                Ask questions. Explore knowledge.
                Get intelligent answers.
            </div>

            <div class="online">

                <span class="online-dot"></span>

                AI ONLINE

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
        <div class="welcome">

            <div class="welcome-title">
                What can I help you with?
            </div>

            <div class="welcome-text">
                Ask naturally — IntelliMind understands
                ML, DL, CV, NLP, AI, LLM and other
                technical short forms.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="prompt-card">

                <div class="prompt-icon">
                    🧠
                </div>

                <div class="prompt-title">
                    Machine Learning
                </div>

                <div class="prompt-text">
                    What is ML?
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="prompt-card">

                <div class="prompt-icon">
                    👁️
                </div>

                <div class="prompt-title">
                    Computer Vision
                </div>

                <div class="prompt-text">
                    What is CV?
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="prompt-card">

                <div class="prompt-icon">
                    ✨
                </div>

                <div class="prompt-title">
                    Deep Learning
                </div>

                <div class="prompt-text">
                    DL vs ML
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
            and show_matches
            and message.get("matches")
        ):

            with st.expander(
                "🔎 Knowledge used"
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
                        """
                    )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Message IntelliMind AI..."
)


if question:

    if not st.session_state.questions:

        st.warning(
            "Please upload your TXT dataset first."
        )

        st.stop()

    # User message
    st.session_state.messages.append({

        "role": "user",

        "content": question
    })

    with st.chat_message(
        "user",
        avatar="👤"
    ):

        st.markdown(question)

    # AI message
    with st.chat_message(
        "assistant",
        avatar="🤖"
    ):

        with st.spinner(
            "Thinking..."
        ):

            answer, matches, score = (
                get_answer(question)
            )

        st.markdown(answer)

        if show_matches and matches:

            with st.expander(
                "🔎 Knowledge used"
            ):

                for i, item in enumerate(
                    matches[:3],
                    start=1
                ):

                    st.markdown(
                        f"""
                        **{i}. {item["question"]}**

                        Similarity:
                        `{item["score"]:.3f}`
                        """
                    )

    # Save AI response
    st.session_state.messages.append({

        "role": "assistant",

        "content": answer,

        "matches": matches,

        "score": score
    })
