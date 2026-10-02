# ============================================================
# INTELLIMIND AI
# Professional AI Knowledge Assistant
# ============================================================

import streamlit as st
import os
import base64
import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

from google import genai


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="IntelliMind AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False

if "questions" not in st.session_state:
    st.session_state.questions = []

if "answers" not in st.session_state:
    st.session_state.answers = []

if "semantic_embeddings" not in st.session_state:
    st.session_state.semantic_embeddings = None

if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = "No dataset"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 85% 5%,
                rgba(37, 99, 235, 0.08),
                transparent 30%
            ),
            #f5f8fd;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 25px;
        padding-bottom: 80px;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #061733 0%,
                #082c5d 55%,
                #061a38 100%
            );

        border-right: 1px solid #173d69;
    }

    section[data-testid="stSidebar"] * {
        color: #dbeafe;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.12);
    }


    /* Sidebar selectbox */

    section[data-testid="stSidebar"]
    div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(147,197,253,0.20) !important;
        border-radius: 10px !important;
    }


    /* Sidebar uploader */

    section[data-testid="stSidebar"]
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(147,197,253,0.16) !important;
        border-radius: 12px !important;
    }

    section[data-testid="stSidebar"]
    [data-testid="stFileUploader"] section {
        background: rgba(255,255,255,0.03) !important;
        border: 1px dashed rgba(147,197,253,0.30) !important;
    }


    /* Sidebar metric */

    section[data-testid="stSidebar"]
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(147,197,253,0.15) !important;
        border-radius: 12px !important;
    }

    section[data-testid="stSidebar"]
    [data-testid="stMetricLabel"] {
        color: #9fc3ea !important;
    }

    section[data-testid="stSidebar"]
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }


    /* Sidebar buttons */

    section[data-testid="stSidebar"]
    .stButton > button {
        background: rgba(255,255,255,0.06) !important;
        color: #dbeafe !important;
        border: 1px solid rgba(147,197,253,0.18) !important;
        border-radius: 10px !important;
    }

    section[data-testid="stSidebar"]
    .stButton > button:hover {
        background: rgba(59,130,246,0.20) !important;
        border-color: #60a5fa !important;
    }


    /* ======================================================
       HERO
       ====================================================== */

    .hero {
        position: relative;
        overflow: hidden;

        min-height: 345px;

        border-radius: 25px;

        background:
            linear-gradient(
                105deg,
                #061936 0%,
                #0a3268 48%,
                #145db7 100%
            );

        border: 1px solid rgba(96,165,250,0.20);

        box-shadow:
            0 20px 60px rgba(8,47,95,0.18);

        margin-bottom: 25px;
    }


    /* Brain image */

    .hero::before {
        content: "";

        position: absolute;

        width: 500px;
        height: 500px;

        right: -70px;
        top: -85px;

        background-image:
            url("data:image/png;base64,__BRAIN_IMAGE__");

        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;

        opacity: 0.46;

        filter:
            blur(1.5px)
            drop-shadow(
                0 0 30px rgba(96,165,250,0.75)
            );

        transform: rotate(3deg);
    }


    /* Hero overlay */

    .hero::after {
        content: "";

        position: absolute;
        inset: 0;

        background:
            linear-gradient(
                90deg,
                rgba(6,25,54,0.99) 0%,
                rgba(6,25,54,0.91) 38%,
                rgba(6,25,54,0.35) 75%,
                rgba(6,25,54,0.10) 100%
            );

        pointer-events: none;
    }


    .hero-content {
        position: relative;
        z-index: 5;

        padding: 55px 50px;

        max-width: 690px;
    }


    .hero-title {
        color: #ffffff;

        font-size: 42px;

        font-weight: 800;

        letter-spacing: -1.2px;

        margin-bottom: 12px;
    }


    .hero-subtitle {
        color: #c9ddf5;

        font-size: 15px;

        line-height: 1.75;

        max-width: 590px;
    }


    .hero-badge {
        display: inline-block;

        margin-top: 24px;

        padding: 8px 16px;

        border-radius: 30px;

        background: rgba(59,130,246,0.16);

        border: 1px solid rgba(147,197,253,0.30);

        color: #bfdbfe;

        font-size: 11px;

        font-weight: 700;

        letter-spacing: 0.8px;
    }


    /* ======================================================
       CARDS
       ====================================================== */

    [data-testid="stMetric"] {
        background: #ffffff;

        border: 1px solid #dce6f1;

        border-radius: 15px;

        padding: 15px;

        box-shadow:
            0 6px 22px rgba(15,60,110,0.05);
    }

    [data-testid="stMetricLabel"] {
        color: #718096 !important;
    }

    [data-testid="stMetricValue"] {
        color: #1559a5 !important;
        font-weight: 750 !important;
    }


    /* ======================================================
       FILE UPLOADER
       ====================================================== */

    [data-testid="stFileUploader"] {
        background: #ffffff;

        border: 1px solid #d6e3f0;

        border-radius: 14px;

        padding: 8px;
    }

    [data-testid="stFileUploader"] section {
        border: 1px dashed #9bb9d8 !important;

        background: #f7fbff !important;

        border-radius: 10px !important;
    }


    /* ======================================================
       CHAT
       ====================================================== */

    [data-testid="stChatMessage"] {
        background: transparent !important;

        border: none !important;

        padding-top: 8px;

        padding-bottom: 8px;
    }


    [data-testid="stChatMessageContent"] {
        border-radius: 15px;

        line-height: 1.75;

        font-size: 14px;
    }


    /* ======================================================
       CHAT INPUT
       ====================================================== */

    [data-testid="stChatInput"] {
        background: #ffffff !important;

        border: 1px solid #b9d2ed !important;

        border-radius: 17px !important;

        box-shadow:
            0 8px 30px rgba(30,90,150,0.10);
    }


    /* ======================================================
       BUTTON
       ====================================================== */

    .stButton > button {
        border-radius: 9px !important;

        border: 1px solid #c9daeb !important;

        background: #ffffff !important;

        color: #15549a !important;

        font-weight: 600 !important;
    }

    .stButton > button:hover {
        background: #edf5ff !important;

        border-color: #3982d4 !important;
    }


    /* ======================================================
       EXPANDER
       ====================================================== */

    [data-testid="stExpander"] {
        background: #ffffff !important;

        border: 1px solid #d8e4f0 !important;

        border-radius: 12px !important;
    }


    /* ======================================================
       INFO / SUCCESS
       ====================================================== */

    [data-testid="stAlert"] {
        border-radius: 12px !important;
    }


    /* ======================================================
       MOBILE
       ====================================================== */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 13px;
            padding-right: 13px;
        }

        .hero {
            min-height: 340px;
        }

        .hero-content {
            padding: 35px 25px;
        }

        .hero-title {
            font-size: 30px;
        }

        .hero-subtitle {
            font-size: 13px;
        }

        .hero::before {
            width: 350px;
            height: 350px;

            right: -100px;
            top: 25px;

            opacity: 0.27;
        }
    }

    </style>
    """.replace(
        "__BRAIN_IMAGE__",
        ""
    ),
    unsafe_allow_html=True
)


# ============================================================
# BRAIN IMAGE
# ============================================================

def load_brain_image():

    path = "brain.png"

    if not os.path.exists(path):
        return ""

    try:

        with open(path, "rb") as f:

            return base64.b64encode(
                f.read()
            ).decode()

    except Exception:

        return ""


brain_base64 = load_brain_image()


# ============================================================
# RELOAD CSS WITH BRAIN
# ============================================================

if brain_base64:

    st.markdown(
        f"""
        <style>

        .hero::before {{

            background-image:
                url("data:image/png;base64,{brain_base64}");

            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;

            opacity: 0.46;

            filter:
                blur(1.5px)
                drop-shadow(
                    0 0 30px
                    rgba(96,165,250,0.75)
                );
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TEXT CLEANER
# ============================================================

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SHORT FORM DICTIONARY
# ============================================================

SHORT_FORMS = {

    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "cnn": "convolutional neural network",
    "rnn": "recurrent neural network",
    "lstm": "long short term memory",
    "llm": "large language model",
    "rag": "retrieval augmented generation",
    "tfidf": "term frequency inverse document frequency",
    "api": "application programming interface",
    "eda": "exploratory data analysis",
    "knn": "k nearest neighbors",
    "svm": "support vector machine",
    "pca": "principal component analysis",
    "gan": "generative adversarial network",
    "bert": "bidirectional encoder representations from transformers",
    "gpu": "graphics processing unit",
    "cpu": "central processing unit"
}


def expand_short_forms(text):

    words = text.lower().split()

    expanded = []

    for word in words:

        cleaned = re.sub(
            r"[^a-z0-9]",
            "",
            word
        )

        if cleaned in SHORT_FORMS:

            expanded.append(
                SHORT_FORMS[cleaned]
            )

        expanded.append(word)

    return " ".join(expanded)


# ============================================================
# LOAD SEMANTIC MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# ============================================================
# PARSE DATASET
# ============================================================

def parse_dataset(file):

    questions = []
    answers = []

    try:

        content = file.getvalue().decode(
            "utf-8",
            errors="ignore"
        )

    except Exception:

        content = str(
            file.getvalue(),
            errors="ignore"
        )

    lines = content.splitlines()

    for line in lines:

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


# ============================================================
# BUILD SEARCH INDEX
# ============================================================

@st.cache_data(show_spinner=False)
def prepare_tfidf(questions):

    cleaned_questions = [
        clean_text(q)
        for q in questions
    ]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    matrix = vectorizer.fit_transform(
        cleaned_questions
    )

    return vectorizer, matrix


@st.cache_resource(show_spinner=False)
def create_embeddings(questions):

    model = load_embedding_model()

    expanded_questions = [
        expand_short_forms(q)
        for q in questions
    ]

    embeddings = model.encode(
        expanded_questions,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings


# ============================================================
# SEARCH BEST ANSWER
# ============================================================

def find_best_answer(question):

    if not st.session_state.questions:

        return None, None, 0.0


    questions = st.session_state.questions


    # --------------------------------------------------------
    # Query expansion
    # --------------------------------------------------------

    expanded_query = expand_short_forms(
        question
    )


    # --------------------------------------------------------
    # Semantic similarity
    # --------------------------------------------------------

    model = load_embedding_model()

    query_embedding = model.encode(
        [expanded_query],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    semantic_scores = cosine_similarity(
        query_embedding,
        st.session_state.semantic_embeddings
    )[0]


    # --------------------------------------------------------
    # TF-IDF similarity
    # --------------------------------------------------------

    cleaned_query = clean_text(
        expanded_query
    )

    query_vector = (
        st.session_state.vectorizer
        .transform([cleaned_query])
    )

    tfidf_scores = cosine_similarity(
        query_vector,
        st.session_state.tfidf_matrix
    )[0]


    # --------------------------------------------------------
    # Hybrid score
    # --------------------------------------------------------

    final_scores = (
        0.75 * semantic_scores
        +
        0.25 * tfidf_scores
    )


    best_index = int(
        np.argmax(final_scores)
    )

    best_score = float(
        final_scores[best_index]
    )


    return (
        questions[best_index],
        st.session_state.answers[best_index],
        best_score
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_gemini_client():

    try:

        api_key = st.secrets.get(
            "GEMINI_API_KEY",
            ""
        )

    except Exception:

        api_key = ""

    if not api_key:

        api_key = os.getenv(
            "GEMINI_API_KEY",
            ""
        )

    if not api_key:

        return None

    try:

        return genai.Client(
            api_key=api_key
        )

    except Exception:

        return None


# ============================================================
# AI ANSWER
# ============================================================

def generate_ai_answer(
    question,
    matched_question=None,
    matched_answer=None,
    score=0.0
):

    client = get_gemini_client()

    if client is None:

        if matched_answer:

            return matched_answer

        return (
            "Gemini API key is not configured. "
            "Please add GEMINI_API_KEY in Streamlit Secrets."
        )


    # --------------------------------------------------------
    # Dataset context
    # --------------------------------------------------------

    if matched_answer:

        context = f"""
Knowledge Base Question:
{matched_question}

Knowledge Base Answer:
{matched_answer}

Similarity Score:
{score:.2f}
"""

    else:

        context = """
No reliable matching information was found
in the uploaded knowledge base.
"""


    # --------------------------------------------------------
    # System instruction
    # --------------------------------------------------------

    prompt = f"""
You are IntelliMind AI, a helpful educational AI assistant.

User Question:
{question}

Knowledge Base:
{context}

Instructions:

1. Understand the user's question.
2. Give a direct and useful answer.
3. If the knowledge base contains the answer,
   use it as the primary source.
4. You may improve the wording and explanation.
5. Do not blindly copy irrelevant information.
6. If the knowledge base does not contain enough
   information, answer using your general AI knowledge.
7. Never invent fake facts.
8. For technical questions, explain simply.
9. The user is a beginner, so avoid unnecessarily
   difficult terminology.
10. If the user asks for a definition, give:
    - Simple definition
    - Example
    - Short explanation
11. If the user asks about ML, DL, NLP, CV, AI,
    LLM, RAG or similar terms, explain the full form
    when useful.
12. Keep the answer concise unless detailed explanation
    is specifically requested.

Answer naturally.
"""


    try:

        response = client.models.generate_content(

            model="gemini-2.5-flash",

            contents=prompt
        )

        if response and response.text:

            return response.text.strip()

        if matched_answer:

            return matched_answer

        return "I could not generate an answer."

    except Exception as e:

        error_text = str(e)

        if matched_answer:

            return matched_answer

        if "503" in error_text:

            return (
                "AI service is temporarily busy. "
                "Please try again in a moment."
            )

        return (
            "AI generation failed. "
            "Please check your Gemini API configuration."
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        # IntelliMind AI

        **Intelligent Knowledge Assistant**
        """
    )

    st.caption(
        "Semantic Search • NLP • Generative AI"
    )

    st.divider()


    # --------------------------------------------------------
    # Navigation
    # --------------------------------------------------------

    page = st.selectbox(
        "Navigation",
        [
            "AI Assistant",
            "Knowledge Base",
            "About"
        ]
    )


    st.divider()


    # --------------------------------------------------------
    # Dataset uploader
    # --------------------------------------------------------

    st.subheader(
        "Knowledge Base"
    )

    uploaded_file = st.file_uploader(
        "Upload your TXT dataset",
        type=["txt"],
        help=(
            "Each line should use: "
            "Question | Answer"
        )
    )


    # --------------------------------------------------------
    # Process dataset
    # --------------------------------------------------------

    if uploaded_file is not None:

        if (
            st.session_state.dataset_name
            != uploaded_file.name
        ):

            with st.spinner(
                "Processing knowledge base..."
            ):

                questions, answers = parse_dataset(
                    uploaded_file
                )

                if questions:

                    st.session_state.questions = questions

                    st.session_state.answers = answers

                    st.session_state.dataset_name = (
                        uploaded_file.name
                    )

                    # TF-IDF

                    vectorizer, matrix = (
                        prepare_tfidf(
                            tuple(questions)
                        )
                    )

                    st.session_state.vectorizer = (
                        vectorizer
                    )

                    st.session_state.tfidf_matrix = (
                        matrix
                    )

                    # Semantic embeddings

                    st.session_state.semantic_embeddings = (
                        create_embeddings(
                            tuple(questions)
                        )
                    )

                    st.session_state.dataset_loaded = True

                    st.success(
                        f"{len(questions)} entries loaded"
                    )

                else:

                    st.error(
                        "No valid Question | Answer "
                        "entries found."
                    )


    # --------------------------------------------------------
    # Dataset statistics
    # --------------------------------------------------------

    if st.session_state.dataset_loaded:

        st.metric(
            "Knowledge Entries",
            len(
                st.session_state.questions
            )
        )

        st.caption(
            st.session_state.dataset_name
        )

    else:

        st.info(
            "Upload dataset.txt to activate "
            "the knowledge base."
        )


    st.divider()


    # --------------------------------------------------------
    # Conversation
    # --------------------------------------------------------

    st.subheader(
        "Conversation"
    )

    st.caption(
        f"{len(st.session_state.messages)} messages"
    )

    if st.button(
        "Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


    st.divider()


    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    st.subheader(
        "System Status"
    )

    if st.session_state.dataset_loaded:

        st.success(
            "Knowledge Base: Ready"
        )

    else:

        st.warning(
            "Knowledge Base: Waiting"
        )


    if get_gemini_client():

        st.success(
            "Generative AI: Connected"
        )

    else:

        st.warning(
            "Generative AI: API key missing"
        )


# ============================================================
# HERO
# ============================================================

if page == "AI Assistant":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-content">

                <div class="hero-title">
                    IntelliMind AI
                </div>

                <div class="hero-subtitle">
                    An intelligent knowledge assistant
                    powered by semantic search,
                    natural language processing and
                    generative AI.
                </div>

                <div class="hero-badge">
                    AI SYSTEM READY
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# AI ASSISTANT PAGE
# ============================================================

if page == "AI Assistant":

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Knowledge Entries",
            len(
                st.session_state.questions
            )
        )

    with col2:

        st.metric(
            "Messages",
            len(
                st.session_state.messages
            )
        )

    with col3:

        st.metric(
            "Search Engine",
            "Hybrid"
        )

    with col4:

        st.metric(
            "AI Model",
            "Gemini"
        )


    st.write("")


    # --------------------------------------------------------
    # Welcome
    # --------------------------------------------------------

    if not st.session_state.messages:

        st.subheader(
            "How can I help you?"
        )

        st.write(
            "Ask a question about AI, ML, DL, NLP, "
            "Computer Vision, Python or any topic "
            "included in your knowledge base."
        )


        st.write("")


        # Suggested questions

        st.write(
            "**Try asking:**"
        )

        suggestions = [
            "What is AI?",
            "What is Machine Learning?",
            "What is Deep Learning?",
            "What is NLP?",
            "What is Computer Vision?",
            "What is an LLM?"
        ]


        s1, s2, s3 = st.columns(3)


        with s1:

            if st.button(
                suggestions[0],
                use_container_width=True
            ):

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": suggestions[0]
                    }
                )

                st.rerun()


        with s2:

            if st.button(
                suggestions[1],
                use_container_width=True
            ):

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": suggestions[1]
                    }
                )

                st.rerun()


        with s3:

            if st.button(
                suggestions[2],
                use_container_width=True
            ):

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": suggestions[2]
                    }
                )

                st.rerun()


        s4, s5, s6 = st.columns(3)


        with s4:

            if st.button(
                suggestions[3],
                use_container_width=True
            ):

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": suggestions[3]
                    }
                )

                st.rerun()


        with s5:

            if st.button(
                suggestions[4],
                use_container_width=True
            ):

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": suggestions[4]
                    }
                )

                st.rerun()


        with s6:

            if st.button(
                suggestions[5],
                use_container_width=True
            ):

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": suggestions[5]
                    }
                )

                st.rerun()


    # --------------------------------------------------------
    # Display previous messages
    # --------------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # --------------------------------------------------------
    # Chat Input
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask IntelliMind AI anything..."
    )


    if question:

        # ----------------------------------------------------
        # User message
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        with st.chat_message("user"):

            st.markdown(
                question
            )


        # ----------------------------------------------------
        # AI response
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Thinking..."
            ):

                if (
                    st.session_state.dataset_loaded
                ):

                    matched_question, matched_answer, score = (
                        find_best_answer(
                            question
                        )
                    )

                else:

                    matched_question = None

                    matched_answer = None

                    score = 0.0


                # --------------------------------------------
                # Response strategy
                # --------------------------------------------

                if (
                    matched_answer
                    and score >= 0.58
                ):

                    # Strong dataset match

                    answer = matched_answer


                elif (
                    matched_answer
                    and score >= 0.38
                ):

                    # Medium match
                    # Let Gemini improve it

                    answer = generate_ai_answer(

                        question,

                        matched_question,

                        matched_answer,

                        score
                    )


                else:

                    # Weak/no dataset match
                    # Gemini fallback

                    answer = generate_ai_answer(

                        question,

                        None,

                        None,

                        0.0
                    )


                st.markdown(
                    answer
                )


                # --------------------------------------------
                # Search information
                # --------------------------------------------

                if matched_question:

                    with st.expander(
                        "Knowledge match details"
                    ):

                        st.write(
                            "**Matched Question:**"
                        )

                        st.write(
                            matched_question
                        )

                        st.write(
                            f"**Similarity:** "
                            f"{score:.2%}"
                        )


        # ----------------------------------------------------
        # Save assistant message
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


# ============================================================
# KNOWLEDGE BASE PAGE
# ============================================================

elif page == "Knowledge Base":

    st.title(
        "Knowledge Base"
    )

    st.caption(
        "Manage and inspect your uploaded AI knowledge."
    )


    if not st.session_state.dataset_loaded:

        st.info(
            "No dataset is loaded yet. "
            "Upload dataset.txt from the sidebar."
        )

    else:

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Total Entries",
                len(
                    st.session_state.questions
                )
            )

        with col2:

            st.metric(
                "Dataset",
                st.session_state.dataset_name
            )

        with col3:

            st.metric(
                "Search",
                "Semantic + TF-IDF"
            )


        st.divider()


        st.subheader(
            "Dataset Preview"
        )


        # Show first 20 entries

        for i, (
            question,
            answer
        ) in enumerate(
            zip(
                st.session_state.questions[:20],
                st.session_state.answers[:20]
            ),
            start=1
        ):

            with st.expander(
                f"{i}. {question}"
            ):

                st.write(
                    answer
                )


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "About":

    st.title(
        "About IntelliMind AI"
    )

    st.write(
        """
        **IntelliMind AI** is an intelligent educational
        knowledge assistant that combines a custom
        knowledge base with semantic search and
        Generative AI.
        """
    )


    st.divider()


    st.subheader(
        "Core Technologies"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            """
            **Natural Language Processing**

            - Text preprocessing
            - Query expansion
            - TF-IDF
            - Semantic similarity
            - Sentence embeddings
            """
        )


    with col2:

        st.markdown(
            """
            **Artificial Intelligence**

            - Gemini Generative AI
            - Semantic Search
            - Knowledge Base Retrieval
            - AI Question Answering
            - Hybrid Retrieval
            """
        )


    st.divider()


    st.subheader(
        "Short Forms"
    )


    for short, full in SHORT_FORMS.items():

        st.write(
            f"**{short.upper()}** → {full.title()}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        margin-top:50px;
        padding:20px;
        color:#718096;
        font-size:12px;
    ">
        IntelliMind AI • Semantic Search • NLP • Generative AI
    </div>
    """,
    unsafe_allow_html=True
)
