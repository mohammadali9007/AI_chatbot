import streamlit as st
import os
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

if "questions" not in st.session_state:
    st.session_state.questions = []

if "answers" not in st.session_state:
    st.session_state.answers = []

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = ""

if "loaded_file" not in st.session_state:
    st.session_state.loaded_file = ""

if "semantic_embeddings" not in st.session_state:
    st.session_state.semantic_embeddings = None

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None


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
        background: #f6f8fc;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 25px;
        padding-bottom: 60px;
    }

    h1, h2, h3, h4 {
        color: #102a43;
    }

    p {
        color: #52667a;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background: #0b1f3a;
        border-right: 1px solid #dbe5f0;
    }

    section[data-testid="stSidebar"] * {
        color: #dce8f5;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: white !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.12);
    }


    /* Sidebar selectbox */

    section[data-testid="stSidebar"]
    div[data-baseweb="select"] > div {
        background: #132d50 !important;
        border: 1px solid #29486c !important;
        border-radius: 8px !important;
    }


    /* Sidebar uploader */

    section[data-testid="stSidebar"]
    [data-testid="stFileUploader"] {
        background: #102846 !important;
        border: 1px solid #284665 !important;
        border-radius: 10px !important;
    }

    section[data-testid="stSidebar"]
    [data-testid="stFileUploader"] section {
        background: #0e2440 !important;
        border: 1px dashed #47719c !important;
        border-radius: 8px !important;
    }


    /* Sidebar metrics */

    section[data-testid="stSidebar"]
    [data-testid="stMetric"] {
        background: #102846 !important;
        border: 1px solid #284665 !important;
        border-radius: 10px !important;
    }

    section[data-testid="stSidebar"]
    [data-testid="stMetricLabel"] {
        color: #8fa9c4 !important;
    }

    section[data-testid="stSidebar"]
    [data-testid="stMetricValue"] {
        color: white !important;
    }


    /* Sidebar buttons */

    section[data-testid="stSidebar"]
    .stButton > button {
        background: #132d50 !important;
        color: #dce8f5 !important;
        border: 1px solid #29486c !important;
        border-radius: 8px !important;
    }

    section[data-testid="stSidebar"]
    .stButton > button:hover {
        background: #19406c !important;
        border-color: #4c8dcc !important;
    }


    /* ======================================================
       HEADER
       ====================================================== */

    .app-header {
        background: white;
        border: 1px solid #e0e7ef;
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 20px;
        box-shadow: 0 4px 18px rgba(15, 50, 90, 0.05);
    }

    .app-title {
        font-size: 28px;
        font-weight: 750;
        color: #12365c;
        margin-bottom: 5px;
    }

    .app-subtitle {
        font-size: 13px;
        color: #718096;
    }


    /* ======================================================
       STATUS
       ====================================================== */

    .status {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: #effaf3;
        border: 1px solid #cdebd8;
        border-radius: 20px;
        padding: 6px 12px;
        font-size: 11px;
        color: #24734a;
        font-weight: 600;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #2e9b5b;
    }


    /* ======================================================
       STAT CARDS
       ====================================================== */

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e1e8f0;
        border-radius: 12px;
        padding: 14px;
        box-shadow: 0 3px 14px rgba(20, 55, 90, 0.04);
    }

    [data-testid="stMetricLabel"] {
        color: #718096 !important;
    }

    [data-testid="stMetricValue"] {
        color: #155da5 !important;
        font-weight: 700 !important;
    }


    /* ======================================================
       CHAT
       ====================================================== */

    [data-testid="stChatMessage"] {
        border: none !important;
        background: transparent !important;
        padding-top: 7px;
        padding-bottom: 7px;
    }

    [data-testid="stChatMessageContent"] {
        border-radius: 12px !important;
        font-size: 14px;
        line-height: 1.7;
    }


    /* ======================================================
       CHAT INPUT
       ====================================================== */

    [data-testid="stChatInput"] {
        background: white !important;
        border: 1px solid #cbd9e8 !important;
        border-radius: 13px !important;
        box-shadow: 0 5px 20px rgba(20, 60, 100, 0.08);
    }


    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button {
        background: white !important;
        color: #185b9f !important;
        border: 1px solid #d3dfeb !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    .stButton > button:hover {
        background: #f1f7fd !important;
        border-color: #5891c9 !important;
    }


    /* ======================================================
       EXPANDER
       ====================================================== */

    [data-testid="stExpander"] {
        background: white !important;
        border: 1px solid #dce5ee !important;
        border-radius: 10px !important;
    }


    /* ======================================================
       MOBILE
       ====================================================== */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 12px;
            padding-right: 12px;
        }

        .app-title {
            font-size: 23px;
        }

        .app-header {
            padding: 18px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text).lower()

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
# SHORT FORMS
# ============================================================

SHORT_FORMS = {

    "ai": "artificial intelligence",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "cnn": "convolutional neural network",
    "rnn": "recurrent neural network",
    "lstm": "long short term memory",
    "llm": "large language model",
    "rag": "retrieval augmented generation",
    "api": "application programming interface",
    "svm": "support vector machine",
    "knn": "k nearest neighbors",
    "gan": "generative adversarial network",
    "gpu": "graphics processing unit",
    "cpu": "central processing unit"
}


def expand_short_forms(text):

    words = text.lower().split()

    result = []

    for word in words:

        clean_word = re.sub(
            r"[^a-z0-9]",
            "",
            word
        )

        if clean_word in SHORT_FORMS:

            result.append(
                SHORT_FORMS[clean_word]
            )

        result.append(word)

    return " ".join(result)


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# ============================================================
# BUILD INDEX
# ============================================================

def build_index(questions):

    model = load_model()

    expanded_questions = [
        expand_short_forms(q)
        for q in questions
    ]

    embeddings = model.encode(
        expanded_questions,
        normalize_embeddings=True,
        show_progress_bar=False
    )


    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    tfidf_matrix = vectorizer.fit_transform(
        [
            clean_text(q)
            for q in questions
        ]
    )

    return (
        embeddings,
        vectorizer,
        tfidf_matrix
    )


# ============================================================
# PARSE DATASET
# ============================================================

def parse_dataset(file):

    content = file.getvalue().decode(
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


# ============================================================
# SEARCH
# ============================================================

def search_knowledge(question):

    if not st.session_state.questions:

        return None, None, 0.0


    model = load_model()

    query = expand_short_forms(
        question
    )

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False
    )


    semantic_scores = cosine_similarity(
        query_embedding,
        st.session_state.semantic_embeddings
    )[0]


    query_vector = (
        st.session_state.vectorizer
        .transform(
            [clean_text(query)]
        )
    )


    tfidf_scores = cosine_similarity(
        query_vector,
        st.session_state.tfidf_matrix
    )[0]


    final_scores = (
        semantic_scores * 0.75
        +
        tfidf_scores * 0.25
    )


    index = int(
        np.argmax(final_scores)
    )

    score = float(
        final_scores[index]
    )


    return (
        st.session_state.questions[index],
        st.session_state.answers[index],
        score
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
# GEMINI ANSWER
# ============================================================

def generate_ai_answer(
    question,
    matched_question=None,
    matched_answer=None
):

    client = get_gemini_client()

    if client is None:

        if matched_answer:

            return matched_answer

        return (
            "Gemini API key is not configured. "
            "Please add GEMINI_API_KEY to "
            "Streamlit Secrets."
        )


    if matched_answer:

        knowledge = f"""
Question:
{matched_question}

Answer:
{matched_answer}
"""

    else:

        knowledge = (
            "No reliable matching answer "
            "was found in the knowledge base."
        )


    prompt = f"""
You are IntelliMind AI, a helpful educational AI assistant.

User Question:
{question}

Relevant Knowledge:
{knowledge}

Instructions:

- Understand the user's real question.
- Give a direct and accurate answer.
- If relevant knowledge is available, use it.
- You may rewrite the knowledge in a clearer way.
- Do not mention similarity scores or internal retrieval.
- Do not invent information.
- Keep simple questions simple.
- For technical questions, explain for a beginner.
- If the user asks a short form such as ML, DL, NLP,
  CV, AI, LLM or RAG, give the full form and explanation.
- Use examples when useful.

Answer naturally.
"""


    try:

        response = client.models.generate_content(

            model="gemini-2.5-flash",

            contents=prompt
        )

        if response and response.text:

            return response.text.strip()

        return matched_answer or (
            "I could not generate an answer."
        )

    except Exception as e:

        if matched_answer:

            return matched_answer

        if "503" in str(e):

            return (
                "The AI service is temporarily busy. "
                "Please try again shortly."
            )

        return (
            "AI generation failed. "
            "Please check your Gemini API key."
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding:8px 2px 15px 2px;">

            <div style="
                font-size:22px;
                font-weight:750;
                color:white;
            ">
                IntelliMind AI
            </div>

            <div style="
                font-size:10px;
                color:#7f9bbb;
                margin-top:4px;
                letter-spacing:1px;
            ">
                KNOWLEDGE ASSISTANT
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    page = st.selectbox(
        "Navigation",
        [
            "AI Assistant",
            "Knowledge Base",
            "About"
        ]
    )


    st.divider()


    # ========================================================
    # KNOWLEDGE BASE
    # ========================================================

    st.subheader(
        "Knowledge Base"
    )

    uploaded_file = st.file_uploader(
        "Upload dataset.txt",
        type=["txt"]
    )


    if uploaded_file:

        if (
            st.session_state.loaded_file
            != uploaded_file.name
        ):

            with st.spinner(
                "Loading knowledge..."
            ):

                try:

                    questions, answers = parse_dataset(
                        uploaded_file
                    )

                    if questions:

                        (
                            embeddings,
                            vectorizer,
                            tfidf_matrix
                        ) = build_index(
                            questions
                        )

                        st.session_state.questions = questions

                        st.session_state.answers = answers

                        st.session_state.semantic_embeddings = embeddings

                        st.session_state.vectorizer = vectorizer

                        st.session_state.tfidf_matrix = tfidf_matrix

                        st.session_state.dataset_name = uploaded_file.name

                        st.session_state.loaded_file = uploaded_file.name

                        st.success(
                            f"{len(questions)} entries loaded"
                        )

                    else:

                        st.error(
                            "No valid data found."
                        )

                except Exception as e:

                    st.error(
                        f"Dataset error: {e}"
                    )


    if st.session_state.questions:

        st.caption(
            st.session_state.dataset_name
        )

        st.metric(
            "Entries",
            len(
                st.session_state.questions
            )
        )

    else:

        st.info(
            "Upload your Question | Answer TXT file."
        )


    st.divider()


    # ========================================================
    # CONVERSATION
    # ========================================================

    st.subheader(
        "Conversation"
    )

    st.caption(
        f"{len(st.session_state.messages)} messages"
    )


    if st.button(
        "Clear conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


    st.divider()


    # ========================================================
    # SYSTEM STATUS
    # ========================================================

    st.subheader(
        "System Status"
    )


    if st.session_state.questions:

        st.success(
            "Knowledge Base Ready"
        )

    else:

        st.warning(
            "Knowledge Base Waiting"
        )


    if get_gemini_client():

        st.success(
            "Gemini Connected"
        )

    else:

        st.warning(
            "Gemini API Missing"
        )


# ============================================================
# AI ASSISTANT PAGE
# ============================================================

if page == "AI Assistant":

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="app-header">

            <div class="app-title">
                IntelliMind AI
            </div>

            <div class="app-subtitle">
                Intelligent knowledge assistant powered by
                semantic search and generative AI.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Knowledge",
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
            "Retrieval",
            "Hybrid"
        )


    with col4:

        st.markdown(
            """
            <div style="
                background:white;
                border:1px solid #e1e8f0;
                border-radius:12px;
                padding:16px;
                height:100%;
            ">

                <div style="
                    font-size:12px;
                    color:#718096;
                ">
                    System
                </div>

                <div class="status" style="
                    margin-top:7px;
                ">

                    <div class="status-dot"></div>

                    Online

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")


    # --------------------------------------------------------
    # WELCOME
    # --------------------------------------------------------

    if not st.session_state.messages:

        st.subheader(
            "How can I help you?"
        )

        st.caption(
            "Ask a question from your knowledge base "
            "or ask a general AI question."
        )

        st.write("")


        q1, q2, q3 = st.columns(3)


        suggestions = [
            "What is AI?",
            "What is Machine Learning?",
            "What is NLP?"
        ]


        with q1:

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


        with q2:

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


        with q3:

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


    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask IntelliMind AI..."
    )


    if question:

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


        with st.chat_message("assistant"):

            with st.spinner(
                "Thinking..."
            ):

                matched_q = None
                matched_a = None
                score = 0.0


                if st.session_state.questions:

                    (
                        matched_q,
                        matched_a,
                        score
                    ) = search_knowledge(
                        question
                    )


                # --------------------------------------------
                # STRONG MATCH
                # --------------------------------------------

                if matched_a and score >= 0.58:

                    answer = matched_a


                # --------------------------------------------
                # MEDIUM MATCH
                # --------------------------------------------

                elif matched_a and score >= 0.35:

                    answer = generate_ai_answer(
                        question,
                        matched_q,
                        matched_a
                    )


                # --------------------------------------------
                # GENERAL AI
                # --------------------------------------------

                else:

                    answer = generate_ai_answer(
                        question
                    )


                st.markdown(
                    answer
                )


                if matched_q:

                    with st.expander(
                        "View knowledge match"
                    ):

                        st.write(
                            f"**Matched:** {matched_q}"
                        )

                        st.write(
                            f"**Similarity:** {score:.2%}"
                        )


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

    st.markdown(
        """
        <div class="app-header">

            <div class="app-title">
                Knowledge Base
            </div>

            <div class="app-subtitle">
                Manage and inspect your uploaded knowledge.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    if not st.session_state.questions:

        st.info(
            "Upload dataset.txt from the sidebar."
        )

    else:

        c1, c2, c3 = st.columns(3)


        with c1:

            st.metric(
                "Total Entries",
                len(
                    st.session_state.questions
                )
            )


        with c2:

            st.metric(
                "Dataset",
                st.session_state.dataset_name
            )


        with c3:

            st.metric(
                "Search Method",
                "Semantic + TF-IDF"
            )


        st.divider()


        st.subheader(
            "Knowledge Preview"
        )


        for i, (
            question,
            answer
        ) in enumerate(
            zip(
                st.session_state.questions[:30],
                st.session_state.answers[:30]
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

else:

    st.markdown(
        """
        <div class="app-header">

            <div class="app-title">
                About IntelliMind AI
            </div>

            <div class="app-subtitle">
                Intelligent NLP and Generative AI assistant.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.write(
        """
        IntelliMind AI combines a custom Question & Answer
        knowledge base with semantic search and Generative AI.
        It can understand natural questions and retrieve
        relevant information from the uploaded dataset.
        """
    )


    st.divider()


    c1, c2 = st.columns(2)


    with c1:

        st.subheader(
            "NLP & Retrieval"
        )

        st.write(
            """
            • Text preprocessing

            • TF-IDF

            • Sentence embeddings

            • Semantic similarity

            • Hybrid retrieval
            """
        )


    with c2:

        st.subheader(
            "Generative AI"
        )

        st.write(
            """
            • Gemini

            • Natural language understanding

            • AI question answering

            • Knowledge-grounded responses

            • Intelligent fallback
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        margin-top:50px;
        padding:15px;
        color:#8798aa;
        font-size:11px;
    ">
        IntelliMind AI • NLP • Semantic Search • Generative AI
    </div>
    """,
    unsafe_allow_html=True
)
