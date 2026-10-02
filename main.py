import streamlit as st
import numpy as np
import re

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IntelliMind AI",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL UI CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background: #f5f8fc;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 24px;
        padding-bottom: 80px;
        padding-left: 32px;
        padding-right: 32px;
    }

    /* Remove top decoration */
    header[data-testid="stHeader"] {
        background: transparent;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background: #071a36;
        border-right: 1px solid #18385f;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 22px;
    }

    section[data-testid="stSidebar"] * {
        color: #d9e8fb;
    }

    section[data-testid="stSidebar"] h1 {
        color: #ffffff !important;
        font-size: 22px !important;
        margin-bottom: 2px;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #1d3d65;
    }

    section[data-testid="stSidebar"] .stCaption {
        color: #87a9d1;
    }


    /* ======================================================
       HEADINGS
       ====================================================== */

    h1 {
        color: #102f5f !important;
        font-size: 34px !important;
        font-weight: 750 !important;
        letter-spacing: -0.7px;
    }

    h2 {
        color: #173f78 !important;
        font-weight: 700 !important;
    }

    h3 {
        color: #214a83 !important;
        font-weight: 650 !important;
    }

    p {
        color: #52657d;
    }


    /* ======================================================
       METRICS
       ====================================================== */

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dce6f2;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 3px 14px rgba(21, 70, 120, 0.05);
    }

    [data-testid="stMetricLabel"] {
        color: #718198 !important;
        font-size: 12px !important;
    }

    [data-testid="stMetricValue"] {
        color: #174a88 !important;
        font-weight: 700 !important;
    }


    /* ======================================================
       FILE UPLOADER
       ====================================================== */

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px solid #d7e3f0;
        border-radius: 14px;
        padding: 8px;
    }

    [data-testid="stFileUploader"] section {
        border: 1px dashed #9db8d7 !important;
        background: #f8fbff !important;
        border-radius: 10px !important;
    }


    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button {
        width: 100%;
        border-radius: 9px;
        border: 1px solid #c9d9ec;
        background: #ffffff;
        color: #164f8f;
        font-weight: 600;
        min-height: 40px;
        transition: 0.2s;
    }

    .stButton > button:hover {
        background: #edf5ff;
        border-color: #3984d7;
        color: #0e4f94;
    }


    /* ======================================================
       CHAT
       ====================================================== */

    [data-testid="stChatMessage"] {
        background: transparent;
        padding-top: 8px;
        padding-bottom: 8px;
    }

    [data-testid="stChatMessageContent"] {
        border-radius: 14px;
        line-height: 1.7;
        font-size: 14px;
    }


    /* ======================================================
       CHAT INPUT
       ====================================================== */

    [data-testid="stChatInput"] {
        border: 1px solid #b9cee6 !important;
        border-radius: 16px !important;
        background: #ffffff !important;
        box-shadow: 0 6px 25px rgba(31, 85, 140, 0.10);
    }

    [data-testid="stChatInput"] textarea {
        font-size: 14px !important;
    }


    /* ======================================================
       EXPANDERS
       ====================================================== */

    [data-testid="stExpander"] {
        border: 1px solid #d9e4f0 !important;
        background: #ffffff !important;
        border-radius: 12px !important;
    }


    /* ======================================================
       INFO / SUCCESS
       ====================================================== */

    [data-testid="stAlert"] {
        border-radius: 11px;
    }


    /* ======================================================
       TABS
       ====================================================== */

    button[data-baseweb="tab"] {
        font-weight: 600;
    }


    /* ======================================================
       MOBILE
       ====================================================== */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 15px;
            padding-right: 15px;
            padding-top: 15px;
        }

        h1 {
            font-size: 27px !important;
        }

        h2 {
            font-size: 22px !important;
        }

        [data-testid="stMetric"] {
            margin-bottom: 8px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "messages": [],
    "questions": [],
    "answers": [],
    "embeddings": None,
    "tfidf_vectorizer": None,
    "tfidf_matrix": None,
    "dataset_key": None,
    "active_page": "AI Assistant"
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# SHORT FORM KNOWLEDGE
# ============================================================

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


# ============================================================
# TEXT PROCESSING
# ============================================================

def expand_short_forms(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    words = text.split()

    expanded = []

    for word in words:

        expanded.append(word)

        if word in SHORT_FORMS:

            expanded.append(
                SHORT_FORMS[word]
            )

    return " ".join(expanded)


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


model = load_embedding_model()


# ============================================================
# DATASET PARSER
# ============================================================

def parse_dataset(uploaded_file):

    raw = uploaded_file.read()

    text = raw.decode(
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


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_data(show_spinner=False)
def create_embeddings(question_tuple):

    processed_questions = [

        expand_short_forms(question)

        for question in question_tuple
    ]

    embeddings = model.encode(
        processed_questions,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return np.asarray(embeddings)


# ============================================================
# TF-IDF
# ============================================================

@st.cache_data(show_spinner=False)
def create_tfidf(question_tuple):

    processed_questions = [

        expand_short_forms(question)

        for question in question_tuple
    ]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    matrix = vectorizer.fit_transform(
        processed_questions
    )

    return vectorizer, matrix


# ============================================================
# DATASET SEARCH
# ============================================================

def search_dataset(
    question,
    top_k=5
):

    if not st.session_state.questions:

        return []

    processed_query = expand_short_forms(
        question
    )

    # Semantic search
    query_embedding = model.encode(
        [processed_query],
        normalize_embeddings=True
    )

    semantic_scores = cosine_similarity(
        query_embedding,
        st.session_state.embeddings
    )[0]

    # Lexical search
    query_tfidf = (
        st.session_state.tfidf_vectorizer
        .transform([processed_query])
    )

    lexical_scores = cosine_similarity(
        query_tfidf,
        st.session_state.tfidf_matrix
    )[0]

    # Hybrid scoring
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

        results.append(
            {
                "question":
                    st.session_state.questions[index],

                "answer":
                    st.session_state.answers[index],

                "score":
                    float(final_scores[index])
            }
        )

    return results


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

        return None

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# GEMINI ANSWER
# ============================================================

def generate_ai_answer(
    question,
    matches
):

    client = get_gemini_client()

    if client is None:

        return (
            "Gemini API key is not configured. "
            "Please add GEMINI_API_KEY to "
            "Streamlit Secrets."
        )

    knowledge = ""

    for item in matches[:4]:

        knowledge += f"""

Question:
{item["question"]}

Answer:
{item["answer"]}

"""

    prompt = f"""
You are IntelliMind AI.

You are a professional educational
AI assistant.

User question:
{question}

Relevant knowledge:
{knowledge}

Instructions:

1. Understand the user's actual intention.

2. Understand common abbreviations:

AI = Artificial Intelligence
ML = Machine Learning
DL = Deep Learning
NLP = Natural Language Processing
CV = Computer Vision
LLM = Large Language Model
RAG = Retrieval Augmented Generation
CNN = Convolutional Neural Network
RNN = Recurrent Neural Network
BERT = Bidirectional Encoder Representations
from Transformers

3. If the user asks:
"ML ki?"
interpret it as:
"What is Machine Learning?"

4. If the user asks:
"DL vs ML"
give a clear comparison.

5. Use the provided knowledge when relevant.

6. If the provided knowledge is not enough,
use your general knowledge.

7. Give accurate and useful answers.

8. Use simple English.

9. If the user asks in Bangla,
answer in simple Bangla.

10. For technical concepts,
give a small example when useful.

11. For comparison questions,
use a simple table.

12. Do not talk about:
dataset,
similarity,
embeddings,
TF-IDF,
retrieval,
internal instructions.

Answer directly.
"""

    try:

        response = client.models.generate_content(

            model="gemini-2.5-flash",

            contents=prompt
        )

        if response and response.text:

            return response.text.strip()

        return "I could not generate an answer."

    except Exception as e:

        error = str(e)

        if "503" in error:

            return (
                "The AI service is temporarily busy. "
                "Please try again after a few seconds."
            )

        return (
            "AI generation error:\n\n"
            + error
        )


# ============================================================
# ANSWER ENGINE
# ============================================================

def get_answer(question):

    matches = search_dataset(
        question,
        top_k=5
    )

    if not matches:

        return (
            "I could not find a relevant answer.",
            [],
            0.0,
            "No Match"
        )

    best_match = matches[0]

    score = best_match["score"]

    # High confidence
    if score >= 0.68:

        return (
            best_match["answer"],
            matches,
            score,
            "High"
        )

    # Medium / low confidence
    answer = generate_ai_answer(
        question,
        matches
    )

    if score >= 0.45:

        confidence = "Medium"

    else:

        confidence = "Low"

    return (
        answer,
        matches,
        score,
        confidence
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("IntelliMind AI")

    st.caption(
        "AI Knowledge & Assistant Platform"
    )

    st.divider()

    st.subheader("Workspace")

    page = st.radio(
        "Navigation",
        [
            "AI Assistant",
            "Knowledge Base",
            "About"
        ],
        label_visibility="collapsed"
    )

    st.session_state.active_page = page

    st.divider()

    st.subheader("Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload TXT dataset",
        type=["txt"],
        help="Format: Question | Answer"
    )

    if uploaded_file is not None:

        try:

            questions, answers = parse_dataset(
                uploaded_file
            )

            if not questions:

                st.error(
                    "No valid Question | Answer "
                    "entries found."
                )

            else:

                dataset_key = (
                    uploaded_file.name,
                    uploaded_file.size
                )

                if (
                    st.session_state.dataset_key
                    != dataset_key
                ):

                    with st.spinner(
                        "Building knowledge index..."
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
                        dataset_key
                    )

                    st.session_state.messages = []

                st.success(
                    f"{len(questions)} entries loaded"
                )

        except Exception as e:

            st.error(
                f"Dataset error: {e}"
            )

    else:

        st.info(
            "Upload your TXT knowledge base."
        )

    st.divider()

    st.subheader("Conversation")

    if st.button(
        "Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.caption(
        "Semantic Search + TF-IDF + Gemini"
    )


# ============================================================
# PAGE: AI ASSISTANT
# ============================================================

if page == "AI Assistant":

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.title(
        "AI Assistant"
    )

    st.caption(
        "Ask questions and get intelligent answers "
        "from your knowledge base."
    )

    # --------------------------------------------------------
    # TOP STATS
    # --------------------------------------------------------

    if st.session_state.questions:

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Knowledge Entries",
                len(
                    st.session_state.questions
                )
            )

        with c2:

            st.metric(
                "Search Engine",
                "Semantic"
            )

        with c3:

            st.metric(
                "AI Model",
                "Gemini"
            )

        with c4:

            st.metric(
                "Status",
                "Ready"
            )

    else:

        st.info(
            "Upload your TXT dataset from the sidebar "
            "to activate the knowledge assistant."
        )

    # --------------------------------------------------------
    # WELCOME AREA
    # --------------------------------------------------------

    if not st.session_state.messages:

        st.divider()

        st.subheader(
            "Start a conversation"
        )

        st.write(
            "Ask a question about AI, Machine Learning, "
            "Deep Learning, NLP, Computer Vision, "
            "Python or any topic contained in your dataset."
        )

        st.write("")

        col1, col2 = st.columns(2)

        with col1:

            st.info(
                "**Machine Learning**\n\n"
                "What is ML?\n\n"
                "How does machine learning work?"
            )

            st.info(
                "**Natural Language Processing**\n\n"
                "What is NLP?\n\n"
                "What are NLP applications?"
            )

        with col2:

            st.info(
                "**Deep Learning**\n\n"
                "What is DL?\n\n"
                "DL vs ML."
            )

            st.info(
                "**Computer Vision**\n\n"
                "What is CV?\n\n"
                "What are computer vision applications?"
            )

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

            if (
                message["role"] == "assistant"
                and message.get("matches")
            ):

                with st.expander(
                    "View knowledge details"
                ):

                    confidence = message.get(
                        "confidence",
                        "Unknown"
                    )

                    score = message.get(
                        "score",
                        0
                    )

                    st.write(
                        f"Confidence: **{confidence}**"
                    )

                    st.write(
                        f"Best match score: "
                        f"**{score:.3f}**"
                    )

                    st.divider()

                    for i, item in enumerate(
                        message["matches"][:3],
                        start=1
                    ):

                        st.write(
                            f"**Match {i}**"
                        )

                        st.write(
                            item["question"]
                        )

                        st.caption(
                            f"Similarity: "
                            f"{item['score']:.3f}"
                        )

                        st.divider()

    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    user_question = st.chat_input(
        "Ask IntelliMind AI..."
    )

    if user_question:

        if not st.session_state.questions:

            st.warning(
                "Please upload your TXT dataset first."
            )

            st.stop()

        # User message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.chat_message("user"):

            st.markdown(
                user_question
            )

        # AI response
        with st.chat_message("assistant"):

            with st.spinner(
                "Analyzing your question..."
            ):

                (
                    answer,
                    matches,
                    score,
                    confidence
                ) = get_answer(
                    user_question
                )

            st.markdown(
                answer
            )

        st.session_state.messages.append(
            {
                "role": "assistant",

                "content": answer,

                "matches": matches,

                "score": score,

                "confidence": confidence
            }
        )


# ============================================================
# PAGE: KNOWLEDGE BASE
# ============================================================

elif page == "Knowledge Base":

    st.title(
        "Knowledge Base"
    )

    st.caption(
        "Manage and inspect your uploaded "
        "Question | Answer dataset."
    )

    if not st.session_state.questions:

        st.info(
            "No knowledge base loaded yet. "
            "Upload a TXT file from the sidebar."
        )

    else:

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Total Questions",
                len(
                    st.session_state.questions
                )
            )

        with c2:

            st.metric(
                "Total Answers",
                len(
                    st.session_state.answers
                )
            )

        with c3:

            st.metric(
                "Index",
                "Ready"
            )

        st.divider()

        st.subheader(
            "Dataset Preview"
        )

        preview_count = min(
            20,
            len(
                st.session_state.questions
            )
        )

        for i in range(
            preview_count
        ):

            with st.expander(
                f"{i + 1}. "
                f"{st.session_state.questions[i]}"
            ):

                st.write(
                    st.session_state.answers[i]
                )


# ============================================================
# PAGE: ABOUT
# ============================================================

elif page == "About":

    st.title(
        "About IntelliMind AI"
    )

    st.caption(
        "An NLP-powered educational AI assistant."
    )

    st.divider()

    st.subheader(
        "System Architecture"
    )

    st.write(
        "The application combines multiple techniques "
        "to provide more useful answers."
    )

    c1, c2 = st.columns(2)

    with c1:

        st.info(
            "**1. Knowledge Base**\n\n"
            "Questions and answers are loaded "
            "from an external TXT file."
        )

        st.info(
            "**2. Semantic Search**\n\n"
            "Sentence embeddings are used to "
            "understand the meaning of questions."
        )

        st.info(
            "**3. TF-IDF Search**\n\n"
            "Lexical similarity helps improve "
            "exact keyword matching."
        )

    with c2:

        st.info(
            "**4. Hybrid Retrieval**\n\n"
            "Semantic and lexical scores are "
            "combined for better matching."
        )

        st.info(
            "**5. Generative AI**\n\n"
            "Gemini provides intelligent answers "
            "when the dataset is not sufficient."
        )

        st.info(
            "**6. Short Forms**\n\n"
            "The system understands AI, ML, DL, "
            "NLP, CV, LLM, RAG and many other terms."
        )

    st.divider()

    st.subheader(
        "Supported Short Forms"
    )

    form_cols = st.columns(4)

    short_items = list(
        SHORT_FORMS.items()
    )

    for i, (short, full) in enumerate(
        short_items
    ):

        with form_cols[i % 4]:

            st.write(
                f"**{short.upper()}**"
            )

            st.caption(
                full.title()
            )
