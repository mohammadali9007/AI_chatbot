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
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CLEAN BLUE AI UI
# =========================================================

st.markdown(
    """
    <style>

    /* ==============================
       GLOBAL
    ============================== */

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap'
    );

    html, body, [class*="css"] {
        font-family: "Inter", sans-serif;
    }

    .stApp {
        background:
            linear-gradient(
                135deg,
                #f8fbff 0%,
                #edf5ff 50%,
                #f8fbff 100%
            );
    }

    .block-container {
        max-width: 1050px;
        padding-top: 25px;
        padding-bottom: 120px;
    }


    /* ==============================
       SIDEBAR
    ============================== */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #071b3d 0%,
            #092b63 55%,
            #071c40 100%
        );

        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #dbeafe;
    }

    .brand {
        padding: 10px 0 20px 0;
        border-bottom: 1px solid rgba(255,255,255,0.12);
        margin-bottom: 20px;
    }

    .brand-name {
        color: white;
        font-size: 21px;
        font-weight: 700;
        margin-top: 5px;
    }

    .brand-subtitle {
        color: #8db7ee;
        font-size: 11px;
        margin-top: 3px;
    }

    .section-title {
        color: #72a8ed;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .dataset-status {
        background: rgba(37,99,235,0.18);
        border: 1px solid rgba(147,197,253,0.20);
        border-radius: 12px;
        padding: 12px;
        margin-top: 10px;
    }

    .dataset-status-title {
        color: #bfdbfe;
        font-size: 12px;
        font-weight: 600;
    }

    .dataset-status-text {
        color: #8db7ee;
        font-size: 10px;
        margin-top: 4px;
    }


    /* ==============================
       TOP HEADER
    ============================== */

    .header-box {
        text-align: center;
        padding: 20px 10px 28px 10px;
    }

    .header-title {
        color: #123c7a;
        font-size: 34px;
        font-weight: 700;
        letter-spacing: -1px;
    }

    .header-subtitle {
        color: #64748b;
        font-size: 13px;
        margin-top: 6px;
    }

    .status {
        display: inline-block;
        margin-top: 12px;
        padding: 5px 12px;
        border-radius: 20px;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #2563eb;
        font-size: 10px;
        font-weight: 600;
    }


    /* ==============================
       WELCOME
    ============================== */

    .welcome-title {
        text-align: center;
        color: #183e78;
        font-size: 21px;
        font-weight: 600;
        margin-top: 25px;
    }

    .welcome-text {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        max-width: 650px;
        margin: 8px auto 25px auto;
        line-height: 1.6;
    }


    /* ==============================
       PROMPT CARDS
    ============================== */

    .prompt-box {
        background: rgba(255,255,255,0.85);
        border: 1px solid #dbeafe;
        border-radius: 15px;
        padding: 17px;
        min-height: 105px;
        box-shadow: 0 5px 20px rgba(37,99,235,0.05);
    }

    .prompt-title {
        color: #1d4ed8;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 5px;
    }

    .prompt-text {
        color: #64748b;
        font-size: 11px;
        line-height: 1.5;
    }


    /* ==============================
       CHAT
    ============================== */

    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding-top: 7px;
        padding-bottom: 7px;
    }

    [data-testid="stChatMessageContent"] {
        font-size: 14px;
        line-height: 1.7;
    }


    /* ==============================
       CHAT INPUT
    ============================== */

    [data-testid="stChatInput"] {
        border-radius: 18px !important;
        border: 1px solid #bfdbfe !important;
        background: white !important;

        box-shadow:
            0 8px 30px rgba(37,99,235,0.12);
    }


    /* ==============================
       BUTTONS
    ============================== */

    .stButton > button {
        border-radius: 10px !important;
        border: 1px solid #bfdbfe !important;
        background: white !important;
        color: #1d4ed8 !important;
        font-size: 12px !important;
    }

    .stButton > button:hover {
        border-color: #2563eb !important;
        background: #eff6ff !important;
    }


    /* ==============================
       EXPANDER
    ============================== */

    [data-testid="stExpander"] {
        border: 1px solid #dbeafe !important;
        border-radius: 12px !important;
        background: #f8fbff !important;
    }


    /* ==============================
       MOBILE
    ============================== */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 14px;
            padding-right: 14px;
        }

        .header-title {
            font-size: 27px;
        }

        .header-subtitle {
            font-size: 12px;
        }

        .prompt-box {
            margin-bottom: 10px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


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
# SHORT FORMS
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
    "bert": "bidirectional encoder representations from transformers",
    "api": "application programming interface",
    "svm": "support vector machine",
    "knn": "k nearest neighbors",
    "rf": "random forest",
    "dt": "decision tree",
    "ann": "artificial neural network",
    "gan": "generative adversarial network",
    "ocr": "optical character recognition",
    "tts": "text to speech",
    "stt": "speech to text",
    "eda": "exploratory data analysis"
}


# =========================================================
# EXPAND SHORT FORMS
# =========================================================

def expand_short_forms(text):

    text = text.lower()

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
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


model = load_embedding_model()


# =========================================================
# PARSE DATASET
# =========================================================

def parse_dataset(uploaded_file):

    raw_data = uploaded_file.read()

    text = raw_data.decode(
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
# CREATE EMBEDDINGS
# =========================================================

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

    return np.array(embeddings)


# =========================================================
# CREATE TF-IDF
# =========================================================

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


# =========================================================
# SEARCH DATASET
# =========================================================

def search_dataset(
    user_question,
    top_k=5
):

    if not st.session_state.questions:
        return []

    query = expand_short_forms(
        user_question
    )

    # Semantic similarity
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    semantic_scores = cosine_similarity(
        query_embedding,
        st.session_state.embeddings
    )[0]

    # TF-IDF similarity
    query_tfidf = (
        st.session_state.tfidf_vectorizer
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

    for index in indices:

        results.append({
            "question":
                st.session_state.questions[index],

            "answer":
                st.session_state.answers[index],

            "score":
                float(final_scores[index])
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
# GEMINI ANSWER
# =========================================================

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

    context = ""

    for i, item in enumerate(
        matches[:4],
        start=1
    ):

        context += f"""

Knowledge {i}

Question:
{item["question"]}

Answer:
{item["answer"]}

"""

    prompt = f"""
You are IntelliMind AI, a professional
educational AI assistant.

User Question:
{question}

Relevant Knowledge:
{context}

Instructions:

1. Understand natural language questions.

2. Understand technical short forms:

AI = Artificial Intelligence
ML = Machine Learning
DL = Deep Learning
NLP = Natural Language Processing
CV = Computer Vision
LLM = Large Language Model
RAG = Retrieval Augmented Generation
CNN = Convolutional Neural Network
RNN = Recurrent Neural Network

3. If the user asks "ML ki?",
understand it as "What is Machine Learning?"

4. If the user asks "DL vs ML",
understand it as "Deep Learning vs Machine Learning."

5. Use the provided knowledge when relevant.

6. If the knowledge is not enough,
use your general knowledge.

7. Give a direct answer.

8. Keep explanations easy to understand.

9. For comparisons, use a small table.

10. Give examples for technical concepts
when useful.

11. Do not mention:
- dataset retrieval
- similarity score
- embeddings
- TF-IDF
- internal system
- prompt

12. If the user writes in Bangla,
you can answer in simple Bangla.

Now answer the user.
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
                "AI service is temporarily busy. "
                "Please try again in a few seconds."
            )

        return (
            "AI generation error:\n\n"
            + error
        )


# =========================================================
# ANSWER ENGINE
# =========================================================

def get_answer(question):

    matches = search_dataset(
        question,
        top_k=5
    )

    if not matches:

        return (
            "I couldn't find a relevant answer.",
            [],
            0
        )

    best_match = matches[0]

    score = best_match["score"]

    # High confidence
    if score >= 0.68:

        return (
            best_match["answer"],
            matches,
            score
        )

    # AI explanation / fallback
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

    # Brand
    st.markdown(
        '<div class="brand">'
        '<div class="brand-name">IntelliMind AI</div>'
        '<div class="brand-subtitle">'
        'Smart Knowledge Assistant'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # Knowledge base
    st.markdown(
        '<div class="section-title">'
        'Knowledge Base'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload TXT Dataset",
        type=["txt"],
        help="Format: Question | Answer"
    )

    if uploaded_file is not None:

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
                        "Processing knowledge base..."
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

                st.markdown(
                    f"""
                    <div class="dataset-status">

                        <div class="dataset-status-title">
                            Knowledge Base Ready
                        </div>

                        <div class="dataset-status-text">
                            {len(questions)}
                            Q&A entries loaded
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.error(
                    "No valid Question | Answer "
                    "entries found."
                )

        except Exception as e:

            st.error(
                f"Dataset error: {e}"
            )

    else:

        st.markdown(
            """
            <div class="dataset-status">

                <div class="dataset-status-title">
                    No Dataset
                </div>

                <div class="dataset-status-text">
                    Upload your Question | Answer
                    TXT file.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # Options
    st.markdown(
        '<div class="section-title">'
        'Options'
        '</div>',
        unsafe_allow_html=True
    )

    show_matches = st.checkbox(
        "Show knowledge matches",
        value=False
    )

    st.write("")

    if st.button(
        "New Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.markdown(
        """
        <div style="
            margin-top:25px;
            padding-top:15px;
            border-top:1px solid rgba(255,255,255,0.10);
            color:#719bd0;
            font-size:10px;
            line-height:1.7;
        ">
            IntelliMind AI<br>
            NLP • Semantic Search • Gemini
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HEADER
# =========================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div class="header-box">

            <div class="header-title">
                IntelliMind AI
            </div>

            <div class="header-subtitle">
                Intelligent answers powered by
                NLP, semantic search and Generative AI
            </div>

            <div class="status">
                AI SYSTEM ONLINE
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# WELCOME SCREEN
# =========================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div class="welcome-title">
            What can I help you with?
        </div>

        <div class="welcome-text">
            Ask your question naturally.
            IntelliMind can understand technical
            terms and short forms such as ML, DL,
            CV, NLP, AI and LLM.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="prompt-box">

                <div class="prompt-title">
                    Machine Learning
                </div>

                <div class="prompt-text">
                    What is ML?
                    <br>
                    Explain machine learning.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="prompt-box">

                <div class="prompt-title">
                    Computer Vision
                </div>

                <div class="prompt-text">
                    What is CV?
                    <br>
                    Give some CV applications.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="prompt-box">

                <div class="prompt-title">
                    Deep Learning
                </div>

                <div class="prompt-text">
                    What is DL?
                    <br>
                    DL vs ML.
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

    avatar = "AI" if role == "assistant" else "You"

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
                "Knowledge matches"
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

user_question = st.chat_input(
    "Message IntelliMind AI..."
)


if user_question:

    # Dataset check
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

    with st.chat_message(
        "user",
        avatar="You"
    ):

        st.markdown(
            user_question
        )

    # AI response
    with st.chat_message(
        "assistant",
        avatar="AI"
    ):

        with st.spinner(
            "Thinking..."
        ):

            answer, matches, score = (
                get_answer(
                    user_question
                )
            )

        st.markdown(
            answer
        )

        if (
            show_matches
            and matches
        ):

            with st.expander(
                "Knowledge matches"
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

                        {item["answer"]}
                        """
                    )

    # Save response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "matches": matches,
            "score": score
        }
    )
