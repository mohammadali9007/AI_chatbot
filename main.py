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
# CSS ONLY
# NO HTML
# =========================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #f4f8ff;
    }

    /* Main content width */
    .block-container {
        max-width: 1050px;
        padding-top: 35px;
        padding-bottom: 120px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #081d3d;
    }

    section[data-testid="stSidebar"] * {
        color: #dbeafe;
    }

    /* Sidebar buttons */
    section[data-testid="stSidebar"] button {
        background-color: #102e5c;
        border: 1px solid #244d83;
        color: white;
    }

    section[data-testid="stSidebar"] button:hover {
        border-color: #4f9cff;
        background-color: #163a70;
    }

    /* Main title */
    h1 {
        color: #123c7a !important;
        font-weight: 700 !important;
        letter-spacing: -1px;
    }

    h2, h3 {
        color: #183e78 !important;
    }

    /* Normal text */
    p {
        color: #475569;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: white;
        border-radius: 14px;
        padding: 8px;
        border: 1px solid #d6e5fa;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        border: 1px solid #b8d4f7;
        border-radius: 16px;
        background-color: white;
        box-shadow: 0 5px 20px rgba(30, 100, 200, 0.10);
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
    }

    /* Buttons */
    .stButton button {
        border-radius: 10px;
        border: 1px solid #b8d4f7;
        background-color: white;
        color: #1757a6;
        font-weight: 500;
    }

    .stButton button:hover {
        border-color: #2878d8;
        background-color: #edf5ff;
        color: #0d4f9e;
    }

    /* Expander */
    [data-testid="stExpander"] {
        border: 1px solid #d6e5fa;
        border-radius: 12px;
        background-color: white;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #d6e5fa;
        padding: 12px;
        border-radius: 12px;
    }

    /* Divider */
    hr {
        border-color: #dce9f8;
    }

    /* Mobile */
    @media (max-width: 768px) {

        .block-container {
            padding-left: 15px;
            padding-right: 15px;
            padding-top: 20px;
        }

        h1 {
            font-size: 28px !important;
        }

        [data-testid="stMetric"] {
            margin-bottom: 8px;
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

    expanded = []

    for word in words:

        expanded.append(word)

        if word in SHORT_FORMS:
            expanded.append(
                SHORT_FORMS[word]
            )

    return " ".join(expanded)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


model = load_model()


# =========================================================
# DATASET PARSER
# =========================================================

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


# =========================================================
# EMBEDDINGS
# =========================================================

@st.cache_data(show_spinner=False)
def create_embeddings(question_tuple):

    processed = [
        expand_short_forms(q)
        for q in question_tuple
    ]

    embeddings = model.encode(
        processed,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return np.asarray(embeddings)


# =========================================================
# TF-IDF
# =========================================================

@st.cache_data(show_spinner=False)
def create_tfidf(question_tuple):

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
# SEARCH
# =========================================================

def search_dataset(
    question,
    top_k=5
):

    if not st.session_state.questions:
        return []

    query = expand_short_forms(
        question
    )

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    semantic_scores = cosine_similarity(
        query_embedding,
        st.session_state.embeddings
    )[0]

    query_tfidf = (
        st.session_state.tfidf_vectorizer
        .transform([query])
    )

    lexical_scores = cosine_similarity(
        query_tfidf,
        st.session_state.tfidf_matrix
    )[0]

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
# GEMINI
# =========================================================

def generate_ai_answer(
    question,
    matches
):

    client = get_gemini_client()

    if client is None:

        return (
            "Gemini API key is not configured. "
            "Please add GEMINI_API_KEY in "
            "Streamlit Secrets."
        )

    knowledge = ""

    for item in matches[:4]:

        knowledge += (
            "\nQuestion: "
            + item["question"]
            + "\nAnswer: "
            + item["answer"]
            + "\n"
        )

    prompt = f"""
You are IntelliMind AI.

You are an intelligent educational assistant.

User question:
{question}

Relevant knowledge:
{knowledge}

Important rules:

- Understand short forms such as AI, ML, DL,
  NLP, CV, LLM, RAG, CNN, RNN and BERT.

- If the user asks "ML ki?",
  understand it as:
  "What is Machine Learning?"

- If the user asks "DL ki?",
  understand it as:
  "What is Deep Learning?"

- If the user asks "CV ki?",
  understand it as:
  "What is Computer Vision?"

- If the user asks "NLP ki?",
  understand it as:
  "What is Natural Language Processing?"

- Use the provided knowledge when it is relevant.

- If the provided knowledge is insufficient,
  answer using your general knowledge.

- Give clear and accurate answers.

- Use simple English.

- If the user asks in Bangla,
  answer in simple Bangla.

- Do not mention the dataset,
  similarity score,
  embeddings,
  TF-IDF,
  retrieval or internal system.

- For comparison questions,
  use a simple table when useful.

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


# =========================================================
# ANSWER FUNCTION
# =========================================================

def get_answer(question):

    matches = search_dataset(
        question,
        top_k=5
    )

    if not matches:

        return (
            "I could not find a relevant answer.",
            [],
            0.0
        )

    best = matches[0]

    score = best["score"]

    # Very strong dataset match
    if score >= 0.68:

        return (
            best["answer"],
            matches,
            score
        )

    # Otherwise use Gemini
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

    st.title("IntelliMind AI")

    st.caption(
        "Smart Knowledge Assistant"
    )

    st.divider()

    st.subheader(
        "Knowledge Base"
    )

    uploaded_file = st.file_uploader(
        "Upload your TXT dataset",
        type=["txt"]
    )

    if uploaded_file:

        try:

            questions, answers = parse_dataset(
                uploaded_file
            )

            if len(questions) == 0:

                st.error(
                    "No valid data found."
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
                        "Loading knowledge base..."
                    ):

                        st.session_state.embeddings = (
                            create_embeddings(
                                tuple(questions)
                            )
                        )

                        (
                            st.session_state.tfidf_vectorizer,
                            st.session_state.tfidf_matrix
                        ) = create_tfidf(
                            tuple(questions)
                        )

                    st.session_state.questions = (
                        questions
                    )

                    st.session_state.answers = (
                        answers
                    )

                    st.session_state.dataset_key = (
                        dataset_key
                    )

                    st.session_state.messages = []

                st.success(
                    f"{len(questions)} Q&A entries loaded"
                )

        except Exception as e:

            st.error(
                f"Dataset error: {e}"
            )

    else:

        st.info(
            "Upload your Question | Answer TXT file."
        )

    st.divider()

    st.subheader(
        "Settings"
    )

    show_matches = st.checkbox(
        "Show knowledge matches",
        value=False
    )

    if st.button(
        "New Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.caption(
        "NLP + Semantic Search + Gemini"
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.title(
    "IntelliMind AI"
)

st.caption(
    "Intelligent answers from your knowledge base "
    "with Generative AI support."
)

# Status
if st.session_state.questions:

    st.success(
        "AI system ready"
    )

else:

    st.warning(
        "Upload your TXT knowledge base to start."
    )


# =========================================================
# WELCOME
# =========================================================

if not st.session_state.messages:

    st.divider()

    st.subheader(
        "What can I help you with?"
    )

    st.write(
        "Ask questions naturally. "
        "The system understands common AI and "
        "technology abbreviations."
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            "**Machine Learning**\n\n"
            "What is ML?\n\n"
            "Explain machine learning."
        )

    with col2:

        st.info(
            "**Deep Learning**\n\n"
            "What is DL?\n\n"
            "DL vs ML."
        )

    with col3:

        st.info(
            "**Natural Language Processing**\n\n"
            "What is NLP?\n\n"
            "What are NLP applications?"
        )

    st.write("")

    if st.session_state.questions:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Knowledge Entries",
                len(
                    st.session_state.questions
                )
            )

        with col2:
            st.metric(
                "Search",
                "Semantic + TF-IDF"
            )

        with col3:
            st.metric(
                "AI",
                "Gemini"
            )


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    role = message["role"]

    with st.chat_message(role):

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

                    st.write(
                        f"{i}. {item['question']}"
                    )

                    st.caption(
                        f"Similarity: "
                        f"{item['score']:.3f}"
                    )

                    st.write(
                        item["answer"]
                    )

                    st.divider()


# =========================================================
# CHAT INPUT
# =========================================================

user_question = st.chat_input(
    "Ask IntelliMind AI..."
)


if user_question:

    if not st.session_state.questions:

        st.warning(
            "Please upload your TXT dataset first."
        )

        st.stop()

    # User
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

    # AI
    with st.chat_message("assistant"):

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

                    st.write(
                        f"{i}. {item['question']}"
                    )

                    st.caption(
                        f"Similarity: "
                        f"{item['score']:.3f}"
                    )

    # Save
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "matches": matches,
            "score": score
        }
    )
