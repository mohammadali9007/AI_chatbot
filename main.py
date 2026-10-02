import streamlit as st
import os
import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

from google import genai


# =========================================================
# APP CONFIG
# =========================================================

st.set_page_config(
    page_title="IntelliMind AI",
    page_icon="🤖",
    layout="wide"
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

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = "No dataset loaded"

if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if "question_embeddings" not in st.session_state:
    st.session_state.question_embeddings = None

if "tfidf_vectorizer" not in st.session_state:
    st.session_state.tfidf_vectorizer = None

if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# =========================================================
# GEMINI CLIENT
# =========================================================

def get_gemini_client():

    try:
        api_key = st.secrets.get("GEMINI_API_KEY")

        if not api_key:
            return None

        return genai.Client(api_key=api_key)

    except Exception:
        return None


# =========================================================
# PARSE DATASET
# =========================================================

def parse_dataset(text):

    questions = []
    answers = []

    lines = text.splitlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Ignore ID lines
        if line.startswith("id="):
            continue

        # Question | Answer
        if "|" in line:

            question, answer = line.split("|", 1)

            question = question.strip()
            answer = answer.strip()

            if question and answer:
                questions.append(question)
                answers.append(answer)

    return questions, answers


# =========================================================
# BUILD KNOWLEDGE BASE
# =========================================================

def build_knowledge_base(questions):

    if not questions:
        return

    # Sentence Transformer
    model = load_embedding_model()

    embeddings = model.encode(
        questions,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        stop_words=None
    )

    tfidf_matrix = vectorizer.fit_transform(questions)

    st.session_state.embedding_model = model
    st.session_state.question_embeddings = embeddings
    st.session_state.tfidf_vectorizer = vectorizer
    st.session_state.tfidf_matrix = tfidf_matrix


# =========================================================
# NORMALIZE QUESTION
# =========================================================

def normalize_question(text):

    text = text.lower().strip()

    # Common short forms
    replacements = {
        r"\bai\b": "artificial intelligence",
        r"\bml\b": "machine learning",
        r"\bdl\b": "deep learning",
        r"\bnlp\b": "natural language processing",
        r"\bcv\b": "computer vision",
        r"\bcnn\b": "convolutional neural network",
        r"\brnn\b": "recurrent neural network",
        r"\blstm\b": "long short term memory",
        r"\bllm\b": "large language model",
        r"\brag\b": "retrieval augmented generation",
        r"\bapi\b": "application programming interface",
        r"\bsvm\b": "support vector machine",
        r"\bknn\b": "k nearest neighbors",
        r"\bgan\b": "generative adversarial network",
        r"\bgpu\b": "graphics processing unit",
        r"\bcpu\b": "central processing unit"
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    return text


# =========================================================
# SEARCH KNOWLEDGE BASE
# =========================================================

def search_knowledge(question):

    if not st.session_state.dataset_loaded:
        return None

    if not st.session_state.questions:
        return None

    model = st.session_state.embedding_model

    if model is None:
        return None

    normalized_question = normalize_question(question)

    # -----------------------------------------
    # Semantic similarity
    # -----------------------------------------

    query_embedding = model.encode(
        [normalized_question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    semantic_scores = cosine_similarity(
        query_embedding,
        st.session_state.question_embeddings
    )[0]

    # -----------------------------------------
    # TF-IDF similarity
    # -----------------------------------------

    query_tfidf = st.session_state.tfidf_vectorizer.transform(
        [normalized_question]
    )

    tfidf_scores = cosine_similarity(
        query_tfidf,
        st.session_state.tfidf_matrix
    )[0]

    # -----------------------------------------
    # Hybrid score
    # -----------------------------------------

    hybrid_scores = (
        0.75 * semantic_scores
        +
        0.25 * tfidf_scores
    )

    best_index = int(np.argmax(hybrid_scores))

    return {
        "index": best_index,
        "question": st.session_state.questions[best_index],
        "answer": st.session_state.answers[best_index],
        "semantic_score": float(semantic_scores[best_index]),
        "tfidf_score": float(tfidf_scores[best_index]),
        "hybrid_score": float(hybrid_scores[best_index])
    }


# =========================================================
# GEMINI ANSWER
# =========================================================

def generate_ai_answer(question, context=""):

    client = get_gemini_client()

    if client is None:
        return (
            "Gemini API is not configured. "
            "Please add GEMINI_API_KEY to Streamlit Secrets."
        )

    if context:

        prompt = f"""
You are IntelliMind AI, a helpful AI assistant.

Answer the user's question clearly and accurately.

Use the provided knowledge base context when it is relevant.

Knowledge Base Context:
{context}

User Question:
{question}

Instructions:
- Give a direct answer.
- Use simple English.
- Explain technical terms when necessary.
- Do not mention internal retrieval or similarity scores.
- Do not say that you are using a dataset.
- If the context does not contain enough information, use your general knowledge.
"""

    else:

        prompt = f"""
You are IntelliMind AI, a helpful AI assistant.

Answer the following question clearly and accurately.

User Question:
{question}

Instructions:
- Give a direct and useful answer.
- Use simple English.
- Explain technical concepts in beginner-friendly language.
- Use examples when helpful.
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
                "Gemini is temporarily busy. "
                "Please try again after a few seconds."
            )

        return (
            "AI generation failed.\n\n"
            f"Error: {error_message}"
        )


# =========================================================
# PROCESS QUESTION
# =========================================================

def process_question(question):

    result = search_knowledge(question)

    # No knowledge base match
    if result is None:

        answer = generate_ai_answer(question)

        return answer, None

    score = result["hybrid_score"]

    # -----------------------------------------
    # Strong match
    # -----------------------------------------

    if score >= 0.72:

        answer = result["answer"]

        return answer, result

    # -----------------------------------------
    # Medium match
    # -----------------------------------------

    elif score >= 0.45:

        context = f"""
Question:
{result["question"]}

Answer:
{result["answer"]}
"""

        answer = generate_ai_answer(
            question,
            context
        )

        return answer, result

    # -----------------------------------------
    # Weak match
    # -----------------------------------------

    else:

        answer = generate_ai_answer(question)

        return answer, result


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🤖 IntelliMind AI")

    st.caption(
        "Intelligent Knowledge Assistant"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "AI Assistant",
            "Knowledge Base",
            "About"
        ]
    )

    st.divider()

    st.subheader("Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload TXT file",
        type=["txt"],
        help="Upload your Question | Answer knowledge base."
    )

    if uploaded_file is not None:

        try:

            text = uploaded_file.read().decode(
                "utf-8",
                errors="ignore"
            )

            questions, answers = parse_dataset(text)

            if questions:

                # Only rebuild if new dataset
                if (
                    st.session_state.dataset_name
                    != uploaded_file.name
                    or not st.session_state.dataset_loaded
                ):

                    with st.spinner(
                        "Building knowledge base..."
                    ):

                        build_knowledge_base(
                            questions
                        )

                    st.session_state.questions = questions
                    st.session_state.answers = answers
                    st.session_state.dataset_name = uploaded_file.name
                    st.session_state.dataset_loaded = True

                    st.success(
                        f"{len(questions)} entries loaded."
                    )

            else:

                st.error(
                    "No valid Question | Answer data found."
                )

        except Exception as e:

            st.error(
                f"Could not read file: {e}"
            )

    # Dataset information
    if st.session_state.dataset_loaded:

        st.metric(
            "Knowledge Entries",
            len(st.session_state.questions)
        )

    else:

        st.info(
            "Upload a TXT knowledge base to start."
        )

    st.divider()

    st.subheader("Conversation")

    st.metric(
        "Messages",
        len(st.session_state.messages)
    )

    if st.button(
        "Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    if st.session_state.dataset_loaded:

        st.success("System Ready")

    else:

        st.warning("Waiting for Dataset")


# =========================================================
# AI ASSISTANT PAGE
# =========================================================

if page == "AI Assistant":

    st.title("IntelliMind AI")

    st.caption(
        "Ask questions and get answers from your knowledge base "
        "with AI-powered fallback."
    )

    st.divider()

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Knowledge",
            (
                len(st.session_state.questions)
                if st.session_state.dataset_loaded
                else 0
            )
        )

    with col2:

        st.metric(
            "Messages",
            len(st.session_state.messages)
        )

    with col3:

        st.metric(
            "Search",
            "Hybrid"
        )

    with col4:

        st.metric(
            "AI",
            "Gemini"
        )

    st.divider()

    # Suggested questions
    st.subheader("Try asking")

    suggestion_cols = st.columns(3)

    suggestions = [
        "What is Artificial Intelligence?",
        "What is Machine Learning?",
        "What is NLP?"
    ]

    for i, suggestion in enumerate(suggestions):

        with suggestion_cols[i]:

            if st.button(
                suggestion,
                use_container_width=True,
                key=f"suggestion_{i}"
            ):

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": suggestion
                    }
                )

                answer, result = process_question(
                    suggestion
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "result": result
                    }
                )

                st.rerun()

    st.divider()

    # Chat history
    for message in st.session_state.messages:

        if message["role"] == "user":

            with st.chat_message("user"):

                st.write(
                    message["content"]
                )

        else:

            with st.chat_message("assistant"):

                st.write(
                    message["content"]
                )

                result = message.get(
                    "result"
                )

                if result:

                    with st.expander(
                        "View Knowledge Match"
                    ):

                        st.write(
                            "**Matched Question:**"
                        )

                        st.write(
                            result["question"]
                        )

                        st.write(
                            "**Semantic Score:**",
                            f"{result['semantic_score']:.3f}"
                        )

                        st.write(
                            "**TF-IDF Score:**",
                            f"{result['tfidf_score']:.3f}"
                        )

                        st.write(
                            "**Hybrid Score:**",
                            f"{result['hybrid_score']:.3f}"
                        )

    # Chat input
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

        with st.spinner(
            "Thinking..."
        ):

            answer, result = process_question(
                question
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "result": result
            }
        )

        st.rerun()


# =========================================================
# KNOWLEDGE BASE PAGE
# =========================================================

elif page == "Knowledge Base":

    st.title("Knowledge Base")

    st.caption(
        "Explore the questions and answers loaded from your TXT file."
    )

    st.divider()

    if not st.session_state.dataset_loaded:

        st.info(
            "No knowledge base loaded yet. "
            "Upload a TXT file from the sidebar."
        )

    else:

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Dataset",
                st.session_state.dataset_name
            )

        with col2:

            st.metric(
                "Total Entries",
                len(st.session_state.questions)
            )

        st.divider()

        st.subheader("Knowledge Entries")

        for i, (question, answer) in enumerate(
            zip(
                st.session_state.questions,
                st.session_state.answers
            )
        ):

            with st.expander(
                f"{i + 1}. {question}"
            ):

                st.write(answer)


# =========================================================
# ABOUT PAGE
# =========================================================

elif page == "About":

    st.title("About IntelliMind AI")

    st.caption(
        "A professional NLP-based knowledge assistant."
    )

    st.divider()

    st.subheader("How it works")

    st.write(
        """
        IntelliMind AI combines semantic search, TF-IDF
        and Generative AI to answer user questions.
        """
    )

    st.write("### 1. Knowledge Base")

    st.write(
        "The application reads Question | Answer pairs "
        "from an uploaded TXT file."
    )

    st.write("### 2. Semantic Search")

    st.write(
        "Sentence Transformers are used to understand "
        "the meaning of the user's question."
    )

    st.write("### 3. TF-IDF Search")

    st.write(
        "TF-IDF provides an additional lexical similarity "
        "signal between the question and knowledge base."
    )

    st.write("### 4. Hybrid Retrieval")

    st.write(
        "Semantic similarity and TF-IDF similarity are "
        "combined to find the most relevant knowledge."
    )

    st.write("### 5. Generative AI")

    st.write(
        "Gemini is used when the knowledge base does not "
        "provide a sufficiently strong answer."
    )

    st.divider()

    st.subheader("Technology")

    st.write(
        """
        • Python  
        • Streamlit  
        • Sentence Transformers  
        • Scikit-learn  
        • TF-IDF  
        • Cosine Similarity  
        • Google Gemini API
        """
    )

    st.divider()

    st.caption(
        "IntelliMind AI • NLP & Generative AI Project"
    )
