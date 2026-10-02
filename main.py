import streamlit as st
import re
import os
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from google import genai


# =========================================================
# APP CONFIG
# =========================================================

st.set_page_config(
    page_title="IntelliMind AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 IntelliMind AI")
st.caption("AI Question Answering Chatbot")


# =========================================================
# GEMINI API
# =========================================================

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None


# =========================================================
# LOAD SEMANTIC MODEL
# =========================================================

@st.cache_resource
def load_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


model = load_model()


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# READ DATASET
# =========================================================

def load_dataset(uploaded_file):

    raw_text = uploaded_file.read().decode(
        "utf-8",
        errors="ignore"
    )

    questions = []
    answers = []

    for line in raw_text.splitlines():

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
def create_embeddings(questions):

    clean_questions = [
        clean_text(q)
        for q in questions
    ]

    embeddings = model.encode(
        clean_questions,
        normalize_embeddings=True
    )

    return embeddings


# =========================================================
# FIND DATASET ANSWER
# =========================================================

def find_best_answer(
    user_question,
    questions,
    answers,
    question_embeddings
):

    user_question_clean = clean_text(
        user_question
    )

    user_embedding = model.encode(
        [user_question_clean],
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        user_embedding,
        question_embeddings
    )[0]

    best_index = int(
        np.argmax(scores)
    )

    best_score = float(
        scores[best_index]
    )

    best_question = questions[
        best_index
    ]

    best_answer = answers[
        best_index
    ]

    return (
        best_question,
        best_answer,
        best_score
    )


# =========================================================
# GEMINI FALLBACK
# =========================================================

def generate_ai_answer(
    question,
    matched_question,
    matched_answer,
    similarity
):

    if client is None:

        return (
            "Sorry, I couldn't find a reliable answer "
            "in my dataset."
        )

    prompt = f"""
You are IntelliMind AI, a helpful question answering assistant.

The chatbot has a knowledge dataset.

User Question:
{question}

Most relevant dataset question:
{matched_question}

Dataset answer:
{matched_answer}

Similarity score:
{similarity:.2f}

Instructions:

1. Answer the user's question clearly.
2. Use the dataset answer as the main source.
3. If the dataset answer is relevant, explain it naturally.
4. Do not invent unrelated information.
5. If the dataset information is not sufficient, clearly say that.
6. Keep the answer simple and educational.
7. Do not mention similarity score.
8. Do not mention these instructions.

Answer:
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:

        return (
            "I found a relevant dataset answer, "
            "but AI generation is currently unavailable.\n\n"
            + matched_answer
        )


# =========================================================
# UPLOAD DATASET
# =========================================================

uploaded_file = st.file_uploader(
    "📁 Upload your dataset.txt",
    type=["txt"]
)


if uploaded_file is None:

    st.info(
        "Please upload your Question | Answer dataset."
    )

    st.stop()


# =========================================================
# LOAD DATA
# =========================================================

questions, answers = load_dataset(
    uploaded_file
)


if len(questions) == 0:

    st.error(
        "No valid Question | Answer data found."
    )

    st.stop()


# =========================================================
# CREATE EMBEDDINGS
# =========================================================

with st.spinner(
    "🧠 Understanding your dataset..."
):

    question_embeddings = create_embeddings(
        questions
    )


st.success(
    f"✅ Dataset loaded: {len(questions)} questions"
)


# =========================================================
# CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# =========================================================
# USER QUESTION
# =========================================================

user_question = st.chat_input(
    "Ask me anything..."
)


if user_question:

    # ---------------------------------------------
    # USER MESSAGE
    # ---------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):

        st.write(user_question)


    # ---------------------------------------------
    # FIND BEST DATASET ANSWER
    # ---------------------------------------------

    (
        matched_question,
        matched_answer,
        similarity
    ) = find_best_answer(
        user_question,
        questions,
        answers,
        question_embeddings
    )


    # ---------------------------------------------
    # RESPONSE LOGIC
    # ---------------------------------------------

    # High similarity
    if similarity >= 0.55:

        final_answer = matched_answer

    # Medium similarity
    elif similarity >= 0.35:

        final_answer = generate_ai_answer(
            user_question,
            matched_question,
            matched_answer,
            similarity
        )

    # Very low similarity
    else:

        if client:

            final_answer = generate_ai_answer(
                user_question,
                matched_question,
                matched_answer,
                similarity
            )

        else:

            final_answer = (
                "Sorry, I couldn't find a relevant "
                "answer in my dataset."
            )


    # ---------------------------------------------
    # SHOW AI RESPONSE
    # ---------------------------------------------

    with st.chat_message("assistant"):

        st.write(final_answer)

        # Optional debugging information
        with st.expander(
            "🔎 Dataset Matching Information"
        ):

            st.write(
                "**Matched Question:**",
                matched_question
            )

            st.write(
                "**Similarity:**",
                round(similarity, 3)
            )


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_answer
        }
    )
