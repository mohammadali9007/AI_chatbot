import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==============================
# LOAD DATASET FILE
# ==============================

uploaded_file = st.file_uploader(
    "Upload your Question-Answer Dataset",
    type=["txt"]
)

if uploaded_file is not None:

    text = uploaded_file.read().decode("utf-8")

    questions = []
    answers = []

    # ==============================
    # READ QUESTION | ANSWER
    # ==============================

    for line in text.splitlines():

        if "|" in line:

            question, answer = line.split("|", 1)

            questions.append(question.strip())
            answers.append(answer.strip())


    # ==============================
    # CLEAN TEXT
    # ==============================

    def clean_text(text):

        text = text.lower()

        text = re.sub(r"[^a-z0-9\s]", "", text)

        text = re.sub(r"\s+", " ", text)

        return text.strip()


    clean_questions = [
        clean_text(q) for q in questions
    ]


    # ==============================
    # TF-IDF
    # ==============================

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english"
    )

    question_vectors = vectorizer.fit_transform(
        clean_questions
    )


    # ==============================
    # ANSWER FUNCTION
    # ==============================

    def get_answer(user_question):

        user_question = clean_text(user_question)

        user_vector = vectorizer.transform(
            [user_question]
        )

        similarities = cosine_similarity(
            user_vector,
            question_vectors
        )[0]

        best_index = similarities.argmax()

        best_score = similarities[best_index]

        # Similarity threshold
        if best_score < 0.25:

            return "Sorry, I couldn't find a relevant answer in my dataset."

        return answers[best_index]


    # ==============================
    # CHATBOT UI
    # ==============================

    st.success(
        f"Dataset loaded successfully! "
        f"{len(questions)} questions found."
    )

    user_question = st.text_input(
        "Ask your question:"
    )

    if st.button("Ask"):

        if user_question.strip():

            answer = get_answer(user_question)

            st.write("### 🤖 Answer")
            st.write(answer)

        else:

            st.warning("Please enter a question.")
