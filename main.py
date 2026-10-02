import streamlit as st
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from google import genai
from google.genai import types


# =========================
# APP CONFIG
# =========================

st.set_page_config(
    page_title="IntelliMind AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 IntelliMind AI")
st.write("Upload a TXT file and ask questions from it.")


# =========================
# SETTINGS
# =========================

MIN_SIMILARITY = 0.10
TOP_K = 3


# =========================
# SESSION STATE
# =========================

if "text" not in st.session_state:
    st.session_state.text = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

if "matrix" not in st.session_state:
    st.session_state.matrix = None

if "filename" not in st.session_state:
    st.session_state.filename = ""


# =========================
# GEMINI
# =========================

def get_gemini():

    api_key = None

    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except:
        pass

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    try:
        return genai.Client(api_key=api_key)
    except:
        return None


# =========================
# CLEAN TEXT
# =========================

def clean_text(text):

    text = text.replace("\x00", "")

    # Remove extra spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove too many empty lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


# =========================
# CREATE CHUNKS
# =========================

def create_chunks(text, chunk_size=700):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(words[i:i + chunk_size])

        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks


# =========================
# BUILD KNOWLEDGE BASE
# =========================

def build_knowledge_base(text):

    text = clean_text(text)

    chunks = create_chunks(text)

    if not chunks:
        return False

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(chunks)

    st.session_state.text = text
    st.session_state.chunks = chunks
    st.session_state.vectorizer = vectorizer
    st.session_state.matrix = matrix

    return True


# =========================
# SEARCH TXT
# =========================

def search_txt(question):

    if not st.session_state.chunks:
        return []

    vectorizer = st.session_state.vectorizer
    matrix = st.session_state.matrix

    question_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        question_vector,
        matrix
    )[0]

    results = []

    for index in scores.argsort()[::-1][:TOP_K]:

        score = float(scores[index])

        if score >= MIN_SIMILARITY:

            results.append({
                "text": st.session_state.chunks[index],
                "score": score
            })

    return results


# =========================
# GEMINI ANSWER
# =========================

def generate_gemini_answer(question, context):

    client = get_gemini()

    if client is None:
        return None

    prompt = f"""
You are a helpful AI assistant.

The user uploaded a TXT knowledge file.

Answer the user's question using the information from the TXT file.

Important rules:

1. Use the provided TXT information.
2. Do not invent information.
3. Do not copy unnecessary large parts of the TXT.
4. Understand the question first.
5. Give a direct and natural answer.
6. If the answer has steps, use numbered steps.
7. If the user asks a simple question, give a short answer.
8. If the user writes Bangla or Banglish, answer in Bangla/Banglish.
9. If the answer is not available in the provided text, say that clearly.

TXT INFORMATION:
----------------
{context}
----------------

USER QUESTION:
{question}

Now answer the question naturally.
"""

    try:

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=500
            )
        )

        if response.text:
            return response.text.strip()

    except Exception:
        return None

    return None


# =========================
# LOCAL TXT ANSWER
# =========================

def local_answer(question, results):

    if not results:
        return None

    question_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            question.lower()
        )
    )

    candidates = []

    for result in results:

        text = result["text"]

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

        for sentence in sentences:

            sentence_words = set(
                re.findall(
                    r"\b[a-zA-Z0-9]+\b",
                    sentence.lower()
                )
            )

            overlap = len(
                question_words.intersection(sentence_words)
            )

            if overlap > 0:

                candidates.append(
                    (overlap, sentence.strip())
                )

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    if candidates:

        answer_sentences = []

        for _, sentence in candidates[:5]:

            if sentence not in answer_sentences:
                answer_sentences.append(sentence)

        return " ".join(answer_sentences)

    # If no sentence matched exactly,
    # return relevant chunk
    return results[0]["text"]


# =========================
# FINAL ANSWER
# =========================

def answer_question(question):

    results = search_txt(question)

    if not results:

        return (
            "❌ এই প্রশ্নের উত্তর TXT file-এর মধ্যে পাওয়া যায়নি।"
        ), "TXT File"

    # Combine relevant TXT parts
    context_parts = []

    for result in results:

        context_parts.append(
            result["text"]
        )

    context = "\n\n".join(context_parts)

    # Limit context
    context = context[:6000]

    # Try Gemini
    ai_answer = generate_gemini_answer(
        question,
        context
    )

    if ai_answer:

        return ai_answer, "TXT + Gemini"

    # Gemini unavailable/quota exceeded
    local = local_answer(
        question,
        results
    )

    return local, "TXT File"


# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.header("📁 Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload TXT file",
        type=["txt"]
    )

    if uploaded_file:

        if st.button(
            "🔨 Build Knowledge Base",
            use_container_width=True
        ):

            try:

                file_text = uploaded_file.read().decode(
                    "utf-8"
                )

            except:

                file_text = uploaded_file.read().decode(
                    "latin-1"
                )

            if build_knowledge_base(file_text):

                st.session_state.filename = (
                    uploaded_file.name
                )

                st.success(
                    "Knowledge Base Ready! ✅"
                )

                st.info(
                    f"File: {uploaded_file.name}"
                )

            else:

                st.error(
                    "TXT file is empty."
                )


# =========================
# STATUS
# =========================

if st.session_state.chunks:

    st.success(
        f"📚 Knowledge Base Active: "
        f"{st.session_state.filename}"
    )

    st.write(
        f"Total sections: "
        f"{len(st.session_state.chunks)}"
    )

else:

    st.info(
        "👈 First upload a TXT file and "
        "click 'Build Knowledge Base'."
    )


# =========================
# CHAT
# =========================

st.divider()

question = st.chat_input(
    "Ask a question from your TXT file..."
)


if question:

    if not st.session_state.chunks:

        st.warning(
            "Please upload and build a TXT Knowledge Base first."
        )

    else:

        with st.chat_message("user"):

            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                answer, source = answer_question(
                    question
                )

            st.write(answer)

            st.caption(
                f"📌 Source: {source}"
            )
