import streamlit as st
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from google import genai
from google.genai import types


# =========================================================
# APP CONFIG
# =========================================================

st.set_page_config(
    page_title="IntelliMind AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 IntelliMind AI")
st.write("Ask questions from your uploaded TXT knowledge file.")


# =========================================================
# SETTINGS
# =========================================================

MIN_SCORE = 0.08
TOP_RESULTS = 5


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "text": "",
    "sentences": [],
    "word_vectorizer": None,
    "word_matrix": None,
    "char_vectorizer": None,
    "char_matrix": None,
    "filename": "",
    "chat_history": []
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# GEMINI
# =========================================================

def get_gemini_client():

    api_key = None

    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    try:
        return genai.Client(api_key=api_key)

    except Exception:
        return None


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    if not text:
        return ""

    text = text.replace("\x00", " ")

    # Normalize line breaks
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


# =========================================================
# CREATE SENTENCES
# =========================================================

def create_sentences(text):

    text = clean_text(text)

    if not text:
        return []

    # Split on English punctuation and Bangla danda
    parts = re.split(
        r"(?<=[.!?।])\s+|\n+",
        text
    )

    sentences = []

    for part in parts:

        part = part.strip()

        if len(part) >= 5:
            sentences.append(part)

    return sentences


# =========================================================
# BUILD KNOWLEDGE BASE
# =========================================================

def build_knowledge_base(text):

    text = clean_text(text)

    sentences = create_sentences(text)

    if not sentences:
        return False

    # -----------------------------------------
    # WORD TF-IDF
    # -----------------------------------------

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=20000
    )

    word_matrix = word_vectorizer.fit_transform(
        sentences
    )

    # -----------------------------------------
    # CHARACTER TF-IDF
    # Helps with Bangla / spelling variations
    # -----------------------------------------

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        max_features=30000
    )

    char_matrix = char_vectorizer.fit_transform(
        sentences
    )

    st.session_state.text = text
    st.session_state.sentences = sentences

    st.session_state.word_vectorizer = word_vectorizer
    st.session_state.word_matrix = word_matrix

    st.session_state.char_vectorizer = char_vectorizer
    st.session_state.char_matrix = char_matrix

    return True


# =========================================================
# NORMALIZE QUESTION
# =========================================================

def normalize_text(text):

    text = text.lower()

    # Keep Unicode letters/numbers
    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# GET QUESTION KEYWORDS
# =========================================================

def get_keywords(question):

    question = normalize_text(question)

    words = question.split()

    # Common question words
    stop_words = {
        "what",
        "is",
        "are",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "to",
        "for",
        "and",
        "or",
        "who",
        "when",
        "where",
        "why",
        "how",
        "can",
        "could",
        "would",
        "does",
        "do",
        "did",
        "tell",
        "me",
        "about",

        # Bangla
        "কি",
        "কী",
        "কে",
        "কেন",
        "কখন",
        "কোথায়",
        "কোথায়",
        "কিভাবে",
        "কীভাবে",
        "সম্পর্কে",
        "বল",
        "বলুন",
        "হলো",
        "হয়",
        "হয়"
    }

    keywords = []

    for word in words:

        if word not in stop_words and len(word) > 1:

            keywords.append(word)

    return keywords


# =========================================================
# SEARCH TXT
# =========================================================

def search_txt(question):

    sentences = st.session_state.sentences

    if not sentences:
        return []

    # -----------------------------------------
    # WORD SIMILARITY
    # -----------------------------------------

    word_vectorizer = st.session_state.word_vectorizer
    word_matrix = st.session_state.word_matrix

    q_word = word_vectorizer.transform(
        [question]
    )

    word_scores = cosine_similarity(
        q_word,
        word_matrix
    )[0]

    # -----------------------------------------
    # CHARACTER SIMILARITY
    # -----------------------------------------

    char_vectorizer = st.session_state.char_vectorizer
    char_matrix = st.session_state.char_matrix

    q_char = char_vectorizer.transform(
        [question]
    )

    char_scores = cosine_similarity(
        q_char,
        char_matrix
    )[0]

    # -----------------------------------------
    # KEYWORD MATCH
    # -----------------------------------------

    keywords = get_keywords(question)

    final_results = []

    for i, sentence in enumerate(sentences):

        normalized_sentence = normalize_text(
            sentence
        )

        keyword_matches = 0

        for keyword in keywords:

            if keyword in normalized_sentence:
                keyword_matches += 1

        if keywords:

            keyword_score = (
                keyword_matches / len(keywords)
            )

        else:

            keyword_score = 0

        # -----------------------------------------
        # COMBINE SCORES
        # -----------------------------------------

        final_score = (
            (word_scores[i] * 0.50)
            +
            (char_scores[i] * 0.25)
            +
            (keyword_score * 0.25)
        )

        final_results.append({
            "text": sentence,
            "score": float(final_score),
            "word_score": float(word_scores[i]),
            "char_score": float(char_scores[i]),
            "keyword_score": float(keyword_score)
        })

    # Sort highest score first
    final_results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Only return relevant results
    results = []

    for result in final_results[:TOP_RESULTS]:

        if result["score"] >= MIN_SCORE:

            results.append(result)

    return results


# =========================================================
# GET CONTEXT
# =========================================================

def get_context(results):

    if not results:
        return ""

    context = []

    for result in results:

        context.append(
            result["text"]
        )

    return "\n".join(context)


# =========================================================
# GEMINI NATURAL ANSWER
# =========================================================

def generate_gemini_answer(question, context):

    client = get_gemini_client()

    if client is None:
        return None

    prompt = f"""
You are IntelliMind AI.

The user uploaded a TXT knowledge file.

Your job is to answer the user's question using ONLY the
information provided from the TXT file.

TXT CONTENT:
-------------------------
{context}
-------------------------

USER QUESTION:
{question}

RULES:

1. Answer the actual question directly.
2. Do not copy the entire TXT.
3. Use your own natural wording.
4. Do not invent information.
5. If the TXT contains the answer, use it.
6. If the TXT does not contain enough information, say:
   "This information is not available in the uploaded TXT file."
7. For simple questions, give a short answer.
8. For difficult questions, explain step by step.
9. If the question is in Bangla or Banglish, answer in Bangla/Banglish.
10. Do not mention TF-IDF, similarity, chunks, retrieval,
    context or internal processing.

Give ONLY the final answer.
"""

    try:

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=500
            )
        )

        if response and response.text:

            return response.text.strip()

    except Exception:

        return None

    return None


# =========================================================
# LOCAL ANSWER
# =========================================================

def local_answer(question, results):

    if not results:
        return None

    # -----------------------------------------
    # Best result
    # -----------------------------------------

    best = results[0]

    # If strong match, use best sentence
    if best["score"] >= 0.30:

        return best["text"]

    # -----------------------------------------
    # Multiple related sentences
    # -----------------------------------------

    good_results = []

    for result in results:

        if result["score"] >= 0.15:

            good_results.append(
                result["text"]
            )

    if good_results:

        # Remove duplicates
        unique = []

        for sentence in good_results:

            if sentence not in unique:

                unique.append(sentence)

        return " ".join(unique[:3])

    # Weak result
    return None


# =========================================================
# FINAL QUESTION ANSWER
# =========================================================

def answer_question(question):

    results = search_txt(question)

    # No relevant information
    if not results:

        return (
            "❌ এই প্রশ্নের উত্তর uploaded TXT file-এর মধ্যে "
            "পাওয়া যায়নি।"
        ), "TXT File"

    # -----------------------------------------
    # Build small context
    # -----------------------------------------

    context = get_context(results)

    context = context[:7000]

    # -----------------------------------------
    # Try Gemini
    # -----------------------------------------

    ai_answer = generate_gemini_answer(
        question,
        context
    )

    if ai_answer:

        return ai_answer, "📚 TXT + Gemini"

    # -----------------------------------------
    # Gemini unavailable
    # Use local TXT answer
    # -----------------------------------------

    local = local_answer(
        question,
        results
    )

    if local:

        return local, "📄 TXT File"

    return (
        "❌ এই প্রশ্নের জন্য TXT file-এ "
        "যথেষ্ট তথ্য পাওয়া যায়নি।"
    ), "TXT File"


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📁 Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload TXT File",
        type=["txt"]
    )

    if uploaded_file:

        if st.button(
            "🔨 Build Knowledge Base",
            use_container_width=True
        ):

            try:

                file_bytes = uploaded_file.read()

                try:

                    file_text = file_bytes.decode(
                        "utf-8"
                    )

                except UnicodeDecodeError:

                    file_text = file_bytes.decode(
                        "latin-1"
                    )

                if build_knowledge_base(
                    file_text
                ):

                    st.session_state.filename = (
                        uploaded_file.name
                    )

                    st.session_state.chat_history = []

                    st.success(
                        "Knowledge Base Ready! ✅"
                    )

                    st.info(
                        f"📄 {uploaded_file.name}"
                    )

                    st.info(
                        f"📝 {len(st.session_state.sentences)} "
                        f"sentences indexed"
                    )

                else:

                    st.error(
                        "TXT file is empty."
                    )

            except Exception as e:

                st.error(
                    f"File processing error: {e}"
                )


    # -----------------------------------------
    # STATUS
    # -----------------------------------------

    st.divider()

    st.subheader("⚙️ System Status")

    if st.session_state.sentences:

        st.success("🟢 Knowledge Base Active")

        st.write(
            f"📄 File: "
            f"{st.session_state.filename}"
        )

        st.write(
            f"📝 Sentences: "
            f"{len(st.session_state.sentences)}"
        )

    else:

        st.warning(
            "🟡 No TXT file loaded"
        )


    # -----------------------------------------
    # CLEAR CHAT
    # -----------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()


# =========================================================
# MAIN STATUS
# =========================================================

if st.session_state.sentences:

    st.success(
        f"📚 Ready to answer from: "
        f"{st.session_state.filename}"
    )

else:

    st.info(
        "👈 Upload your TXT file from the sidebar "
        "and click Build Knowledge Base."
    )


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )

        if message["role"] == "assistant":

            st.caption(
                message.get(
                    "source",
                    ""
                )
            )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask a question from your TXT file..."
)


if question:

    # -----------------------------------------
    # User message
    # -----------------------------------------

    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.write(question)


    # -----------------------------------------
    # Assistant
    # -----------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching your TXT file..."
        ):

            answer, source = answer_question(
                question
            )

        st.write(answer)

        st.caption(
            f"📌 Source: {source}"
        )


    # Save answer
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "source": f"📌 Source: {source}"
    })
