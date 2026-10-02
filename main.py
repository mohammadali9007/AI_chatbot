import os
import re
import streamlit as st

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from google import genai


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "gemini-3.8-flash"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

MIN_SIMILARITY = 0.20
TOP_K = 5


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="IntelliMind AI",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
}

.sub-title {
    text-align: center;
    color: #888;
    margin-bottom: 30px;
}

.source-box {
    padding: 10px;
    border-radius: 10px;
    background: rgba(100,100,100,0.08);
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🤖 IntelliMind AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Smart Knowledge Assistant • RAG + Gemini'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chat" not in st.session_state:
    st.session_state.chat = []


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_gemini_client():

    api_key = None

    # Streamlit Cloud
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    # Local environment
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


client = get_gemini_client()


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    text = text.replace("\x00", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# SMART CHUNKING
# ============================================================

def create_chunks(text):

    text = clean_text(text)

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + CHUNK_SIZE,
            len(text)
        )

        chunk = text[start:end]

        # Try to finish at sentence boundary
        if end < len(text):

            last_period = max(
                chunk.rfind("."),
                chunk.rfind("?"),
                chunk.rfind("!")
            )

            if last_period > CHUNK_SIZE * 0.60:

                chunk = chunk[:last_period + 1]

                end = start + len(chunk)

        if len(chunk.strip()) > 50:

            chunks.append(chunk.strip())

        next_start = end - CHUNK_OVERLAP

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


# ============================================================
# READ TXT
# ============================================================

def read_txt(file):

    try:

        return clean_text(
            file.read().decode(
                "utf-8",
                errors="ignore"
            )
        )

    except Exception as e:

        st.error(f"TXT error: {e}")

        return ""


# ============================================================
# READ PDF
# ============================================================

def read_pdf(file):

    try:

        reader = PdfReader(file)

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages.append(text)

        return clean_text(
            "\n".join(pages)
        )

    except Exception as e:

        st.error(f"PDF error: {e}")

        return ""


# ============================================================
# PROCESS FILE
# ============================================================

def process_file(file):

    filename = file.name.lower()

    if filename.endswith(".txt"):

        text = read_txt(file)

    elif filename.endswith(".pdf"):

        text = read_pdf(file)

    else:

        return []

    if not text:
        return []

    chunks = create_chunks(text)

    result = []

    for i, chunk in enumerate(chunks):

        result.append(
            {
                "text": chunk,
                "source": file.name,
                "chunk_id": i + 1
            }
        )

    return result


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_context(question):

    chunks = st.session_state.chunks

    if not chunks:

        return [], 0.0

    texts = [
        item["text"]
        for item in chunks
    ]

    try:

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        matrix = vectorizer.fit_transform(
            texts + [question]
        )

        question_vector = matrix[-1]

        document_vectors = matrix[:-1]

        scores = cosine_similarity(
            question_vector,
            document_vectors
        )[0]

        ranked = scores.argsort()[::-1]

        results = []

        for index in ranked[:TOP_K]:

            score = float(scores[index])

            results.append(
                {
                    "text": chunks[index]["text"],
                    "source": chunks[index]["source"],
                    "chunk_id": chunks[index]["chunk_id"],
                    "score": score
                }
            )

        best_score = (
            results[0]["score"]
            if results
            else 0.0
        )

        return results, best_score

    except Exception:

        return [], 0.0


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):

    useful = [
        item
        for item in results
        if item["score"] >= MIN_SIMILARITY
    ]

    if not useful:
        return ""

    context_parts = []

    for item in useful:

        context_parts.append(
            f"""
SOURCE: {item['source']}
CHUNK: {item['chunk_id']}
RELEVANCE: {item['score']:.2f}

{item['text']}
"""
        )

    return "\n-----------------------\n".join(
        context_parts
    )


# ============================================================
# GEMINI
# ============================================================

def ask_gemini(question, context):

    if client is None:

        return (
            None,
            "Gemini API key is not configured."
        )

    if context:

        prompt = f"""
You are IntelliMind AI, a professional knowledge assistant.

The user uploaded documents to this application.

Your job is to answer the user's question using the
retrieved document context below.

IMPORTANT RULES:

1. Use the retrieved context as the primary source.
2. Do not invent information from the documents.
3. If the context clearly answers the question,
   explain the answer naturally.
4. If the context only partially answers the question,
   explain what is supported and what is missing.
5. Do not claim that something is from the document
   if it is not actually present.
6. Keep the answer clear and useful.
7. Use bullet points when appropriate.

RETRIEVED DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Now provide the best answer.
"""

        source = "📚 Knowledge Base + Gemini"

    else:

        prompt = f"""
You are IntelliMind AI, a general AI assistant.

The user asked:

{question}

No sufficiently relevant information was found
in the uploaded documents.

Answer using your general knowledge.

IMPORTANT:

1. Do not pretend the answer came from the uploaded files.
2. Clearly answer the user's question.
3. Keep the answer easy to understand.
4. If the information may be uncertain or time-sensitive,
   mention that limitation.

Answer:
"""

        source = "🌐 Gemini General Knowledge"

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        if response and response.text:

            return (
                response.text.strip(),
                source
            )

        return (
            None,
            "Gemini returned an empty response."
        )

    except Exception as e:

        error = str(e)

        if "404" in error:

            return (
                None,
                "Model not available. Check Gemini model name."
            )

        if "429" in error:

            return (
                None,
                "API quota/rate limit reached. Try again later."
            )

        if "401" in error or "403" in error:

            return (
                None,
                "API key authentication failed."
            )

        if "503" in error:

            return (
                None,
                "Gemini is temporarily overloaded. Try again."
            )

        return (
            None,
            f"Gemini error: {error}"
        )


# ============================================================
# FINAL ANSWER ENGINE
# ============================================================

def generate_answer(question):

    results, best_score = retrieve_context(
        question
    )

    context = build_context(results)

    answer, source = ask_gemini(
        question,
        context
    )

    # Gemini successful
    if answer:

        return (
            answer,
            source,
            best_score,
            results
        )

    # Gemini unavailable but KB exists
    if context:

        fallback = (
            "Gemini could not generate the final response "
            "right now.\n\n"
            "However, relevant information was found "
            "in your uploaded knowledge base:\n\n"
            + context
        )

        return (
            fallback,
            "📚 Knowledge Base Fallback",
            best_score,
            results
        )

    return (
        "I could not generate an answer right now. "
        "Please check your Gemini API key and try again.",
        "⚠️ System Fallback",
        best_score,
        results
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📂 Knowledge Base")

    files = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button(
        "🚀 Build Knowledge Base",
        use_container_width=True
    ):

        if not files:

            st.warning(
                "Please upload at least one file."
            )

        else:

            all_chunks = []
            file_names = []

            progress = st.progress(0)

            for i, file in enumerate(files):

                chunks = process_file(file)

                all_chunks.extend(chunks)

                file_names.append(file.name)

                progress.progress(
                    (i + 1) / len(files)
                )

            st.session_state.chunks = all_chunks

            st.session_state.documents = file_names

            st.success(
                f"{len(file_names)} file(s) processed!"
            )

    st.divider()

    st.subheader("📊 Status")

    st.write(
        f"📄 Files: **{len(st.session_state.documents)}**"
    )

    st.write(
        f"🧩 Chunks: **{len(st.session_state.chunks)}**"
    )

    if client:

        st.success(
            "🟢 Gemini Connected"
        )

    else:

        st.error(
            "🔴 Gemini Not Connected"
        )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.chat = []

        st.rerun()


# ============================================================
# DOCUMENTS
# ============================================================

if st.session_state.documents:

    st.markdown("### 📚 Your Documents")

    for name in st.session_state.documents:

        st.caption(
            f"📄 {name}"
        )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.chat:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if message["role"] == "assistant":

            if message.get("source"):

                st.caption(
                    f"Source: {message['source']}"
                )

            if message.get("score") is not None:

                st.caption(
                    f"Best similarity: "
                    f"{message['score']:.2f}"
                )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask anything about your documents..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # USER
    st.session_state.chat.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # AI
    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching knowledge base..."
        ):

            answer, source, score, results = generate_answer(
                question
            )

        st.markdown(answer)

        st.caption(
            f"Source: {source}"
        )

        st.caption(
            f"Best similarity: {score:.2f}"
        )

        # Show retrieved sources
        relevant = [
            r for r in results
            if r["score"] >= MIN_SIMILARITY
        ]

        if relevant:

            with st.expander(
                "🔍 View retrieved sources"
            ):

                for item in relevant:

                    st.write(
                        f"📄 {item['source']} "
                        f"| Chunk {item['chunk_id']} "
                        f"| Score {item['score']:.2f}"
                    )

    st.session_state.chat.append(
        {
            "role": "assistant",
            "content": answer,
            "source": source,
            "score": score
        }
    )
