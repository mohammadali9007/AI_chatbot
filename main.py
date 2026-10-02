import os
import re
import streamlit as st

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai


# =========================================================
# CONFIG
# =========================================================

APP_NAME = "IntelliMind AI"

MODEL_NAME = "gemini-3.8-flash"

# Only one fallback to reduce waiting time
FALLBACK_MODEL = None

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# Faster retrieval
TOP_K = 3
MIN_SIMILARITY = 0.20

# Limit context sent to Gemini
MAX_CONTEXT_CHARS = 5000


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0;
}

.subtitle {
    color: #777;
    font-size: 17px;
    margin-bottom: 25px;
}

.card {
    padding: 18px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 15px;
}

.source-box {
    padding: 10px;
    border-radius: 10px;
    background: rgba(128,128,128,0.08);
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "chunks": [],
    "documents": [],
    "chat_history": [],
    "document_summary": "",
    "suggested_questions": [],

    # IMPORTANT:
    # TF-IDF objects are stored after building KB
    "vectorizer": None,
    "tfidf_matrix": None,

    "kb_ready": False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# GEMINI CLIENT
# =========================================================

@st.cache_resource
def get_gemini_client():

    api_key = None

    try:
        api_key = st.secrets["GEMINI_API_KEY"]
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

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Remove excessive punctuation spacing
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    return text.strip()


# =========================================================
# CREATE CHUNKS
# =========================================================

def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    text = clean_text(text)

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


# =========================================================
# READ TXT
# =========================================================

def read_txt(uploaded_file):

    try:

        raw = uploaded_file.read()

        try:
            text = raw.decode("utf-8")

        except UnicodeDecodeError:
            text = raw.decode("latin-1")

        return clean_text(text)

    except Exception as e:

        st.error(f"TXT reading error: {e}")

        return ""


# =========================================================
# READ PDF
# =========================================================

def read_pdf(uploaded_file):

    text_parts = []

    try:

        reader = PdfReader(uploaded_file)

        for page in reader.pages:

            try:
                page_text = page.extract_text()

                if page_text:
                    text_parts.append(page_text)

            except Exception:
                continue

        return clean_text("\n".join(text_parts))

    except Exception as e:

        st.error(f"PDF reading error: {e}")

        return ""


# =========================================================
# PROCESS FILE
# =========================================================

def process_file(uploaded_file):

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".txt"):

        text = read_txt(uploaded_file)

    elif file_name.endswith(".pdf"):

        text = read_pdf(uploaded_file)

    else:

        return []

    if not text:
        return []

    return create_chunks(text)


# =========================================================
# BUILD TF-IDF INDEX
# =========================================================

def build_tfidf_index():

    chunks = st.session_state.chunks

    if not chunks:
        return False

    try:

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=10000
        )

        matrix = vectorizer.fit_transform(chunks)

        # SAVE THEM
        st.session_state.vectorizer = vectorizer
        st.session_state.tfidf_matrix = matrix

        return True

    except Exception as e:

        st.error(f"TF-IDF error: {e}")

        return False


# =========================================================
# RETRIEVE KNOWLEDGE
# =========================================================

def retrieve_knowledge(question):

    chunks = st.session_state.chunks
    vectorizer = st.session_state.vectorizer
    matrix = st.session_state.tfidf_matrix

    if not chunks:
        return []

    if vectorizer is None or matrix is None:
        return []

    try:

        # IMPORTANT:
        # We DO NOT fit TF-IDF again.
        question_vector = vectorizer.transform([question])

        scores = cosine_similarity(
            question_vector,
            matrix
        ).flatten()

        # Get top results
        top_indices = scores.argsort()[::-1][:TOP_K]

        results = []

        for index in top_indices:

            score = float(scores[index])

            results.append({
                "index": int(index),
                "score": score,
                "text": chunks[index]
            })

        return results

    except Exception as e:

        st.error(f"Retrieval error: {e}")

        return []


# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(results):

    if not results:
        return ""

    context_parts = []
    total_chars = 0

    for item in results:

        score = item["score"]

        if score < MIN_SIMILARITY:
            continue

        text = item["text"]

        remaining = MAX_CONTEXT_CHARS - total_chars

        if remaining <= 0:
            break

        text = text[:remaining]

        context_parts.append(
            f"[Knowledge Chunk | Similarity: {score:.2f}]\n{text}"
        )

        total_chars += len(text)

    return "\n\n".join(context_parts)


# =========================================================
# GEMINI CALL
# =========================================================

def call_gemini(prompt):

    client = get_gemini_client()

    if client is None:
        return None, "Gemini API key not configured."

    models_to_try = [MODEL_NAME]

    if FALLBACK_MODEL:
        models_to_try.append(FALLBACK_MODEL)

    last_error = ""

    for model in models_to_try:

        try:

            response = client.models.generate_content(
                model=model,
                contents=prompt
            )

            if response and response.text:

                return response.text.strip(), model

        except Exception as e:

            last_error = str(e)

            # If primary fails and fallback exists,
            # try only the fallback.
            continue

    return None, last_error


# =========================================================
# ASK GEMINI
# =========================================================

def ask_gemini(question, context="", mode="Normal"):

    if context:

        prompt = f"""
You are IntelliMind AI.

Mode: {mode}

Answer the user's question using the knowledge base below.

Rules:
- Give a direct and useful answer.
- Prefer the provided knowledge base.
- If the answer is not available in the knowledge base, clearly say that.
- Do not invent facts from the knowledge base.
- Keep the answer concise.
- Use simple language.

KNOWLEDGE BASE:
{context}

USER QUESTION:
{question}
"""

    else:

        prompt = f"""
You are IntelliMind AI.

Mode: {mode}

Answer this question clearly and accurately.

Keep the answer concise and easy to understand.

USER QUESTION:
{question}
"""

    return call_gemini(prompt)


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(question, mode):

    results = retrieve_knowledge(question)

    context = build_context(results)

    answer, model_or_error = ask_gemini(
        question,
        context,
        mode
    )

    if answer:

        if context:

            best_score = max(
                [x["score"] for x in results],
                default=0
            )

            return (
                answer,
                f"📚 Knowledge Base + Gemini ({model_or_error})",
                best_score
            )

        else:

            return (
                answer,
                f"🤖 Gemini ({model_or_error})",
                0
            )

    # Gemini failed
    if context:

        best_score = max(
            [x["score"] for x in results],
            default=0
        )

        return (
            "Gemini is temporarily unavailable. "
            "However, I found relevant information in your "
            "knowledge base:\n\n" + context,
            "📚 Knowledge Base only",
            best_score
        )

    return (
        f"⚠️ Gemini error:\n{model_or_error}",
        "❌ Gemini Error",
        0
    )


# =========================================================
# TEST GEMINI
# =========================================================

def test_gemini():

    answer, info = call_gemini(
        "Reply with exactly: Gemini connection successful."
    )

    if answer:

        st.success(f"✅ {answer}")
        st.caption(f"Model: {info}")

    else:

        st.error(f"❌ Gemini error: {info}")


# =========================================================
# DOCUMENT SUMMARY
# =========================================================

def generate_summary():

    chunks = st.session_state.chunks

    if not chunks:
        return "No knowledge base available."

    # Keep summary input small
    text = "\n\n".join(chunks[:8])

    text = text[:7000]

    prompt = f"""
Summarize the following document in simple language.

Give:
1. Main topic
2. Important points
3. Key findings

DOCUMENT:

{text}
"""

    answer, info = call_gemini(prompt)

    if answer:
        return answer

    return f"Summary unavailable.\n\nError: {info}"


# =========================================================
# SUGGESTED QUESTIONS
# =========================================================

def generate_questions():

    chunks = st.session_state.chunks

    if not chunks:
        return []

    text = "\n\n".join(chunks[:5])
    text = text[:5000]

    prompt = f"""
Create 5 useful questions from this document.

Return only the questions.
Number them 1 to 5.

DOCUMENT:

{text}
"""

    answer, info = call_gemini(prompt)

    if not answer:
        return []

    questions = []

    for line in answer.splitlines():

        line = line.strip()

        line = re.sub(
            r"^[0-9]+[\.\)\-\s]+",
            "",
            line
        )

        if line:
            questions.append(line)

    return questions[:5]


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🤖 IntelliMind AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Fast AI Assistant with PDF/TXT Knowledge Base'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Settings")

    mode = st.selectbox(
        "Response Mode",
        [
            "Normal",
            "Student",
            "Research"
        ]
    )

    st.divider()

    st.subheader("📄 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button(
        "🚀 Build Knowledge Base",
        use_container_width=True
    ):

        if not uploaded_files:

            st.warning("Please upload a PDF or TXT file.")

        else:

            all_chunks = []
            document_names = []

            progress = st.progress(0)

            for i, file in enumerate(uploaded_files):

                chunks = process_file(file)

                if chunks:

                    all_chunks.extend(chunks)
                    document_names.append(file.name)

                progress.progress(
                    (i + 1) / len(uploaded_files)
                )

            st.session_state.chunks = all_chunks
            st.session_state.documents = document_names

            # Build TF-IDF ONLY ONCE
            success = build_tfidf_index()

            if success:

                st.session_state.kb_ready = True

                st.session_state.document_summary = ""
                st.session_state.suggested_questions = []

                st.success(
                    f"✅ Knowledge Base ready!\n\n"
                    f"Documents: {len(document_names)}\n\n"
                    f"Chunks: {len(all_chunks)}"
                )

    st.divider()

    st.subheader("🔌 System Status")

    if get_gemini_client():

        st.success("Gemini API: Connected")

    else:

        st.error("Gemini API: Not Connected")

    if st.session_state.kb_ready:

        st.success(
            f"Knowledge Base: {len(st.session_state.chunks)} chunks"
        )

    else:

        st.info("Knowledge Base: Empty")

    st.divider()

    if st.button(
        "🧪 Test Gemini",
        use_container_width=True
    ):

        test_gemini()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()


# =========================================================
# MAIN FEATURES
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
        <div class="card">
        <h3>📚 Knowledge Base</h3>
        <p>Ask questions from uploaded PDF/TXT files.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        """
        <div class="card">
        <h3>⚡ Fast Retrieval</h3>
        <p>TF-IDF index is created only once.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        """
        <div class="card">
        <h3>🤖 Gemini AI</h3>
        <p>Gemini generates the final answer.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# DOCUMENT TOOLS
# =========================================================

if st.session_state.kb_ready:

    st.divider()

    st.subheader("🛠️ Document Tools")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "📝 Generate Summary",
            use_container_width=True
        ):

            with st.spinner("Generating summary..."):

                summary = generate_summary()

            st.session_state.document_summary = summary

    with col2:

        if st.button(
            "❓ Generate Questions",
            use_container_width=True
        ):

            with st.spinner("Generating questions..."):

                questions = generate_questions()

            st.session_state.suggested_questions = questions


# =========================================================
# SHOW SUMMARY
# =========================================================

if st.session_state.document_summary:

    st.divider()

    st.subheader("📝 Document Summary")

    st.write(
        st.session_state.document_summary
    )


# =========================================================
# SHOW QUESTIONS
# =========================================================

if st.session_state.suggested_questions:

    st.subheader("❓ Suggested Questions")

    for question in st.session_state.suggested_questions:

        st.markdown(f"- {question}")


# =========================================================
# CHAT HISTORY
# =========================================================

st.divider()

st.subheader("💬 Chat")


for item in st.session_state.chat_history:

    with st.chat_message("user"):
        st.write(item["question"])

    with st.chat_message("assistant"):

        st.write(item["answer"])

        st.caption(
            f"{item['source']} | "
            f"Similarity: {item['score']:.2f}"
        )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask IntelliMind AI..."
)


if question:

    question = question.strip()

    if question:

        with st.chat_message("user"):

            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                answer, source, score = generate_answer(
                    question,
                    mode
                )

            st.write(answer)

            st.caption(
                f"{source} | Similarity: {score:.2f}"
            )

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "source": source,
            "score": score
        })
