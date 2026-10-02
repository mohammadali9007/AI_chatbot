import os
import re
import hashlib

import streamlit as st
from pypdf import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from google import genai
from google.genai import types


# =========================================================
# CONFIGURATION
# =========================================================

APP_NAME = "IntelliMind AI"

# Current Gemini model
MODEL_NAME = "gemini-3.8-flash"

# Chunk settings
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# Retrieval settings
TOP_K = 3
MIN_SIMILARITY = 0.18

# Maximum knowledge sent to Gemini
MAX_CONTEXT_CHARS = 6000

# Maximum chat history used in prompt
MAX_HISTORY_MESSAGES = 4


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0;
}

.subtitle {
    font-size: 17px;
    color: #777;
    margin-bottom: 25px;
}

.info-card {
    padding: 18px;
    border-radius: 15px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 15px;
}

.source-box {
    padding: 8px 12px;
    border-radius: 8px;
    background: rgba(128,128,128,0.08);
    font-size: 13px;
}

.small-text {
    font-size: 13px;
    color: #777;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

default_state = {

    "chunks": [],

    "chunk_sources": [],

    "documents": [],

    "vectorizer": None,

    "tfidf_matrix": None,

    "kb_ready": False,

    "chat_history": [],

    "answer_cache": {},

    "document_summary": "",

    "suggested_questions": [],

    "quota_error": False,

}

for key, value in default_state.items():

    if key not in st.session_state:

        st.session_state[key] = value


# =========================================================
# GEMINI CLIENT
# =========================================================

@st.cache_resource
def get_gemini_client():

    api_key = None

    # Streamlit Cloud Secrets
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")

    except Exception:
        api_key = None

    # Local environment fallback
    if not api_key:

        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:

        return None

    try:

        return genai.Client(
            api_key=api_key
        )

    except Exception:

        return None


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    if not text:
        return ""

    # Remove null characters
    text = text.replace("\x00", " ")

    # Normalize line breaks
    text = text.replace("\r", "\n")

    # Multiple spaces/newlines
    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove spaces before punctuation
    text = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        text
    )

    return text.strip()


# =========================================================
# CREATE TEXT CHUNKS
# =========================================================

def create_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

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

        st.error(
            f"TXT reading error: {e}"
        )

        return ""


# =========================================================
# READ PDF
# =========================================================

def read_pdf(uploaded_file):

    pages = []

    try:

        reader = PdfReader(uploaded_file)

        for page in reader.pages:

            try:

                text = page.extract_text()

                if text:

                    pages.append(text)

            except Exception:

                continue

        return clean_text(
            "\n".join(pages)
        )

    except Exception as e:

        st.error(
            f"PDF reading error: {e}"
        )

        return ""


# =========================================================
# PROCESS UPLOADED FILE
# =========================================================

def process_file(uploaded_file):

    name = uploaded_file.name.lower()

    if name.endswith(".txt"):

        text = read_txt(uploaded_file)

    elif name.endswith(".pdf"):

        text = read_pdf(uploaded_file)

    else:

        return []

    if not text:

        return []

    return create_chunks(text)


# =========================================================
# BUILD KNOWLEDGE BASE
# =========================================================

def build_knowledge_base(files):

    all_chunks = []

    all_sources = []

    document_names = []

    for uploaded_file in files:

        chunks = process_file(
            uploaded_file
        )

        if not chunks:
            continue

        document_names.append(
            uploaded_file.name
        )

        for chunk in chunks:

            all_chunks.append(chunk)

            all_sources.append(
                uploaded_file.name
            )

    if not all_chunks:

        return False, "No readable text found."

    # -----------------------------------------
    # Build TF-IDF only ONCE
    # -----------------------------------------

    try:

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=15000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        tfidf_matrix = vectorizer.fit_transform(
            all_chunks
        )

    except Exception as e:

        return False, f"TF-IDF error: {e}"

    # -----------------------------------------
    # Save everything
    # -----------------------------------------

    st.session_state.chunks = all_chunks

    st.session_state.chunk_sources = all_sources

    st.session_state.documents = document_names

    st.session_state.vectorizer = vectorizer

    st.session_state.tfidf_matrix = tfidf_matrix

    st.session_state.kb_ready = True

    # Clear old cache
    st.session_state.answer_cache = {}

    st.session_state.document_summary = ""

    st.session_state.suggested_questions = []

    return True, len(all_chunks)


# =========================================================
# RETRIEVE RELEVANT KNOWLEDGE
# =========================================================

def retrieve_knowledge(question):

    chunks = st.session_state.chunks

    vectorizer = st.session_state.vectorizer

    matrix = st.session_state.tfidf_matrix

    if not chunks:

        return []

    if vectorizer is None:

        return []

    if matrix is None:

        return []

    try:

        # Only transform the question.
        # We DO NOT fit TF-IDF again.

        question_vector = vectorizer.transform(
            [question]
        )

        scores = cosine_similarity(
            question_vector,
            matrix
        ).flatten()

        # Top results
        top_indices = scores.argsort()[::-1][:TOP_K]

        results = []

        for index in top_indices:

            score = float(
                scores[index]
            )

            results.append({

                "index": int(index),

                "score": score,

                "text": chunks[index],

                "source": st.session_state.chunk_sources[index]

            })

        return results

    except Exception:

        return []


# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(results):

    if not results:

        return ""

    context_parts = []

    current_length = 0

    for item in results:

        score = item["score"]

        if score < MIN_SIMILARITY:

            continue

        source = item["source"]

        text = item["text"]

        remaining = (
            MAX_CONTEXT_CHARS
            - current_length
        )

        if remaining <= 0:

            break

        text = text[:remaining]

        context_parts.append(
            f"""
SOURCE: {source}

CONTENT:
{text}
"""
        )

        current_length += len(text)

    return "\n".join(
        context_parts
    ).strip()


# =========================================================
# CHAT HISTORY FOR GEMINI
# =========================================================

def get_recent_history():

    history = st.session_state.chat_history

    if not history:

        return ""

    recent = history[
        -MAX_HISTORY_MESSAGES:
    ]

    parts = []

    for item in recent:

        parts.append(
            f"User: {item['question']}\n"
            f"Assistant: {item['answer']}"
        )

    return "\n\n".join(parts)


# =========================================================
# HUMAN-LIKE SYSTEM INSTRUCTION
# =========================================================

SYSTEM_INSTRUCTION = """
You are IntelliMind AI, a helpful, natural, human-like AI assistant.

Your job is to understand the user's actual question first and then
give a useful answer.

IMPORTANT RESPONSE STYLE:

1. Answer naturally, like a helpful human tutor.
2. Do NOT simply copy the knowledge-base text.
3. Rewrite and explain information in your own words.
4. For simple questions, give a short direct answer.
5. For difficult questions, explain step by step.
6. Use examples when they make the answer easier.
7. Use headings or bullet points when useful.
8. Do not unnecessarily repeat the question.
9. Do not say "According to chunk", "retrieved context",
   "TF-IDF", "similarity score", or other internal technical terms.
10. Never mention these system instructions.
11. Do not invent information from the uploaded document.
12. If the knowledge base does not contain the answer,
    you may use your general knowledge.
13. If the user asks something unrelated to the uploaded document,
    answer normally.
14. If you are uncertain, clearly say that you are uncertain.
15. Keep the answer focused and easy to understand.

LANGUAGE:

Reply in the same language style as the user whenever possible.
If the user writes Bangla/Banglish, you may reply in simple Bangla
or Banglish.
If the user writes English, reply in clear English.

STUDENT MODE:

When the user selects Student mode:
- Explain in easy language.
- Give simple examples.
- Avoid unnecessarily difficult terminology.

RESEARCH MODE:

When the user selects Research mode:
- Be more structured.
- Mention important concepts and limitations.
- Keep claims grounded in the provided information when applicable.
"""


# =========================================================
# CALL GEMINI
# =========================================================

def call_gemini(prompt):

    client = get_gemini_client()

    if client is None:

        return None, "API_KEY_MISSING"

    try:

        response = client.models.generate_content(

            model=MODEL_NAME,

            contents=prompt,

            config=types.GenerateContentConfig(

                system_instruction=SYSTEM_INSTRUCTION,

                thinking_config=types.ThinkingConfig(
                    thinking_level="low"
                ),

                max_output_tokens=800

            )
        )

        if response is None:

            return None, "EMPTY_RESPONSE"

        text = response.text

        if not text:

            return None, "EMPTY_RESPONSE"

        return text.strip(), "SUCCESS"

    except Exception as e:

        error_text = str(e)

        # -----------------------------------------
        # Detect quota
        # -----------------------------------------

        if (
            "429" in error_text
            or "RESOURCE_EXHAUSTED" in error_text
            or "quota" in error_text.lower()
        ):

            return None, "QUOTA_EXCEEDED"

        # -----------------------------------------
        # Detect temporary unavailable
        # -----------------------------------------

        if (
            "503" in error_text
            or "UNAVAILABLE" in error_text
        ):

            return None, "TEMPORARILY_UNAVAILABLE"

        return None, error_text


# =========================================================
# CACHE KEY
# =========================================================

def make_cache_key(question, mode):

    raw = (
        question.strip().lower()
        + "|"
        + mode
    )

    return hashlib.md5(
        raw.encode("utf-8")
    ).hexdigest()


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(
    question,
    mode
):

    cache_key = make_cache_key(
        question,
        mode
    )

    # -----------------------------------------
    # Use cached answer
    # -----------------------------------------

    if cache_key in st.session_state.answer_cache:

        cached = (
            st.session_state.answer_cache[
                cache_key
            ]
        )

        return cached

    # -----------------------------------------
    # Retrieve knowledge
    # -----------------------------------------

    results = retrieve_knowledge(
        question
    )

    context = build_context(
        results
    )

    best_score = 0

    if results:

        best_score = max(
            item["score"]
            for item in results
        )

    # -----------------------------------------
    # Recent conversation
    # -----------------------------------------

    history = get_recent_history()

    # -----------------------------------------
    # Build prompt
    # -----------------------------------------

    if context:

        prompt = f"""
RESPONSE MODE:
{mode}

RELEVANT KNOWLEDGE FROM USER'S DOCUMENTS:

{context}

RECENT CONVERSATION:

{history}

CURRENT USER QUESTION:

{question}

Now answer the user's current question naturally.

Use the document information when relevant.
Do not mention internal retrieval processes.
Do not copy the document word-for-word unless a short quotation
is necessary.

Give the answer directly.
"""

    else:

        prompt = f"""
RESPONSE MODE:
{mode}

There was no sufficiently relevant information found in the
uploaded knowledge base.

You may answer using your general knowledge.

RECENT CONVERSATION:

{history}

CURRENT USER QUESTION:

{question}

Answer naturally and directly.
"""

    # -----------------------------------------
    # Gemini
    # -----------------------------------------

    answer, status = call_gemini(
        prompt
    )

    if answer:

        if context:

            source = (
                f"📚 Knowledge Base + Gemini"
            )

        else:

            source = (
                f"🤖 Gemini General Knowledge"
            )

        result = (
            answer,
            source,
            best_score
        )

        # Save cache
        st.session_state.answer_cache[
            cache_key
        ] = result

        return result

    # =====================================================
    # GEMINI QUOTA FALLBACK
    # =====================================================

    if status == "QUOTA_EXCEEDED":

        st.session_state.quota_error = True

        if context:

            # Give useful answer from KB
            fallback_answer = (
                "Gemini API quota is currently exhausted, "
                "so I couldn't generate the AI response. "
                "However, I found relevant information in "
                "your uploaded document:\n\n"
                + context
            )

            return (
                fallback_answer,
                "📚 Knowledge Base only • Gemini quota exceeded",
                best_score
            )

        return (
            "⚠️ Gemini API quota is currently exhausted. "
            "Please try again after your quota resets.",
            "⚠️ Gemini quota exceeded",
            0
        )

    # =====================================================
    # TEMPORARY ERROR
    # =====================================================

    if status == "TEMPORARILY_UNAVAILABLE":

        if context:

            return (
                "Gemini is temporarily unavailable. "
                "Here is the relevant information I found "
                "in your knowledge base:\n\n"
                + context,
                "📚 Knowledge Base only • Gemini unavailable",
                best_score
            )

        return (
            "Gemini is temporarily unavailable. "
            "Please try again later.",
            "⚠️ Gemini unavailable",
            0
        )

    # =====================================================
    # API KEY ERROR
    # =====================================================

    if status == "API_KEY_MISSING":

        if context:

            return (
                "Gemini API key is not configured. "
                "Here is the relevant information from "
                "your knowledge base:\n\n"
                + context,
                "📚 Knowledge Base only",
                best_score
            )

        return (
            "⚠️ Gemini API key is not configured.",
            "⚠️ API key missing",
            0
        )

    # =====================================================
    # OTHER ERROR
    # =====================================================

    if context:

        return (
            "I couldn't generate the Gemini response right now.\n\n"
            "Relevant information from your document:\n\n"
            + context,
            "📚 Knowledge Base only",
            best_score
        )

    return (
        f"⚠️ Gemini error:\n{status}",
        "❌ Gemini Error",
        0
    )


# =========================================================
# TEST GEMINI
# =========================================================

def test_gemini():

    client = get_gemini_client()

    if client is None:

        st.error(
            "❌ Gemini API key is not configured."
        )

        return

    try:

        response = client.models.generate_content(

            model=MODEL_NAME,

            contents="Reply with exactly: Connection successful.",

            config=types.GenerateContentConfig(

                thinking_config=types.ThinkingConfig(
                    thinking_level="low"
                ),

                max_output_tokens=30

            )
        )

        if response and response.text:

            st.success(
                "✅ Gemini connection successful."
            )

        else:

            st.warning(
                "⚠️ Gemini returned an empty response."
            )

    except Exception as e:

        error = str(e)

        if (
            "429" in error
            or "RESOURCE_EXHAUSTED" in error
            or "quota" in error.lower()
        ):

            st.error(
                "❌ Gemini quota exceeded."
            )

            st.info(
                "Your API project has reached its current "
                "Gemini request quota."
            )

        else:

            st.error(
                f"❌ Gemini error:\n{error}"
            )


# =========================================================
# DOCUMENT SUMMARY
# =========================================================

def generate_summary():

    chunks = st.session_state.chunks

    if not chunks:

        return (
            "No document is available."
        )

    # Keep prompt reasonable
    text = "\n\n".join(
        chunks[:8]
    )

    text = text[:7000]

    prompt = f"""
Summarize the following document naturally.

Give:

1. Main topic
2. Important points
3. Key concepts
4. Short conclusion

Use simple language.

DOCUMENT:

{text}
"""

    answer, status = call_gemini(
        prompt
    )

    if answer:

        return answer

    if status == "QUOTA_EXCEEDED":

        return (
            "⚠️ Gemini quota is currently exhausted. "
            "Summary cannot be generated right now."
        )

    return (
        "⚠️ Summary could not be generated."
    )


# =========================================================
# SUGGESTED QUESTIONS
# =========================================================

def generate_questions():

    chunks = st.session_state.chunks

    if not chunks:

        return []

    text = "\n\n".join(
        chunks[:6]
    )

    text = text[:6000]

    prompt = f"""
Create 5 useful questions that a student could ask about
this document.

Return ONLY the questions.

Do not add explanations.

DOCUMENT:

{text}
"""

    answer, status = call_gemini(
        prompt
    )

    if not answer:

        return []

    questions = []

    for line in answer.splitlines():

        line = line.strip()

        line = re.sub(
            r"^[0-9]+[\.\)\-\:]\s*",
            "",
            line
        )

        if line:

            questions.append(
                line
            )

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
    'Human-like AI Assistant with PDF & TXT Knowledge Base'
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

    # ---------------------------------------------
    # Upload
    # ---------------------------------------------

    st.subheader(
        "📄 Knowledge Base"
    )

    uploaded_files = st.file_uploader(

        "Upload PDF or TXT files",

        type=[
            "pdf",
            "txt"
        ],

        accept_multiple_files=True

    )

    if st.button(
        "🚀 Build Knowledge Base",
        use_container_width=True
    ):

        if not uploaded_files:

            st.warning(
                "Please upload a PDF or TXT file."
            )

        else:

            with st.spinner(
                "Building knowledge base..."
            ):

                success, result = (
                    build_knowledge_base(
                        uploaded_files
                    )
                )

            if success:

                st.success(
                    f"✅ Knowledge Base ready!\n\n"
                    f"Documents: "
                    f"{len(st.session_state.documents)}\n\n"
                    f"Chunks: {result}"
                )

            else:

                st.error(
                    f"❌ {result}"
                )

    st.divider()

    # ---------------------------------------------
    # Status
    # ---------------------------------------------

    st.subheader(
        "🔌 System Status"
    )

    if get_gemini_client():

        st.success(
            "Gemini API: Connected"
        )

    else:

        st.error(
            "Gemini API: Not Connected"
        )

    if st.session_state.kb_ready:

        st.success(
            f"Knowledge Base: "
            f"{len(st.session_state.chunks)} chunks"
        )

    else:

        st.info(
            "Knowledge Base: Empty"
        )

    st.caption(
        f"Model: {MODEL_NAME}"
    )

    st.divider()

    # ---------------------------------------------
    # Test
    # ---------------------------------------------

    if st.button(
        "🧪 Test Gemini",
        use_container_width=True
    ):

        test_gemini()

    # ---------------------------------------------
    # Clear
    # ---------------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.session_state.answer_cache = {}

        st.rerun()


# =========================================================
# FEATURE CARDS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
        <div class="info-card">
        <h3>📚 Knowledge Base</h3>
        <p>Ask questions from your uploaded PDF or TXT files.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        """
        <div class="info-card">
        <h3>⚡ Fast Retrieval</h3>
        <p>TF-IDF is built once instead of rebuilding for every question.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        """
        <div class="info-card">
        <h3>🧠 Human-like Answers</h3>
        <p>Gemini rewrites and explains information naturally.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# DOCUMENT TOOLS
# =========================================================

if st.session_state.kb_ready:

    st.divider()

    st.subheader(
        "🛠️ Document Tools"
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "📝 Generate Summary",
            use_container_width=True
        ):

            with st.spinner(
                "Generating summary..."
            ):

                summary = (
                    generate_summary()
                )

            st.session_state.document_summary = summary

    with col2:

        if st.button(
            "❓ Suggested Questions",
            use_container_width=True
        ):

            with st.spinner(
                "Generating questions..."
            ):

                questions = (
                    generate_questions()
                )

            st.session_state.suggested_questions = questions


# =========================================================
# SUMMARY
# =========================================================

if st.session_state.document_summary:

    st.divider()

    st.subheader(
        "📝 Document Summary"
    )

    st.write(
        st.session_state.document_summary
    )


# =========================================================
# SUGGESTED QUESTIONS
# =========================================================

if st.session_state.suggested_questions:

    st.subheader(
        "❓ Suggested Questions"
    )

    for question in (
        st.session_state.suggested_questions
    ):

        st.markdown(
            f"- {question}"
        )


# =========================================================
# CHAT
# =========================================================

st.divider()

st.subheader(
    "💬 Chat with IntelliMind"
)


# ---------------------------------------------
# Display previous chat
# ---------------------------------------------

for item in st.session_state.chat_history:

    with st.chat_message("user"):

        st.write(
            item["question"]
        )

    with st.chat_message("assistant"):

        st.write(
            item["answer"]
        )

        st.caption(
            f"{item['source']} "
            f"| Similarity: "
            f"{item['score']:.2f}"
        )


# =========================================================
# USER INPUT
# =========================================================

question = st.chat_input(
    "Ask anything..."
)


if question:

    question = question.strip()

    if question:

        # ---------------------------------------------
        # User message
        # ---------------------------------------------

        with st.chat_message("user"):

            st.write(
                question
            )

        # ---------------------------------------------
        # AI response
        # ---------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Thinking..."
            ):

                answer, source, score = (
                    generate_answer(
                        question,
                        mode
                    )
                )

            st.write(
                answer
            )

            st.caption(
                f"{source} "
                f"| Similarity: "
                f"{score:.2f}"
            )

        # ---------------------------------------------
        # Save history
        # ---------------------------------------------

        st.session_state.chat_history.append({

            "question": question,

            "answer": answer,

            "source": source,

            "score": score

        })
