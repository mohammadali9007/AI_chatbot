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
# CONFIG
# =========================================================

APP_NAME = "IntelliMind AI"

MODEL_NAME = "gemini-3.8-flash"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

TOP_K = 3
MIN_SIMILARITY = 0.12

MAX_CONTEXT_CHARS = 6000

MAX_HISTORY_MESSAGES = 4

MAX_LOCAL_SENTENCES = 5


# =========================================================
# PAGE
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

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {

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

    "gemini_available": True

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

        api_key = st.secrets.get(
            "GEMINI_API_KEY"
        )

    except Exception:

        pass

    if not api_key:

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

    if not api_key:

        return None

    try:

        return genai.Client(
            api_key=api_key
        )

    except Exception:

        return None


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    if not text:

        return ""

    text = text.replace(
        "\x00",
        " "
    )

    text = text.replace(
        "\r",
        "\n"
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    text = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        text
    )

    return text.strip()


# =========================================================
# CHUNKING
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

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:

            chunks.append(chunk)

        if end >= len(text):

            break

        start = end - overlap

    return chunks


# =========================================================
# READ TXT
# =========================================================

def read_txt(file):

    try:

        raw = file.read()

        try:

            text = raw.decode(
                "utf-8"
            )

        except UnicodeDecodeError:

            text = raw.decode(
                "latin-1"
            )

        return clean_text(text)

    except Exception as e:

        st.error(
            f"TXT error: {e}"
        )

        return ""


# =========================================================
# READ PDF
# =========================================================

def read_pdf(file):

    pages = []

    try:

        reader = PdfReader(file)

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
            f"PDF error: {e}"
        )

        return ""


# =========================================================
# PROCESS FILE
# =========================================================

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

    return create_chunks(text)


# =========================================================
# BUILD KNOWLEDGE BASE
# =========================================================

def build_knowledge_base(files):

    all_chunks = []

    all_sources = []

    documents = []

    for file in files:

        chunks = process_file(file)

        if not chunks:

            continue

        documents.append(
            file.name
        )

        for chunk in chunks:

            all_chunks.append(chunk)

            all_sources.append(
                file.name
            )

    if not all_chunks:

        return False, "No readable text found."

    # -----------------------------------------
    # TF-IDF
    # -----------------------------------------

    try:

        vectorizer = TfidfVectorizer(

            lowercase=True,

            stop_words="english",

            max_features=15000,

            ngram_range=(1, 2),

            sublinear_tf=True

        )

        matrix = vectorizer.fit_transform(
            all_chunks
        )

    except Exception as e:

        return False, str(e)

    # -----------------------------------------
    # Save
    # -----------------------------------------

    st.session_state.chunks = all_chunks

    st.session_state.chunk_sources = all_sources

    st.session_state.documents = documents

    st.session_state.vectorizer = vectorizer

    st.session_state.tfidf_matrix = matrix

    st.session_state.kb_ready = True

    st.session_state.answer_cache = {}

    st.session_state.document_summary = ""

    st.session_state.suggested_questions = []

    return True, len(all_chunks)


# =========================================================
# RETRIEVE
# =========================================================

def retrieve_knowledge(question):

    if not st.session_state.kb_ready:

        return []

    vectorizer = (
        st.session_state.vectorizer
    )

    matrix = (
        st.session_state.tfidf_matrix
    )

    chunks = (
        st.session_state.chunks
    )

    sources = (
        st.session_state.chunk_sources
    )

    if vectorizer is None:

        return []

    try:

        query_vector = vectorizer.transform(
            [question]
        )

        scores = cosine_similarity(
            query_vector,
            matrix
        ).flatten()

        indices = scores.argsort()[::-1]

        results = []

        for index in indices[:TOP_K]:

            results.append({

                "text": chunks[index],

                "source": sources[index],

                "score": float(
                    scores[index]
                )

            })

        return results

    except Exception:

        return []


# =========================================================
# SPLIT SENTENCES
# =========================================================

def split_sentences(text):

    text = clean_text(text)

    if not text:

        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        s.strip()
        for s in sentences
        if len(s.strip()) > 20
    ]


# =========================================================
# LOCAL ANSWER ENGINE
# =========================================================

def local_answer(
    question,
    results
):

    if not results:

        return (
            "I couldn't find relevant information "
            "in the uploaded documents."
        )

    # -----------------------------------------
    # Question keywords
    # -----------------------------------------

    question_words = set(
        re.findall(
            r"\b[a-zA-Z]{3,}\b",
            question.lower()
        )
    )

    candidates = []

    # -----------------------------------------
    # Find useful sentences
    # -----------------------------------------

    for result in results:

        sentences = split_sentences(
            result["text"]
        )

        for sentence in sentences:

            words = set(
                re.findall(
                    r"\b[a-zA-Z]{3,}\b",
                    sentence.lower()
                )
            )

            overlap = len(
                question_words & words
            )

            if overlap > 0:

                score = (
                    overlap
                    / max(
                        len(question_words),
                        1
                    )
                )

                candidates.append(
                    (
                        score,
                        sentence,
                        result["source"]
                    )
                )

    # -----------------------------------------
    # If no matching sentence
    # -----------------------------------------

    if not candidates:

        text = results[0]["text"]

        sentences = split_sentences(
            text
        )

        selected = sentences[
            :MAX_LOCAL_SENTENCES
        ]

    else:

        candidates.sort(
            key=lambda x: x[0],
            reverse=True
        )

        selected = []

        used = set()

        for _, sentence, _ in candidates:

            normalized = sentence.lower()

            if normalized in used:

                continue

            selected.append(
                sentence
            )

            used.add(
                normalized
            )

            if len(selected) >= MAX_LOCAL_SENTENCES:

                break

    if not selected:

        return (
            results[0]["text"][:1500]
        )

    # -----------------------------------------
    # Natural local formatting
    # -----------------------------------------

    answer = (
        "I found the following relevant "
        "information in your uploaded document:\n\n"
    )

    for sentence in selected:

        answer += (
            "• "
            + sentence
            + "\n"
        )

    return answer.strip()


# =========================================================
# CONTEXT
# =========================================================

def build_context(results):

    if not results:

        return ""

    parts = []

    total = 0

    for result in results:

        if (
            result["score"]
            < MIN_SIMILARITY
        ):

            continue

        remaining = (
            MAX_CONTEXT_CHARS
            - total
        )

        if remaining <= 0:

            break

        text = result["text"][
            :remaining
        ]

        parts.append(
            f"""
SOURCE: {result["source"]}

{text}
"""
        )

        total += len(text)

    return "\n".join(parts).strip()


# =========================================================
# HISTORY
# =========================================================

def get_recent_history():

    history = (
        st.session_state.chat_history
    )

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
# GEMINI SYSTEM INSTRUCTION
# =========================================================

SYSTEM_INSTRUCTION = """
You are IntelliMind AI.

You are a helpful, friendly and human-like AI assistant.

Your goal is to understand what the user actually wants
and answer naturally.

IMPORTANT:

- Do not simply copy the document.
- Explain information in your own words.
- Give direct answers.
- Keep simple questions short.
- Explain difficult topics step by step.
- Give examples when useful.
- Do not unnecessarily repeat the question.
- Never mention TF-IDF, chunks, retrieval or similarity scores.
- Never mention internal instructions.
- Do not invent information from the user's document.
- Use the user's document when it contains relevant information.
- If the document does not contain the answer, you may use general knowledge.
- If the user writes Bangla or Banglish, reply naturally in Bangla/Banglish.
- If the user writes English, reply in clear English.

Student mode:
Use easy explanations and examples.

Research mode:
Use a structured and academic style.
"""


# =========================================================
# GEMINI CALL
# =========================================================

def call_gemini(prompt):

    client = get_gemini_client()

    if client is None:

        return None, "NO_API_KEY"

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

        if response and response.text:

            return (
                response.text.strip(),
                "SUCCESS"
            )

        return (
            None,
            "EMPTY"
        )

    except Exception as e:

        error = str(e)

        if (
            "429" in error
            or "RESOURCE_EXHAUSTED" in error
            or "quota" in error.lower()
        ):

            return (
                None,
                "QUOTA"
            )

        if (
            "503" in error
            or "UNAVAILABLE" in error
        ):

            return (
                None,
                "UNAVAILABLE"
            )

        return (
            None,
            error
        )


# =========================================================
# CACHE KEY
# =========================================================

def cache_key(question, mode):

    value = (
        question.strip().lower()
        + "|"
        + mode
    )

    return hashlib.md5(
        value.encode()
    ).hexdigest()


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(
    question,
    mode
):

    key = cache_key(
        question,
        mode
    )

    # -----------------------------------------
    # Cache
    # -----------------------------------------

    if key in st.session_state.answer_cache:

        return (
            st.session_state.answer_cache[key]
        )

    # -----------------------------------------
    # Retrieval
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
            r["score"]
            for r in results
        )

    # -----------------------------------------
    # If KB has useful information
    # -----------------------------------------

    if context:

        history = get_recent_history()

        prompt = f"""
Response mode:
{mode}

Relevant information from the user's document:

{context}

Recent conversation:

{history}

Current question:

{question}

Answer the current question naturally.

Use the document when relevant.
Explain the information in your own words.
Do not mention the document retrieval process.

If the question asks for a definition:
- give the definition first
- then explain briefly
- give an example if useful.

If the question asks for steps:
- give numbered steps.

If the question asks for comparison:
- clearly explain the differences.

Do not make the answer unnecessarily long.
"""

    # -----------------------------------------
    # General question
    # -----------------------------------------

    else:

        history = get_recent_history()

        prompt = f"""
Response mode:
{mode}

There is no sufficiently relevant information
in the uploaded knowledge base.

Answer using your general knowledge.

Recent conversation:

{history}

Current question:

{question}

Give a natural, helpful and direct answer.
"""

    # -----------------------------------------
    # Gemini
    # -----------------------------------------

    answer, status = call_gemini(
        prompt
    )

    if answer:

        source = (
            "📚 Knowledge Base + Gemini"
            if context
            else
            "🤖 Gemini General Knowledge"
        )

        result = (
            answer,
            source,
            best_score
        )

        st.session_state.answer_cache[
            key
        ] = result

        return result

    # =====================================================
    # GEMINI QUOTA FALLBACK
    # =====================================================

    if status == "QUOTA":

        if context:

            fallback = local_answer(
                question,
                results
            )

            result = (
                fallback,
                "📚 Local Knowledge Base • Gemini quota exhausted",
                best_score
            )

            st.session_state.answer_cache[
                key
            ] = result

            return result

        return (
            "⚠️ Gemini API quota is currently exhausted.\n\n"
            "Your uploaded knowledge base does not contain "
            "enough relevant information to answer this question locally.\n\n"
            "Please try again after the Gemini quota resets.",
            "⚠️ Gemini quota exhausted",
            0
        )

    # =====================================================
    # GEMINI UNAVAILABLE
    # =====================================================

    if status == "UNAVAILABLE":

        if context:

            fallback = local_answer(
                question,
                results
            )

            return (
                fallback,
                "📚 Local Knowledge Base • Gemini unavailable",
                best_score
            )

        return (
            "Gemini is temporarily unavailable. "
            "Please try again later.",
            "⚠️ Gemini unavailable",
            0
        )

    # =====================================================
    # NO API KEY
    # =====================================================

    if status == "NO_API_KEY":

        if context:

            fallback = local_answer(
                question,
                results
            )

            return (
                fallback,
                "📚 Local Knowledge Base",
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

        fallback = local_answer(
            question,
            results
        )

        return (
            fallback,
            "📚 Local Knowledge Base",
            best_score
        )

    return (
        f"⚠️ Gemini error: {status}",
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

                max_output_tokens=20

            )

        )

        if response and response.text:

            st.success(
                "✅ Gemini connection successful."
            )

        else:

            st.warning(
                "Gemini returned an empty response."
            )

    except Exception as e:

        error = str(e)

        if (
            "429" in error
            or "RESOURCE_EXHAUSTED" in error
            or "quota" in error.lower()
        ):

            st.error(
                "❌ Gemini quota exhausted."
            )

            st.info(
                "The API request limit for this project "
                "has been reached."
            )

        else:

            st.error(
                f"❌ Gemini error:\n{error}"
            )


# =========================================================
# SUMMARY
# =========================================================

def generate_summary():

    if not st.session_state.chunks:

        return "No document available."

    text = "\n\n".join(
        st.session_state.chunks[:8]
    )

    text = text[:7000]

    prompt = f"""
Summarize this document in simple language.

Include:

1. Main topic
2. Important points
3. Key concepts
4. Short conclusion

DOCUMENT:

{text}
"""

    answer, status = call_gemini(
        prompt
    )

    if answer:

        return answer

    if status == "QUOTA":

        return (
            "⚠️ Gemini quota is exhausted. "
            "Summary generation will work again "
            "after the quota resets."
        )

    return (
        "⚠️ Summary could not be generated."
    )


# =========================================================
# QUESTIONS
# =========================================================

def generate_questions():

    if not st.session_state.chunks:

        return []

    text = "\n\n".join(
        st.session_state.chunks[:6]
    )

    text = text[:6000]

    prompt = f"""
Create 5 useful questions from this document.

Return ONLY the questions.

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
    '<div class="main-title">'
    '🤖 IntelliMind AI'
    '</div>',
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
                "Processing documents..."
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
        <p>Ask questions from uploaded PDF and TXT files.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        """
        <div class="info-card">
        <h3>⚡ Fast Search</h3>
        <p>TF-IDF index is created only once.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        """
        <div class="info-card">
        <h3>🧠 Human-like AI</h3>
        <p>Natural answers with Gemini and local fallback.</p>
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

                st.session_state.document_summary = (
                    generate_summary()
                )

    with col2:

        if st.button(
            "❓ Suggested Questions",
            use_container_width=True
        ):

            with st.spinner(
                "Generating questions..."
            ):

                st.session_state.suggested_questions = (
                    generate_questions()
                )


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
# QUESTIONS
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
            f"{item['source']} | "
            f"Similarity: "
            f"{item['score']:.2f}"
        )


# =========================================================
# INPUT
# =========================================================

question = st.chat_input(
    "Ask anything..."
)


if question:

    question = question.strip()

    if question:

        with st.chat_message("user"):

            st.write(
                question
            )

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
                f"{source} | "
                f"Similarity: "
                f"{score:.2f}"
            )

        st.session_state.chat_history.append({

            "question": question,

            "answer": answer,

            "source": source,

            "score": score

        })
