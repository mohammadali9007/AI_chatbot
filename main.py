import os
import re
import time

import streamlit as st
from pypdf import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from google import genai


# ============================================================
# 1. CONFIGURATION
# ============================================================

APP_NAME = "IntelliMind AI"

# ============================================================
# GEMINI MODEL FALLBACK SYSTEM
# ============================================================

MODEL_NAMES = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

# Text chunk settings
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# Retrieval settings
TOP_K = 5
MIN_SIMILARITY = 0.25


# ============================================================
# 2. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 45px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #777;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .source-card {
        padding: 12px;
        border-radius: 10px;
        background: rgba(100,100,100,0.08);
        margin-top: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 4. SESSION STATE
# ============================================================

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "document_summary" not in st.session_state:
    st.session_state.document_summary = ""

if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []


# ============================================================
# 5. GEMINI CLIENT
# ============================================================

def get_gemini_client():

    api_key = None

    # --------------------------------------------------------
    # Streamlit Cloud Secrets
    # --------------------------------------------------------

    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass

    # --------------------------------------------------------
    # Local environment variable
    # --------------------------------------------------------

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    # --------------------------------------------------------
    # No API key
    # --------------------------------------------------------

    if not api_key:
        return None

    # --------------------------------------------------------
    # Create Gemini client
    # --------------------------------------------------------

    try:

        return genai.Client(
            api_key=api_key
        )

    except Exception as e:

        st.error(
            f"Gemini client error: {e}"
        )

        return None


client = get_gemini_client()


# ============================================================
# 6. TEXT CLEANING
# ============================================================

def clean_text(text):

    if not text:
        return ""

    # Remove null characters
    text = text.replace(
        "\x00",
        " "
    )

    # Normalize spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 7. TEXT CHUNKING
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

        # ----------------------------------------------------
        # Try to end at sentence
        # ----------------------------------------------------

        if end < len(text):

            possible_breaks = [
                chunk.rfind("."),
                chunk.rfind("?"),
                chunk.rfind("!"),
            ]

            best_break = max(
                possible_breaks
            )

            if best_break > CHUNK_SIZE * 0.55:

                chunk = chunk[
                    :best_break + 1
                ]

                end = start + len(chunk)

        # ----------------------------------------------------
        # Save valid chunk
        # ----------------------------------------------------

        if len(chunk.strip()) > 40:

            chunks.append(
                chunk.strip()
            )

        # ----------------------------------------------------
        # Move forward
        # ----------------------------------------------------

        next_start = end - CHUNK_OVERLAP

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


# ============================================================
# 8. READ TXT
# ============================================================

def read_txt(file):

    try:

        raw = file.read()

        text = raw.decode(
            "utf-8",
            errors="ignore"
        )

        return clean_text(text)

    except Exception as e:

        st.error(
            f"TXT reading error: {e}"
        )

        return ""


# ============================================================
# 9. READ PDF
# ============================================================

def read_pdf(file):

    try:

        reader = PdfReader(file)

        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            try:

                page_text = page.extract_text()

            except Exception:

                page_text = ""

            if page_text:

                pages.append(
                    f"[Page {page_number}] "
                    f"{page_text}"
                )

        return clean_text(
            "\n".join(pages)
        )

    except Exception as e:

        st.error(
            f"PDF reading error: {e}"
        )

        return ""


# ============================================================
# 10. PROCESS FILE
# ============================================================

def process_file(file):

    filename = file.name.lower()

    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------

    if filename.endswith(".txt"):

        text = read_txt(file)

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    elif filename.endswith(".pdf"):

        text = read_pdf(file)

    else:

        return []

    # --------------------------------------------------------
    # Empty file
    # --------------------------------------------------------

    if not text:

        return []

    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    chunks = create_chunks(text)

    result = []

    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        result.append(
            {
                "text": chunk,
                "source": file.name,
                "chunk_id": index
            }
        )

    return result


# ============================================================
# 11. RETRIEVE KNOWLEDGE
# ============================================================

def retrieve_knowledge(question):

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

        ranked_indices = scores.argsort()[::-1]

        results = []

        for index in ranked_indices[:TOP_K]:

            results.append(
                {
                    "text": chunks[index]["text"],
                    "source": chunks[index]["source"],
                    "chunk_id": chunks[index]["chunk_id"],
                    "score": float(scores[index])
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
# 12. BUILD CONTEXT
# ============================================================

def build_context(results):

    useful_results = [
        item
        for item in results
        if item["score"] >= MIN_SIMILARITY
    ]

    if not useful_results:

        return ""

    context_parts = []

    for item in useful_results:

        context_parts.append(
            f"""
SOURCE: {item['source']}
CHUNK: {item['chunk_id']}
RELEVANCE SCORE: {item['score']:.2f}

{item['text']}
"""
        )

    return "\n------------------------\n".join(
        context_parts
    )


# ============================================================
# 13. GEMINI MODEL CALL WITH AUTOMATIC FALLBACK
# ============================================================

def call_gemini_with_fallback(prompt):

    if client is None:

        return (
            None,
            "Gemini API is not configured."
        )

    errors = []

    # --------------------------------------------------------
    # Try every model
    # --------------------------------------------------------

    for model_name in MODEL_NAMES:

        try:

            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            # ------------------------------------------------
            # Successful response
            # ------------------------------------------------

            if response is not None:

                answer = getattr(
                    response,
                    "text",
                    None
                )

                if answer:

                    return (
                        answer.strip(),
                        model_name
                    )

            errors.append(
                f"{model_name}: Empty response"
            )

        except Exception as e:

            error = str(e)

            errors.append(
                f"{model_name}: {error}"
            )

            # ------------------------------------------------
            # 503 = temporary server problem
            # ------------------------------------------------

            if (
                "503" in error
                or "UNAVAILABLE" in error
            ):

                continue

            # ------------------------------------------------
            # 429 = rate limit
            # ------------------------------------------------

            if "429" in error:

                continue

            # ------------------------------------------------
            # 404 = model unavailable
            # ------------------------------------------------

            if "404" in error:

                continue

            # ------------------------------------------------
            # 401 / 403 = API key problem
            # ------------------------------------------------

            if (
                "401" in error
                or "403" in error
            ):

                # Try next model anyway
                continue

            # ------------------------------------------------
            # Other errors
            # ------------------------------------------------

            continue

    # --------------------------------------------------------
    # All models failed
    # --------------------------------------------------------

    error_message = (
        "All Gemini models failed.\n\n"
        + "\n\n".join(errors)
    )

    return (
        None,
        error_message
    )


# ============================================================
# 14. ASK GEMINI
# ============================================================

def ask_gemini(
    question,
    context="",
    mode="Normal"
):

    # --------------------------------------------------------
    # Check client
    # --------------------------------------------------------

    if client is None:

        return (
            None,
            "Gemini API is not configured."
        )

    # --------------------------------------------------------
    # Mode instructions
    # --------------------------------------------------------

    if "Student" in mode:

        mode_instruction = """
Explain the answer using very simple English.

Use:
- simple explanation
- simple examples
- important exam points when useful
"""

    elif "Research" in mode:

        mode_instruction = """
Give a research-oriented answer.

Focus on:
- Objective
- Methodology
- Findings
- Limitations
- Research implications
"""

    else:

        mode_instruction = """
Give a clear and well-structured answer.

Use headings and bullet points when useful.
"""

    # ========================================================
    # KNOWLEDGE BASE ANSWER
    # ========================================================

    if context:

        prompt = f"""
You are IntelliMind AI,
an AI-powered Knowledge and Research Assistant.

USER QUESTION:
{question}

RETRIEVED INFORMATION FROM USER DOCUMENTS:
{context}

RESPONSE MODE:
{mode}

IMPORTANT RULES:

1. The uploaded knowledge is the primary source.
2. Use the retrieved information whenever possible.
3. Do not invent facts that are not supported by the document.
4. If the document only partially answers the question,
   clearly explain what is available and what is missing.
5. Do not claim something came from the document
   unless it is actually supported by the retrieved text.
6. Do not simply copy the entire document.
7. Answer the user's actual question directly.
8. Keep the answer clear and easy to understand.

{mode_instruction}

Now answer the user's question.
"""

        source_prefix = "📚 Knowledge Base + Gemini"

    # ========================================================
    # GENERAL GEMINI ANSWER
    # ========================================================

    else:

        prompt = f"""
You are IntelliMind AI,
a general-purpose AI assistant.

USER QUESTION:
{question}

There is no sufficiently relevant information
in the uploaded knowledge base.

Answer using your general knowledge.

IMPORTANT RULES:

1. Do not pretend the answer came from uploaded documents.
2. Give a useful and clear answer.
3. Do not invent document information.
4. If information may be uncertain or time-sensitive,
   mention the limitation.
5. Keep the answer easy to understand.

{mode_instruction}

Now answer the user's question.
"""

        source_prefix = "🌐 Gemini General AI"

    # ========================================================
    # CALL GEMINI WITH FALLBACK
    # ========================================================

    answer, model_used = call_gemini_with_fallback(
        prompt
    )

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    if answer:

        return (
            answer,
            f"{source_prefix} ({model_used})"
        )

    # --------------------------------------------------------
    # Failed
    # --------------------------------------------------------

    return (
        None,
        model_used
    )


# ============================================================
# 15. MAIN ANSWER ENGINE
# ============================================================

def generate_answer(
    question,
    mode
):

    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    results, best_score = retrieve_knowledge(
        question
    )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = build_context(
        results
    )

    # --------------------------------------------------------
    # Ask Gemini
    # --------------------------------------------------------

    answer, source = ask_gemini(
        question,
        context,
        mode
    )

    # --------------------------------------------------------
    # Gemini success
    # --------------------------------------------------------

    if answer:

        return (
            answer,
            source,
            best_score,
            results
        )

    # --------------------------------------------------------
    # Knowledge base fallback
    # --------------------------------------------------------

    if context:

        fallback = f"""
### ⚠️ Gemini could not generate the final answer

Relevant information was found in your uploaded
knowledge base.

**Gemini error/details:**

{source}

Please review the retrieved sources below.
"""

        return (
            fallback,
            "📚 Knowledge Base Fallback",
            best_score,
            results
        )

    # --------------------------------------------------------
    # Final fallback
    # --------------------------------------------------------

    fallback = f"""
### ❌ Gemini could not generate an answer

**Details:**

{source}

Please try again after a few moments.
"""

    return (
        fallback,
        "⚠️ System Fallback",
        best_score,
        results
    )


# ============================================================
# 16. TEST GEMINI WITH FALLBACK
# ============================================================

def test_gemini():

    if client is None:

        return (
            "❌ Gemini client is not connected."
        )

    prompt = (
        "Reply with exactly: "
        "Gemini connection successful."
    )

    answer, model_used = call_gemini_with_fallback(
        prompt
    )

    if answer:

        return (
            f"Gemini connection successful.\n"
            f"Model used: {model_used}"
        )

    return (
        f"❌ Gemini failed.\n\n"
        f"{model_used}"
    )


# ============================================================
# 17. DOCUMENT SUMMARY
# ============================================================

def generate_summary():

    if not st.session_state.chunks:

        return "Please upload a document first."

    if client is None:

        return "Gemini API is not connected."

    context = "\n\n".join(
        item["text"]
        for item in st.session_state.chunks[:20]
    )

    prompt = f"""
You are an AI research assistant.

Analyze the following uploaded document content.

DOCUMENT CONTENT:

{context}

Create a structured summary with:

1. Title / Topic
2. Main Objective
3. Key Concepts
4. Methodology or Approach
5. Important Findings
6. Limitations
7. Possible Future Work
8. Short Conclusion

Rules:

- Do not invent information.
- If something is unavailable,
  write "Not mentioned".
- Keep the summary clear and concise.
"""

    answer, model_used = call_gemini_with_fallback(
        prompt
    )

    if answer:

        return (
            f"{answer}\n\n"
            f"---\n"
            f"Model used: `{model_used}`"
        )

    return (
        f"❌ Could not generate summary.\n\n"
        f"{model_used}"
    )


# ============================================================
# 18. SUGGEST QUESTIONS
# ============================================================

def generate_questions():

    if not st.session_state.chunks:

        return []

    if client is None:

        return []

    context = "\n\n".join(
        item["text"]
        for item in st.session_state.chunks[:10]
    )

    prompt = f"""
Based on the following uploaded document:

{context}

Generate 5 useful questions that a
student or researcher could ask.

Rules:

- Questions must be related to the document.
- Keep them meaningful.
- Return only 5 questions.
- Number them from 1 to 5.
"""

    answer, model_used = call_gemini_with_fallback(
        prompt
    )

    if not answer:

        return []

    lines = answer.split("\n")

    questions = []

    for line in lines:

        line = re.sub(
            r"^\s*\d+[\.\)]\s*",
            "",
            line
        ).strip()

        if len(line) > 10:

            questions.append(line)

    return questions[:5]


# ============================================================
# 19. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🧠 IntelliMind AI")

    st.subheader("📂 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    # --------------------------------------------------------
    # Build Knowledge Base
    # --------------------------------------------------------

    if st.button(
        "🚀 Build Knowledge Base",
        use_container_width=True
    ):

        if not uploaded_files:

            st.warning(
                "Please upload a PDF or TXT file."
            )

        else:

            all_chunks = []
            names = []

            progress = st.progress(0)

            for i, file in enumerate(
                uploaded_files
            ):

                chunks = process_file(file)

                if chunks:

                    all_chunks.extend(
                        chunks
                    )

                    names.append(
                        file.name
                    )

                progress.progress(
                    (i + 1) / len(uploaded_files)
                )

            st.session_state.chunks = (
                all_chunks
            )

            st.session_state.documents = (
                names
            )

            st.session_state.document_summary = ""

            st.session_state.suggested_questions = []

            if all_chunks:

                st.success(
                    f"Knowledge Base created! "
                    f"{len(all_chunks)} chunks."
                )

            else:

                st.error(
                    "No readable text was found."
                )

    st.divider()

    # ========================================================
    # AI MODE
    # ========================================================

    st.subheader("🎯 AI Mode")

    mode = st.selectbox(
        "Choose response mode",
        [
            "Normal",
            "🎓 Student",
            "🔬 Research"
        ]
    )

    st.divider()

    # ========================================================
    # SYSTEM STATUS
    # ========================================================

    st.subheader("📊 System Status")

    st.write(
        f"📄 Documents: "
        f"**{len(st.session_state.documents)}**"
    )

    st.write(
        f"🧩 Text Chunks: "
        f"**{len(st.session_state.chunks)}**"
    )

    st.write(
        "🤖 Models:"
    )

    for model_name in MODEL_NAMES:

        st.caption(
            f"• {model_name}"
        )

    if client:

        st.success(
            "🟢 Gemini Connected"
        )

    else:

        st.error(
            "🔴 Gemini Not Connected"
        )

    # ========================================================
    # TEST GEMINI
    # ========================================================

    st.divider()

    if st.button(
        "🧪 Test Gemini",
        use_container_width=True
    ):

        with st.spinner(
            "Testing Gemini models..."
        ):

            test_result = test_gemini()

        if test_result.startswith(
            "Gemini connection successful"
        ):

            st.success(
                test_result
            )

        else:

            st.error(
                test_result
            )

    # ========================================================
    # CLEAR CHAT
    # ========================================================

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()


# ============================================================
# 20. HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🧠 IntelliMind AI'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Research & Knowledge Assistant'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# 21. TOP FEATURES
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "📄 Documents",
        len(
            st.session_state.documents
        )
    )

with col2:

    st.metric(
        "🧩 Chunks",
        len(
            st.session_state.chunks
        )
    )

with col3:

    st.metric(
        "🤖 AI",
        "Gemini"
        if client
        else "Offline"
    )

with col4:

    st.metric(
        "🔎 Retrieval",
        "TF-IDF"
    )


# ============================================================
# 22. DOCUMENT SECTION
# ============================================================

if st.session_state.documents:

    st.markdown(
        "### 📚 Uploaded Documents"
    )

    for name in st.session_state.documents:

        st.write(
            f"📄 **{name}**"
        )


# ============================================================
# 23. AI RESEARCH TOOLS
# ============================================================

if st.session_state.chunks:

    st.markdown(
        "### 🛠️ AI Research Tools"
    )

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    with col1:

        if st.button(
            "📝 Generate Document Summary",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing document..."
            ):

                summary = generate_summary()

            st.session_state.document_summary = (
                summary
            )

    # --------------------------------------------------------
    # Questions
    # --------------------------------------------------------

    with col2:

        if st.button(
            "💡 Generate Questions",
            use_container_width=True
        ):

            with st.spinner(
                "Generating questions..."
            ):

                questions = generate_questions()

            st.session_state.suggested_questions = (
                questions
            )


# ============================================================
# 24. SUMMARY
# ============================================================

if st.session_state.document_summary:

    with st.expander(
        "📝 Document Summary",
        expanded=True
    ):

        st.markdown(
            st.session_state.document_summary
        )


# ============================================================
# 25. SUGGESTED QUESTIONS
# ============================================================

if st.session_state.suggested_questions:

    st.markdown(
        "### 💡 Suggested Questions"
    )

    for question in (
        st.session_state.suggested_questions
    ):

        st.info(
            question
        )


# ============================================================
# 26. CHAT HISTORY
# ============================================================

for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if message["role"] == "assistant":

            if message.get("source"):

                st.caption(
                    f"Source: "
                    f"{message['source']}"
                )

            if message.get("score") is not None:

                st.caption(
                    f"Retrieval score: "
                    f"{message['score']:.2f}"
                )


# ============================================================
# 27. CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask anything about your documents..."
)


# ============================================================
# 28. HANDLE QUESTION
# ============================================================

if question:

    # ========================================================
    # USER MESSAGE
    # ========================================================

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(
            question
        )

    # ========================================================
    # AI MESSAGE
    # ========================================================

    with st.chat_message("assistant"):

        with st.spinner(
            "🧠 Searching knowledge base and thinking..."
        ):

            (
                answer,
                source,
                score,
                results
            ) = generate_answer(
                question,
                mode
            )

        # ----------------------------------------------------
        # Answer
        # ----------------------------------------------------

        st.markdown(
            answer
        )

        # ----------------------------------------------------
        # Source
        # ----------------------------------------------------

        st.caption(
            f"Source: {source}"
        )

        # ----------------------------------------------------
        # Retrieval score
        # ----------------------------------------------------

        st.caption(
            f"Best retrieval score: "
            f"{score:.2f}"
        )

        # ====================================================
        # RETRIEVED SOURCES
        # ====================================================

        relevant = [
            item
            for item in results
            if item["score"] >= MIN_SIMILARITY
        ]

        if relevant:

            with st.expander(
                "🔎 View Retrieved Sources"
            ):

                for item in relevant:

                    st.markdown(
                        f"""
### 📄 {item['source']}

**Chunk:** `{item['chunk_id']}`

**Relevance:** `{item['score']:.2f}`

{item['text']}

---
"""
                    )

    # ========================================================
    # SAVE CHAT
    # ========================================================

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "source": source,
            "score": score
        }
    )
