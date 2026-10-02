import os
import re
import io

import streamlit as st
from pypdf import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from google import genai


# ============================================================
# 1. CONFIGURATION
# ============================================================

APP_NAME = "IntelliMind AI"

MODEL_NAME = "gemini-3.8-flash"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

TOP_K = 5
MIN_SIMILARITY = 0.18


# ============================================================
# 2. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="IntelliMind AI",
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

    .feature-card {
        padding: 18px;
        border-radius: 15px;
        border: 1px solid rgba(120,120,120,0.2);
        margin-bottom: 10px;
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

    # Streamlit Cloud Secrets
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    # Local environment variable
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


client = get_gemini_client()


# ============================================================
# 6. TEXT CLEANING
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = text.replace("\x00", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 7. SMART TEXT CHUNKING
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

        # Try to end at a sentence
        if end < len(text):

            possible_breaks = [
                chunk.rfind("."),
                chunk.rfind("?"),
                chunk.rfind("!"),
                chunk.rfind("\n")
            ]

            best_break = max(
                possible_breaks
            )

            if best_break > CHUNK_SIZE * 0.55:

                chunk = chunk[
                    :best_break + 1
                ]

                end = start + len(chunk)

        if len(chunk.strip()) > 40:

            chunks.append(
                chunk.strip()
            )

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

            page_text = page.extract_text()

            if page_text:

                pages.append(
                    f"[Page {page_number}]\n"
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

        ranked_indices = scores.argsort()[
            ::-1
        ]

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

    context = []

    for item in useful_results:

        context.append(
            f"""
SOURCE: {item['source']}
CHUNK: {item['chunk_id']}
RELEVANCE SCORE: {item['score']:.2f}

{item['text']}
"""
        )

    return "\n------------------------\n".join(
        context
    )


# ============================================================
# 13. ASK GEMINI
# ============================================================

def ask_gemini(
    question,
    context="",
    mode="Normal"
):

    if client is None:

        return (
            None,
            "Gemini API is not configured."
        )

    if context:

        prompt = f"""
You are IntelliMind AI, an AI-powered
Research and Knowledge Assistant.

USER QUESTION:
{question}

USER UPLOADED KNOWLEDGE:
{context}

MODE:
{mode}

IMPORTANT RULES:

1. The uploaded knowledge is the primary source.
2. Answer using the retrieved information whenever possible.
3. Do not invent information from the documents.
4. If the documents only partially answer the question,
   clearly explain what is known and what is missing.
5. Do not claim information came from the document
   unless it is actually supported by the context.
6. Give a clear and well-structured answer.
7. Use headings or bullet points when useful.

If MODE is Student:

Explain the concept in simple English.
Give a simple example.
Mention important exam points when appropriate.

If MODE is Research:

Focus on:
- objective
- methodology
- findings
- limitations
- research implications

Now answer the user's question.
"""

        source = "📚 Knowledge Base + Gemini"

    else:

        prompt = f"""
You are IntelliMind AI,
a general-purpose AI assistant.

USER QUESTION:
{question}

No sufficiently relevant information
was found in the uploaded knowledge base.

MODE:
{mode}

Answer using your general knowledge.

IMPORTANT:

1. Do not pretend the answer came from uploaded documents.
2. Give a useful and clear answer.
3. If information is uncertain or time-sensitive,
   mention the limitation.
4. Use simple language when appropriate.

Answer:
"""

        source = "🌐 Gemini General AI"

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
                "Model not available. Check MODEL_NAME."
            )

        if "429" in error:

            return (
                None,
                "API quota/rate limit reached."
            )

        if "401" in error or "403" in error:

            return (
                None,
                "API key authentication failed."
            )

        if "503" in error:

            return (
                None,
                "Gemini is temporarily unavailable."
            )

        return (
            None,
            f"Gemini error: {error}"
        )


# ============================================================
# 14. MAIN ANSWER ENGINE
# ============================================================

def generate_answer(
    question,
    mode
):

    results, best_score = retrieve_knowledge(
        question
    )

    context = build_context(
        results
    )

    answer, source = ask_gemini(
        question,
        context,
        mode
    )

    if answer:

        return (
            answer,
            source,
            best_score,
            results
        )

    # Knowledge base fallback
    if context:

        fallback = """
⚠️ Gemini could not generate the final
response right now.

However, I found relevant information
in your uploaded knowledge base:

""" + context

        return (
            fallback,
            "📚 Knowledge Base Fallback",
            best_score,
            results
        )

    return (
        "I could not generate an answer right now. "
        "Please check your Gemini API configuration.",
        "⚠️ System Fallback",
        best_score,
        results
    )


# ============================================================
# 15. DOCUMENT SUMMARY
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

DOCUMENT:

{context}

Create a structured summary with:

1. Title/Topic
2. Main Objective
3. Key Concepts
4. Methodology or Approach
5. Important Findings
6. Limitations
7. Possible Future Work
8. Short Conclusion

Do not invent information.
If something is unavailable, say "Not mentioned".
"""

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        if response and response.text:

            return response.text.strip()

        return "Could not generate summary."

    except Exception as e:

        return f"Summary error: {e}"


# ============================================================
# 16. SUGGEST QUESTIONS
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
Based on this document:

{context}

Generate 5 useful questions a student
or researcher could ask about this document.

Return only the questions.
Number them 1 to 5.
"""

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        if not response or not response.text:

            return []

        lines = response.text.split("\n")

        questions = []

        for line in lines:

            line = re.sub(
                r"^\s*\d+[\.\)]\s*",
                "",
                line
            ).strip()

            if len(line) > 10:

                questions.append(
                    line
                )

        return questions[:5]

    except Exception:

        return []


# ============================================================
# 17. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🧠 IntelliMind AI")

    st.subheader("📂 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
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
                    (i + 1) /
                    len(uploaded_files)
                )

            st.session_state.chunks = (
                all_chunks
            )

            st.session_state.documents = (
                names
            )

            st.session_state.document_summary = ""

            st.session_state.suggested_questions = []

            st.success(
                "Knowledge Base created!"
            )

    st.divider()

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

    st.subheader("📊 System Status")

    st.write(
        f"📄 Documents: "
        f"**{len(st.session_state.documents)}**"
    )

    st.write(
        f"🧩 Text Chunks: "
        f"**{len(st.session_state.chunks)}**"
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
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()


# ============================================================
# 18. HEADER
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
# 19. TOP FEATURES
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
# 20. DOCUMENT SECTION
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
# 21. AI TOOLS
# ============================================================

if st.session_state.chunks:

    st.markdown(
        "### 🛠️ AI Research Tools"
    )

    col1, col2 = st.columns(2)

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
# 22. SUMMARY
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
# 23. SUGGESTED QUESTIONS
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
# 24. CHAT HISTORY
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
# 25. CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask anything about your documents..."
)


# ============================================================
# 26. HANDLE QUESTION
# ============================================================

if question:

    # USER
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

    # AI
    with st.chat_message("assistant"):

        with st.spinner(
            "🧠 Thinking and searching..."
        ):

            answer, source, score, results = (
                generate_answer(
                    question,
                    mode
                )
            )

        st.markdown(
            answer
        )

        st.caption(
            f"Source: {source}"
        )

        st.caption(
            f"Best retrieval score: "
            f"{score:.2f}"
        )

        # Sources
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
**📄 {item['source']}**

Chunk: `{item['chunk_id']}`

Relevance: `{item['score']:.2f}`

{item['text']}
"""
                    )

    # SAVE
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "source": source,
            "score": score
        }
    )
