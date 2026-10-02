import streamlit as st
import re
import os
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from google import genai


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="IntelliMind AI",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
}

.answer-box {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    margin-top: 15px;
}

.info-box {
    padding: 15px;
    border-radius: 10px;
    background-color: rgba(100,100,100,0.08);
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">🤖 IntelliMind AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Knowledge Base + Google Gemini AI Assistant</div>',
    unsafe_allow_html=True
)

st.write("")


# =========================================================
# GEMINI CLIENT
# =========================================================

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
        return genai.Client(api_key=api_key)
    except Exception:
        return None


client = get_gemini_client()


# =========================================================
# SESSION STATE
# =========================================================

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = text.replace("\x00", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# SPLIT TEXT INTO CHUNKS
# =========================================================

def split_text(text, chunk_size=700, overlap=100):

    text = clean_text(text)

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


# =========================================================
# READ TXT FILE
# =========================================================

def read_txt(uploaded_file):

    try:

        text = uploaded_file.read().decode(
            "utf-8",
            errors="ignore"
        )

        return clean_text(text)

    except Exception as e:

        st.error(f"TXT reading error: {e}")

        return ""


# =========================================================
# READ PDF FILE
# =========================================================

def read_pdf(uploaded_file):

    try:

        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return clean_text(text)

    except Exception as e:

        st.error(f"PDF reading error: {e}")

        return ""


# =========================================================
# PROCESS UPLOADED FILE
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

    chunks = split_text(text)

    return chunks


# =========================================================
# DATASET SEARCH
# =========================================================

def search_knowledge_base(question, top_k=4):

    chunks = st.session_state.chunks

    if not chunks:

        return [], 0.0

    try:

        documents = chunks + [question]

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        matrix = vectorizer.fit_transform(documents)

        question_vector = matrix[-1]

        document_vectors = matrix[:-1]

        similarities = cosine_similarity(
            question_vector,
            document_vectors
        )[0]

        ranked_indices = similarities.argsort()[::-1]

        results = []

        for index in ranked_indices[:top_k]:

            score = float(similarities[index])

            results.append(
                {
                    "text": chunks[index],
                    "score": score
                }
            )

        if results:

            best_score = results[0]["score"]

        else:

            best_score = 0.0

        return results, best_score

    except Exception as e:

        return [], 0.0


# =========================================================
# GEMINI RESPONSE
# =========================================================

def ask_gemini(question, context=""):

    if client is None:

        return None

    if context:

        prompt = f"""
You are IntelliMind AI, a helpful AI assistant.

The user asked:

{question}

Below is information retrieved from the user's uploaded
knowledge base.

KNOWLEDGE BASE:
{context}

Instructions:

1. Use the knowledge base as the primary source.
2. If the answer is clearly available in the knowledge base,
   answer using that information.
3. Do not invent facts that are not supported by the knowledge base.
4. If the knowledge base does not contain enough information,
   clearly say that the uploaded knowledge base does not provide
   enough information.
5. Keep the answer clear and easy to understand.
6. If appropriate, use bullet points.

Answer the user:
"""

    else:

        prompt = f"""
You are IntelliMind AI, a helpful general-purpose AI assistant.

The user's question is:

{question}

There is no relevant information available in the user's
uploaded knowledge base.

Answer the question using your general knowledge.

Instructions:

1. Give a useful and accurate answer.
2. Do not pretend that the information came from the uploaded files.
3. If the question requires current or uncertain information,
   clearly mention the limitation.
4. Keep the answer simple and well structured.

Answer:
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if response and response.text:

            return response.text.strip()

        return None

    except Exception as e:

        error_text = str(e)

        if "503" in error_text:

            return (
                "⚠️ Gemini AI is temporarily unavailable because "
                "the model is experiencing high demand. "
                "Please try again in a few moments."
            )

        if "429" in error_text:

            return (
                "⚠️ Gemini API request limit reached. "
                "Please wait and try again."
            )

        if "401" in error_text or "403" in error_text:

            return (
                "⚠️ Gemini API authentication failed. "
                "Please check your API key."
            )

        return (
            "⚠️ Gemini could not generate the answer right now.\n\n"
            f"Technical information: {error_text}"
        )


# =========================================================
# GENERATE FINAL ANSWER
# =========================================================

def generate_answer(question):

    results, best_score = search_knowledge_base(
        question,
        top_k=4
    )

    # -----------------------------------------------------
    # CASE 1: Relevant knowledge found
    # -----------------------------------------------------

    if results and best_score >= 0.15:

        selected_results = [
            item
            for item in results
            if item["score"] >= 0.15
        ]

        context = "\n\n".join(
            item["text"]
            for item in selected_results
        )

        answer = ask_gemini(
            question,
            context
        )

        if answer:

            return answer, "📚 Knowledge Base + Gemini", best_score

        # Gemini unavailable → show dataset information

        return (
            context,
            "📚 Knowledge Base",
            best_score
        )

    # -----------------------------------------------------
    # CASE 2: No relevant knowledge
    # -----------------------------------------------------

    answer = ask_gemini(
        question,
        ""
    )

    if answer:

        return answer, "🌐 Gemini General AI", best_score

    # -----------------------------------------------------
    # CASE 3: Gemini unavailable
    # -----------------------------------------------------

    return (
        "I couldn't find relevant information in the uploaded "
        "knowledge base, and the AI service is currently unavailable.",
        "⚠️ Fallback",
        best_score
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📂 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload TXT or PDF files",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        if st.button("🔄 Process Files", use_container_width=True):

            all_chunks = []
            file_names = []

            for file in uploaded_files:

                chunks = process_file(file)

                if chunks:

                    all_chunks.extend(chunks)

                    file_names.append(file.name)

            st.session_state.chunks = all_chunks
            st.session_state.documents = file_names

            st.success(
                f"{len(file_names)} file(s) processed successfully!"
            )

    st.divider()

    st.subheader("📊 Knowledge Base Status")

    st.write(
        f"Files: **{len(st.session_state.documents)}**"
    )

    st.write(
        f"Text chunks: **{len(st.session_state.chunks)}**"
    )

    if client:

        st.success("🟢 Gemini API Connected")

    else:

        st.warning("🔴 Gemini API Not Connected")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.chat_history = []

        st.rerun()


# =========================================================
# SHOW UPLOADED FILES
# =========================================================

if st.session_state.documents:

    st.markdown("### 📚 Uploaded Knowledge")

    cols = st.columns(
        min(len(st.session_state.documents), 4)
    )

    for i, name in enumerate(
        st.session_state.documents
    ):

        with cols[i % len(cols)]:

            st.info(f"📄 {name}")


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            if "source" in message:

                st.caption(
                    f"Source: {message['source']}"
                )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask me anything..."
)


# =========================================================
# HANDLE QUESTION
# =========================================================

if question:

    # User message

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # Assistant

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer, source, score = generate_answer(
                question
            )

        st.markdown(answer)

        st.caption(
            f"Source: {source} | Similarity: {score:.2f}"
        )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "source": source
        }
    )
