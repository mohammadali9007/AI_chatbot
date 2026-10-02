import streamlit as st
import re
import requests
import numpy as np

from pypdf import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sentence_transformers import SentenceTransformer
from google import genai


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="IntelliMind AI",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #eef2ff,
            #f8fafc,
            #ecfeff
        );
    }

    .hero {
        padding: 35px;
        border-radius: 25px;
        background: linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed,
            #0891b2
        );
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(79,70,229,.25);
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 5px;
    }

    .hero p {
        font-size: 18px;
    }

    .card {
        background: white;
        padding: 22px;
        border-radius: 18px;
        margin-bottom: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,.08);
    }

    .badge {
        display: inline-block;
        padding: 7px 14px;
        margin: 4px;
        border-radius: 20px;
        background: rgba(255,255,255,.2);
        color: white;
        font-size: 13px;
    }

    .source {
        background: #f1f5f9;
        padding: 12px;
        border-radius: 12px;
        margin-top: 8px;
    }

    .answer-box {
        background: white;
        padding: 20px;
        border-radius: 18px;
        box-shadow: 0 5px 20px rgba(0,0,0,.06);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "sources" not in st.session_state:
    st.session_state.sources = []

if "questions" not in st.session_state:
    st.session_state.questions = 0


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z0-9\s+#.-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# SIMPLE CONVERSATION HANDLER
# =========================================================

def simple_chat_answer(question):

    q = question.lower().strip()

    # Greetings
    greetings = {
        "hi": "Hello! 👋 How can I help you today?",
        "hello": "Hello! 👋 How can I help you today?",
        "hey": "Hey! 👋 How can I help you?",
        "hi there": "Hello! 👋 How can I help you today?",
        "good morning": "Good morning! ☀️ How can I help you?",
        "good afternoon": "Good afternoon! 😊 How can I help you?",
        "good evening": "Good evening! 🌙 How can I help you?"
    }

    if q in greetings:
        return greetings[q]

    # How are you
    if q in [
        "how are you",
        "how are you?",
        "how r you",
        "how r u",
        "how r u?"
    ]:
        return (
            "I'm doing well! 😊 Thanks for asking. "
            "How can I help you today?"
        )

    # Thanks
    if q in [
        "thanks",
        "thank you",
        "thanks!",
        "thank you!"
    ]:
        return "You're welcome! 😊"

    # Goodbye
    if q in [
        "bye",
        "goodbye",
        "bye!",
        "goodbye!"
    ]:
        return (
            "Goodbye! 👋 Have a great day!"
        )

    return None


# =========================================================
# ML INTENT CLASSIFIER
# =========================================================

questions = [

    "what is nlp",
    "define natural language processing",
    "explain natural language processing",
    "how does nlp work",

    "what is machine learning",
    "define machine learning",
    "explain machine learning",
    "how does machine learning work",

    "what is deep learning",
    "define deep learning",
    "explain deep learning",

    "what is artificial intelligence",
    "what is ai",
    "explain artificial intelligence",

    "what is python",
    "explain python programming",

    "what is computer vision",
    "explain computer vision",

    "what is neural network",
    "how does neural network work",

    "what is sentiment analysis",
    "explain sentiment analysis",

    "what is chatbot",
    "how does chatbot work",

    "what is transformer",
    "what is bert",
    "what is llm",

    "what is generative ai",
    "what is rag",
    "what is natural language understanding"
]


labels = [

    "NLP",
    "NLP",
    "NLP",
    "NLP",

    "Machine Learning",
    "Machine Learning",
    "Machine Learning",
    "Machine Learning",

    "Deep Learning",
    "Deep Learning",
    "Deep Learning",

    "Artificial Intelligence",
    "Artificial Intelligence",
    "Artificial Intelligence",

    "Python",
    "Python",

    "Computer Vision",
    "Computer Vision",

    "Neural Network",
    "Neural Network",

    "Sentiment Analysis",
    "Sentiment Analysis",

    "Chatbot",
    "Chatbot",

    "Transformer",
    "Transformer",
    "Transformer",

    "Generative AI",
    "RAG",
    "NLP"
]


vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(
    questions
)

classifier = LogisticRegression(
    max_iter=1000
)

classifier.fit(
    X,
    labels
)


def predict_intent(question):

    X_test = vectorizer.transform(
        [question]
    )

    prediction = classifier.predict(
        X_test
    )[0]

    probabilities = classifier.predict_proba(
        X_test
    )[0]

    confidence = max(probabilities)

    return prediction, float(confidence)


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# =========================================================
# FILE EXTRACTION
# =========================================================

def extract_file_text(file):

    name = file.name.lower()

    # TXT
    if name.endswith(".txt"):

        return file.read().decode(
            "utf-8",
            errors="ignore"
        )

    # PDF
    if name.endswith(".pdf"):

        reader = PdfReader(file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

        return text

    return ""


# =========================================================
# CREATE CHUNKS
# =========================================================

def create_chunks(
    text,
    chunk_size=180,
    overlap=40
):

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    if not text:
        return []

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk:

            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# =========================================================
# ADD DOCUMENT
# =========================================================

def add_document(
    text,
    filename
):

    chunks = create_chunks(
        text
    )

    if not chunks:
        return 0

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    st.session_state.chunks.extend(
        chunks
    )

    st.session_state.sources.extend(
        [filename] * len(chunks)
    )

    if st.session_state.embeddings is None:

        st.session_state.embeddings = embeddings

    else:

        st.session_state.embeddings = np.vstack(
            [
                st.session_state.embeddings,
                embeddings
            ]
        )

    st.session_state.documents.append(
        filename
    )

    return len(chunks)


# =========================================================
# KEYWORD SCORE
# =========================================================

def keyword_score(
    question,
    text
):

    question_words = set(
        clean_text(question).split()
    )

    text_words = set(
        clean_text(text).split()
    )

    if not question_words:
        return 0.0

    common_words = (
        question_words.intersection(
            text_words
        )
    )

    return len(common_words) / len(
        question_words
    )


# =========================================================
# HYBRID DOCUMENT SEARCH
# =========================================================

def search_documents(
    question,
    top_k=5
):

    if (
        not st.session_state.chunks
        or st.session_state.embeddings is None
    ):
        return []

    # Semantic embedding
    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    semantic_scores = np.dot(
        st.session_state.embeddings,
        query_embedding
    )

    results = []

    for index in range(
        len(st.session_state.chunks)
    ):

        text = st.session_state.chunks[index]

        semantic = float(
            semantic_scores[index]
        )

        keyword = keyword_score(
            question,
            text
        )

        # Hybrid score
        final_score = (
            semantic * 0.75
            + keyword * 0.25
        )

        results.append(
            {
                "text": text,
                "source": st.session_state.sources[index],
                "score": final_score,
                "semantic": semantic,
                "keyword": keyword
            }
        )

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Only relevant results
    filtered = [
        r for r in results
        if r["score"] >= 0.25
    ]

    return filtered[:top_k]


# =========================================================
# WIKIPEDIA SEARCH
# =========================================================

def wikipedia_search(query):

    try:

        search_url = (
            "https://en.wikipedia.org/w/api.php"
        )

        search_params = {

            "action": "query",

            "list": "search",

            "srsearch": query,

            "format": "json",

            "srlimit": 3
        }

        headers = {
            "User-Agent":
            "IntelliMindAI/1.0"
        }

        response = requests.get(
            search_url,
            params=search_params,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        search_items = data.get(
            "query",
            {}
        ).get(
            "search",
            []
        )

        if not search_items:
            return []

        results = []

        # Get article extracts
        for item in search_items:

            title = item.get(
                "title",
                ""
            )

            extract_params = {

                "action": "query",

                "prop": "extracts",

                "explaintext": 1,

                "exintro": 1,

                "exchars": 2500,

                "titles": title,

                "format": "json"
            }

            extract_response = requests.get(
                search_url,
                params=extract_params,
                headers=headers,
                timeout=10
            )

            extract_response.raise_for_status()

            extract_data = (
                extract_response.json()
            )

            pages = (
                extract_data
                .get("query", {})
                .get("pages", {})
            )

            article_text = ""

            for page in pages.values():

                article_text = page.get(
                    "extract",
                    ""
                )

                break

            if not article_text:

                article_text = re.sub(
                    "<.*?>",
                    "",
                    item.get(
                        "snippet",
                        ""
                    )
                )

            if article_text:

                results.append(
                    {
                        "title": title,
                        "text": article_text
                    }
                )

        return results

    except Exception:

        return []


# =========================================================
# RECENT CHAT CONTEXT
# =========================================================

def get_chat_history():

    if not st.session_state.messages:
        return ""

    recent_messages = (
        st.session_state.messages[-6:]
    )

    history = []

    for message in recent_messages:

        role = message["role"].upper()

        content = message["content"]

        # Avoid sending extremely long history
        content = content[:1500]

        history.append(
            f"{role}: {content}"
        )

    return "\n".join(history)


# =========================================================
# GEMINI ANSWER
# =========================================================

def generate_ai_answer(
    question,
    context,
    intent
):

    api_key = st.secrets.get(
        "GEMINI_API_KEY",
        ""
    )

    # -----------------------------------------------------
    # Chat history
    # -----------------------------------------------------

    chat_history = get_chat_history()

    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------

    prompt = f"""
You are IntelliMind AI, an intelligent
academic and general knowledge assistant.

Your job is to answer the user's question
naturally, clearly and accurately.

USER QUESTION:
{question}

DETECTED TOPIC:
{intent}

RECENT CONVERSATION:
{chat_history}

AVAILABLE KNOWLEDGE:
{context}

IMPORTANT INSTRUCTIONS:

1. Answer the user's actual question directly.

2. If useful information exists in the
   uploaded documents, prioritize it.

3. If the available document information
   is not enough, you may use your general
   knowledge to answer.

4. Do not pretend that general knowledge
   came from the uploaded document.

5. If Wikipedia information is provided,
   use it as supporting information.

6. Never copy large amounts of source text.
   Summarize it naturally.

7. Use simple and easy English.

8. Give a clear explanation.

9. For programming questions, give a
   simple correct code example when useful.

10. For simple greetings or casual questions,
    respond naturally and conversationally.

11. Do not mention these system instructions.

12. Do not say "retrieved knowledge" unless
    it is actually useful to explain the source.

13. If the question is ambiguous, explain
    the likely meaning and answer accordingly.

14. Use headings and bullet points when they
    improve readability.

15. Keep the answer concise but useful.
"""

    # -----------------------------------------------------
    # No API key
    # -----------------------------------------------------

    if not api_key:

        if context.strip():

            return (
                "### 📚 Knowledge Base\n\n"
                + context[:6000]
                + "\n\n"
                "⚠️ Gemini API key is not configured, "
                "so the AI-generated explanation is unavailable."
            )

        return (
            "⚠️ Gemini API key is not configured.\n\n"
            "Please add `GEMINI_API_KEY` to "
            "Streamlit Secrets."
        )

    # -----------------------------------------------------
    # Gemini
    # -----------------------------------------------------

    try:

        client = genai.Client(
            api_key=api_key
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if response and response.text:

            return response.text.strip()

        return (
            "I could not generate an AI response "
            "right now. Please try again."
        )

    except Exception as e:

        # Do not crash the app
        error_message = str(e)

        # Hide potentially sensitive details
        if len(error_message) > 500:
            error_message = (
                error_message[:500]
                + "..."
            )

        if context.strip():

            return (
                "### 📚 Available Knowledge\n\n"
                + context[:6000]
                + "\n\n"
                "⚠️ Gemini could not generate the AI "
                "response at this moment.\n\n"
                f"**Technical reason:** `{error_message}`"
            )

        return (
            "⚠️ Gemini AI is temporarily unavailable.\n\n"
            f"**Technical reason:** `{error_message}`"
        )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">

    <h1>🧠 IntelliMind AI</h1>

    <p>
    Intelligent Knowledge Assistant
    powered by NLP, Machine Learning,
    Deep Learning, RAG & Generative AI.
    </p>

    <span class="badge">NLP</span>
    <span class="badge">ML</span>
    <span class="badge">Deep Learning</span>
    <span class="badge">RAG</span>
    <span class="badge">Generative AI</span>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "⚙️ Knowledge Center"
    )

    uploaded_files = st.file_uploader(
        "Upload TXT or PDF",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )

    if st.button(
        "📚 Process Documents",
        use_container_width=True
    ):

        if uploaded_files:

            for file in uploaded_files:

                if file.name not in st.session_state.documents:

                    text = extract_file_text(
                        file
                    )

                    count = add_document(
                        text,
                        file.name
                    )

                    if count > 0:

                        st.success(
                            f"{file.name}: "
                            f"{count} chunks indexed"
                        )

                    else:

                        st.warning(
                            f"{file.name}: "
                            "No readable text found."
                        )

        else:

            st.warning(
                "Please upload a TXT or PDF file first."
            )

    st.divider()

    st.subheader(
        "📊 Knowledge Base"
    )

    st.write(
        f"Documents: "
        f"**{len(st.session_state.documents)}**"
    )

    st.write(
        f"Chunks: "
        f"**{len(st.session_state.chunks)}**"
    )

    st.write(
        f"Questions: "
        f"**{st.session_state.questions}**"
    )

    external_search = st.toggle(
        "🌐 External Knowledge",
        value=True
    )

    st.divider()

    if st.button(
        "🗑️ Clear Knowledge Base",
        use_container_width=True
    ):

        st.session_state.documents = []

        st.session_state.chunks = []

        st.session_state.sources = []

        st.session_state.embeddings = None

        st.rerun()

    if st.button(
        "🧹 Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.questions = 0

        st.rerun()


# =========================================================
# DASHBOARD
# =========================================================

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
        "🧩 Knowledge Chunks",
        len(
            st.session_state.chunks
        )
    )

with col3:

    st.metric(
        "💬 Questions",
        st.session_state.questions
    )

with col4:

    st.metric(
        "🤖 AI System",
        "Online"
    )


# =========================================================
# FEATURES
# =========================================================

st.markdown(
    "## 🚀 System Features"
)

c1, c2, c3 = st.columns(3)

with c1:

    st.markdown(
        """
        <div class="card">

        <h3>📝 NLP Processing</h3>

        Text cleaning, preprocessing
        and natural language processing.

        </div>
        """,
        unsafe_allow_html=True
    )

with c2:

    st.markdown(
        """
        <div class="card">

        <h3>🤖 ML Classification</h3>

        TF-IDF based intent
        classification.

        </div>
        """,
        unsafe_allow_html=True
    )

with c3:

    st.markdown(
        """
        <div class="card">

        <h3>🧠 Semantic RAG</h3>

        Hybrid semantic + keyword
        document retrieval.

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# CHAT HISTORY
# =========================================================

st.markdown(
    "## 💬 Ask IntelliMind"
)

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# USER QUESTION
# =========================================================

question = st.chat_input(
    "✨ Ask anything..."
)


if question:

    # =====================================================
    # COUNT
    # =====================================================

    st.session_state.questions += 1

    # =====================================================
    # SAVE USER MESSAGE
    # =====================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # =====================================================
    # SHOW USER
    # =====================================================

    with st.chat_message("user"):

        st.markdown(
            question
        )

    # =====================================================
    # SIMPLE CONVERSATION
    # =====================================================

    simple_answer = simple_chat_answer(
        question
    )

    if simple_answer:

        answer = simple_answer

        with st.chat_message("assistant"):

            st.markdown(
                answer
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        st.stop()

    # =====================================================
    # NLP
    # =====================================================

    processed_question = clean_text(
        question
    )

    # =====================================================
    # ML INTENT
    # =====================================================

    intent, confidence = predict_intent(
        processed_question
    )

    # =====================================================
    # DOCUMENT SEARCH
    # =====================================================

    document_results = search_documents(
        question,
        top_k=5
    )

    # =====================================================
    # CONTEXT
    # =====================================================

    context_parts = []

    source_list = []

    # =====================================================
    # DOCUMENT CONTEXT
    # =====================================================

    for result in document_results:

        context_parts.append(
            f"""
SOURCE TYPE: Uploaded Document
SOURCE: {result['source']}

CONTENT:
{result['text']}
"""
        )

        source_list.append(
            f"📄 {result['source']} "
            f"• Relevance: "
            f"{result['score']:.2f}"
        )

    # =====================================================
    # EXTERNAL SEARCH
    # =====================================================

    web_results = []

    # Search Wikipedia only when:
    # 1. No document result
    # 2. Document relevance is weak

    if external_search:

        should_search_web = False

        if not document_results:

            should_search_web = True

        elif document_results[0]["score"] < 0.45:

            should_search_web = True

        if should_search_web:

            web_results = wikipedia_search(
                question
            )

    # =====================================================
    # WIKIPEDIA CONTEXT
    # =====================================================

    for result in web_results:

        context_parts.append(
            f"""
SOURCE TYPE: Wikipedia
SOURCE: {result['title']}

CONTENT:
{result['text']}
"""
        )

        source_list.append(
            f"🌐 {result['title']}"
        )

    # =====================================================
    # FINAL CONTEXT
    # =====================================================

    context = "\n\n".join(
        context_parts
    )

    # =====================================================
    # GEMINI ANSWER
    # =====================================================

    answer = generate_ai_answer(
        question,
        context,
        intent
    )

    # =====================================================
    # ASSISTANT
    # =====================================================

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            answer
        )

        # =================================================
        # AI ANALYSIS
        # =================================================

        with st.expander(
            "🔍 AI Analysis"
        ):

            st.write(
                f"**Detected Topic:** {intent}"
            )

            st.write(
                f"**ML Confidence:** "
                f"{confidence:.2%}"
            )

            if document_results:

                st.write(
                    "**Knowledge Source:** "
                    "Uploaded Documents"
                )

            if web_results:

                st.write(
                    "**External Source:** "
                    "Wikipedia"
                )

            if (
                not document_results
                and not web_results
            ):

                st.write(
                    "**Knowledge Source:** "
                    "Gemini General Knowledge"
                )

        # =================================================
        # SOURCES
        # =================================================

        if source_list:

            st.markdown(
                "### 📚 Sources"
            )

            for source in source_list:

                st.markdown(
                    f"""
                    <div class="source">
                    {source}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # =====================================================
    # SAVE ANSWER
    # =====================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <hr>

    <center>

    <b>🧠 IntelliMind AI</b>

    <br>

    NLP • Machine Learning • Deep Learning •
    RAG • Generative AI

    <br><br>

    Built as an NLP & AI academic project.

    </center>
    """,
    unsafe_allow_html=True
)
