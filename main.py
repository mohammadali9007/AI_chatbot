import streamlit as st
import re
import requests
import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from google import genai

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="IntelliMind AI",
    page_icon="🧠",
    layout="wide"
)

# =========================
# CSS
# =========================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #eef2ff, #f8fafc, #ecfeff);
}

.hero {
    padding: 35px;
    border-radius: 25px;
    background: linear-gradient(135deg, #4f46e5, #7c3aed, #0891b2);
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

</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================

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


# =========================
# NLP PROCESSOR
# =========================

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


# =========================
# ML INTENT CLASSIFIER
# =========================

questions = [
    "what is nlp",
    "define natural language processing",
    "explain natural language processing",

    "what is machine learning",
    "define machine learning",
    "explain machine learning",

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
    "what is llm"
]

labels = [
    "NLP",
    "NLP",
    "NLP",

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
    "Transformer"
]

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(questions)

classifier = LogisticRegression(
    max_iter=1000
)

classifier.fit(X, labels)


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


# =========================
# EMBEDDING MODEL
# =========================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# =========================
# DOCUMENT PROCESSING
# =========================

def extract_file_text(file):

    name = file.name.lower()

    if name.endswith(".txt"):

        return file.read().decode(
            "utf-8",
            errors="ignore"
        )

    if name.endswith(".pdf"):

        reader = PdfReader(file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text

    return ""


def create_chunks(
    text,
    chunk_size=250,
    overlap=50
):

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

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


# =========================
# ADD DOCUMENT
# =========================

def add_document(
    text,
    filename
):

    chunks = create_chunks(text)

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


# =========================
# SEARCH DOCUMENTS
# =========================

def search_documents(
    question,
    top_k=5
):

    if not st.session_state.chunks:
        return []

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    scores = np.dot(
        st.session_state.embeddings,
        query_embedding
    )

    indexes = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for index in indexes:

        score = float(
            scores[index]
        )

        if score >= 0.30:

            results.append({
                "text":
                    st.session_state.chunks[index],

                "source":
                    st.session_state.sources[index],

                "score":
                    score
            })

    return results


# =========================
# WIKIPEDIA SEARCH
# =========================

def wikipedia_search(query):

    try:

        url = "https://en.wikipedia.org/w/api.php"

        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 3
        }

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        data = response.json()

        results = []

        for item in data.get(
            "query",
            {}
        ).get(
            "search",
            []
        ):

            title = item.get(
                "title",
                ""
            )

            snippet = item.get(
                "snippet",
                ""
            )

            snippet = re.sub(
                "<.*?>",
                "",
                snippet
            )

            results.append({
                "title": title,
                "text": snippet
            })

        return results

    except:

        return []


# =========================
# GEMINI AI
# =========================

def generate_ai_answer(
    question,
    context,
    intent
):

    api_key = st.secrets.get(
        "GEMINI_API_KEY",
        ""
    )

    if not api_key:

        return (
            "⚠️ Gemini API key is not configured. "
            "Please add GEMINI_API_KEY in "
            "Streamlit Secrets."
        )

    client = genai.Client(
        api_key=api_key
    )

    prompt = f"""

You are IntelliMind AI,
an intelligent NLP-based knowledge assistant.

User Question:
{question}

Detected Topic:
{intent}

Available Knowledge:
{context}

Instructions:

1. Answer the question clearly.
2. Use the provided knowledge.
3. Do not invent information.
4. Use simple English.
5. Give examples when useful.
6. Use bullet points when appropriate.
7. If the knowledge is insufficient,
   clearly say that.
8. Keep the answer understandable
   for a university student.

"""

    try:

        response = client.models.generate_content(

            model="gemini-3.8-flash",

            contents=prompt

        )

        return response.text

    except Exception as e:

        return (
            "AI generation error:\n\n"
            + str(e)
        )


# =========================
# HEADER
# =========================

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


# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.header("⚙️ Knowledge Center")

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

                    st.success(
                        f"{file.name}: "
                        f"{count} chunks indexed"
                    )

        else:

            st.warning(
                "Please upload a file first."
            )

    st.divider()

    st.subheader("📊 Knowledge Base")

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

        st.rerun()


# =========================
# DASHBOARD
# =========================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "📄 Documents",
        len(st.session_state.documents)
    )

with col2:

    st.metric(
        "🧩 Knowledge Chunks",
        len(st.session_state.chunks)
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


# =========================
# FEATURES
# =========================

st.markdown(
    "## 🚀 System Features"
)

c1, c2, c3 = st.columns(3)

with c1:

    st.markdown(
        """
        <div class="card">
        <h3>📝 NLP Processing</h3>
        Text cleaning, tokenization
        and preprocessing.
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
        <h3>🧠 Semantic Search</h3>
        Deep-learning embeddings
        for document retrieval.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# CHAT HISTORY
# =========================

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


# =========================
# USER QUESTION
# =========================

question = st.chat_input(
    "✨ Ask anything..."
)

if question:

    st.session_state.questions += 1

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # NLP

    processed_question = clean_text(
        question
    )

    # ML

    intent, confidence = predict_intent(
        processed_question
    )

    # RAG

    document_results = search_documents(
        question
    )

    context_parts = []

    source_list = []

    for result in document_results:

        context_parts.append(
            result["text"]
        )

        source_list.append(
            f"📄 {result['source']} "
            f"• Similarity: "
            f"{result['score']:.2f}"
        )

    # External knowledge

    web_results = []

    if external_search:

        if not document_results:

            web_results = wikipedia_search(
                question
            )

        elif confidence < 0.45:

            web_results = wikipedia_search(
                question
            )

    for result in web_results:

        context_parts.append(
            result["text"]
        )

        source_list.append(
            f"🌐 {result['title']}"
        )

    context = "\n\n".join(
        context_parts
    )

    # If no knowledge

    if not context:

        context = (
            "No matching document "
            "or external knowledge was found."
        )

    # AI

    answer = generate_ai_answer(
        question,
        context,
        intent
    )

    # Assistant

    with st.chat_message(
        "assistant"
    ):

        st.markdown(answer)

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

            elif web_results:

                st.write(
                    "**Knowledge Source:** "
                    "Wikipedia"
                )

            else:

                st.write(
                    "**Knowledge Source:** "
                    "No source found"
                )

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

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


# =========================
# FOOTER
# =========================

st.markdown(
"""
<hr>

<center>

<b>🧠 IntelliMind AI</b>

<br>

NLP • Machine Learning • Deep Learning • RAG • Generative AI

<br><br>

Built as an NLP & AI academic project.

</center>
""",
unsafe_allow_html=True
)
