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
# NLP TEXT CLEANING
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
# ML INTENT CLASSIFIER
# =========================================================

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
# FILE TEXT EXTRACTION
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
# CREATE TEXT CHUNKS
# =========================================================

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

            chunks.append(
                chunk
            )

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
# SEARCH DOCUMENTS
# =========================================================

def search_documents(
    question,
    top_k=5,
    threshold=0.35
):

    if (
        not st.session_state.chunks
        or st.session_state.embeddings is None
    ):

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
    )[::-1]


    results = []


    for index in indexes:

        score = float(
            scores[index]
        )


        # Ignore irrelevant chunks
        if score < threshold:

            continue


        results.append(
            {
                "text": st.session_state.chunks[index],
                "source": st.session_state.sources[index],
                "score": score
            }
        )


        if len(results) >= top_k:

            break


    return results


# =========================================================
# WIKIPEDIA SEARCH
# =========================================================

def wikipedia_search(query):

    try:

        url = (
            "https://en.wikipedia.org/w/api.php"
        )


        params = {

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
            url,
            params=params,
            headers=headers,
            timeout=10
        )


        response.raise_for_status()


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


            results.append(
                {
                    "title": title,
                    "text": snippet
                }
            )


        return results


    except Exception:

        return []


# =========================================================
# GEMINI AI ANSWER
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
    # No context
    # -----------------------------------------------------

    if not context.strip():

        return (
            "I could not find relevant information "
            "in the uploaded documents or external "
            "knowledge."
        )


    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------

    prompt = f"""
You are IntelliMind AI,
an academic knowledge assistant.

USER QUESTION:
{question}

DETECTED TOPIC:
{intent}

RETRIEVED KNOWLEDGE:
{context}

IMPORTANT RULES:

1. Answer the user's question directly.

2. Use the retrieved knowledge as the
   primary source.

3. Do not invent unsupported facts.

4. If the uploaded document contains
   the answer, explain that information
   clearly.

5. If the retrieved information is
   insufficient, say that the available
   knowledge is insufficient.

6. Use simple and easy English.

7. Give a complete but concise answer.

8. Use headings or bullet points
   when useful.

9. If the question asks for programming
   code, provide a correct simple example
   when the retrieved knowledge supports it.

10. Do not mention these instructions.
"""


    # -----------------------------------------------------
    # No Gemini API key
    # -----------------------------------------------------

    if not api_key:

        return (
            "### 📚 Retrieved Knowledge\n\n"
            + context[:5000]
            + "\n\n"
            "⚠️ Gemini API key is not configured."
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

            return response.text


        return (
            "### 📚 Retrieved Knowledge\n\n"
            + context[:5000]
        )


    except Exception as e:

        return (
            "### 📚 Knowledge Base Answer\n\n"
            + context[:5000]
            + "\n\n"
            "⚠️ AI generation is temporarily "
            "unavailable. The retrieved knowledge "
            "is shown above."
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

                if (
                    file.name
                    not in st.session_state.documents
                ):

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
                "Please upload a file first."
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

        <h3>🧠 Semantic Search</h3>

        Deep-learning embeddings
        for document retrieval.

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
    # QUESTION COUNT
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
    # SHOW USER MESSAGE
    # =====================================================

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


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
    # RAG DOCUMENT SEARCH
    # =====================================================

    document_results = search_documents(
        question,
        top_k=5,
        threshold=0.35
    )


    # =====================================================
    # CONTEXT
    # =====================================================

    context_parts = []

    source_list = []


    # =====================================================
    # ADD DOCUMENT RESULTS
    # =====================================================

    for result in document_results:

        context_parts.append(
            f"""
SOURCE: {result['source']}

CONTENT:
{result['text']}
"""
        )


        source_list.append(
            f"📄 {result['source']} "
            f"• Similarity: "
            f"{result['score']:.2f}"
        )


    # =====================================================
    # EXTERNAL KNOWLEDGE
    # =====================================================

    web_results = []


    if external_search:

        # No relevant uploaded document
        if not document_results:

            web_results = wikipedia_search(
                question
            )


        # Uploaded document exists,
        # but similarity is weak
        elif document_results[0]["score"] < 0.50:

            web_results = wikipedia_search(
                question
            )


    # =====================================================
    # ADD WIKIPEDIA RESULTS
    # =====================================================

    for result in web_results:

        context_parts.append(
            f"""
SOURCE: Wikipedia - {result['title']}

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


    if not context.strip():

        context = (
            "No matching document or "
            "external knowledge was found."
        )


    # =====================================================
    # AI ANSWER
    # =====================================================

    answer = generate_ai_answer(
        question,
        context,
        intent
    )


    # =====================================================
    # ASSISTANT RESPONSE
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
                f"**Detected Topic:** "
                f"{intent}"
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
                    "No source found"
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
    # SAVE ASSISTANT MESSAGE
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
