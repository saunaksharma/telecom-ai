import streamlit as st
from src.query import load_index, load_docs, retrieve, generate_answer

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Airtel AI Assistant",
    page_icon="📡",
    layout="wide"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
body {
    background-color: #0e1117;
}

/* Header */
.header {
    display: flex;
    align-items: center;
    gap: 15px;
    font-size: 34px;
    font-weight: bold;
    color: #e60000;
}

.subtext {
    color: #bbbbbb;
    margin-bottom: 20px;
}

/* Chat bubbles */
.user-msg {
    background-color: #1f2937;
    padding: 12px 15px;
    border-radius: 12px;
    margin: 10px 0;
}

.ai-msg {
    background-color: #111827;
    padding: 15px;
    border-radius: 12px;
    border-left: 4px solid #e60000;
    margin: 10px 0;
}

/* Input */
.stTextInput input {
    background-color: #1c1f26 !important;
    color: white !important;
}

/* Buttons */
.stButton button {
    background-color: #e60000;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ===== HEADER WITH LOGO =====
col1, col2 = st.columns([1, 8])

with col1:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/5/5e/Airtel_logo.svg",
        width=60
    )

with col2:
    st.markdown('<div class="header">Airtel AI Assistant</div>', unsafe_allow_html=True)

st.markdown('<div class="subtext">Smart telecom support powered by AI</div>', unsafe_allow_html=True)

# ===== SIDEBAR =====
st.sidebar.title("⚙️ Controls")

k = st.sidebar.slider("🔍 Retrieval Depth", 1, 5, 3)

st.sidebar.markdown("---")

st.sidebar.markdown("### 📊 System Info")
st.sidebar.write("Model: llama3:8b")
st.sidebar.write("Vector DB: FAISS")
st.sidebar.write("Pipeline: RAG")

st.sidebar.markdown("---")

st.sidebar.markdown("### 📂 Data Info")
# We'll show count later after loading

# ===== LOAD DATA =====
@st.cache_resource
def load_resources():
    return load_index(), load_docs()

index, docs = load_resources()

st.sidebar.write(f"Documents indexed: {len(docs)}")

# ===== SESSION STATE =====
if "messages" not in st.session_state:
    st.session_state.messages = []

# ===== DISPLAY CHAT =====
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-msg">🧑 {msg["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="ai-msg">🤖 {msg["content"]}</div>',
            unsafe_allow_html=True
        )

        with st.expander("📚 Retrieved Context"):
            for doc in msg["context"]:
                st.write("•", doc)

# ===== INPUT (ENTER TO SEND) =====
query = st.chat_input("Ask your telecom question...")

if query:
    # Show user message instantly
    st.session_state.messages.append({"role": "user", "content": query})

    st.markdown(
        f'<div class="user-msg">🧑 {query}</div>',
        unsafe_allow_html=True
    )

    # Generate answer
    with st.spinner("🤖 Thinking..."):
        context_docs = retrieve(query, index, docs, k=k)
        context = " ".join(context_docs)

        answer = generate_answer(query, context)

    # Store AI response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "context": context_docs
    })

    st.markdown(
        f'<div class="ai-msg">🤖 {answer}</div>',
        unsafe_allow_html=True
    )

# ===== CLEAR CHAT =====
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# ===== FOOTER =====
st.markdown("---")
st.markdown(
    "<center style='color:gray;'>🔴 Airtel AI • Built with Ollama + FAISS + Streamlit</center>",
    unsafe_allow_html=True
)