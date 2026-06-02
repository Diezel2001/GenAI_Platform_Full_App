"""
Streamlit Chatbot Frontend for Agentic Generative AI Platform
"""

import streamlit as st
import requests
import json
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="AI AGENT CHATBOT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    /* Thinking animation styles */
    .thinking-container {
        display: flex;
        align-items: center;
        gap: 4px;
        font-size: 16px;
        padding: 0.5rem 0;
    }
    .thinking-emoji {
        animation: pulse 1.5s ease-in-out infinite;
    }
    .thinking-dot {
        animation: blink 1.4s infinite;
        opacity: 0;
    }
    .thinking-dot:nth-child(3) { animation-delay: 0s; }
    .thinking-dot:nth-child(4) { animation-delay: 0.2s; }
    .thinking-dot:nth-child(5) { animation-delay: 0.4s; }
    @keyframes blink {
        0%, 20% { opacity: 0; }
        50% { opacity: 1; }
        100% { opacity: 0; }
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.2); }
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Session State Initialization
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "server_host" not in st.session_state:
    st.session_state.server_host = "localhost"

if "server_port" not in st.session_state:
    st.session_state.server_port = 8000

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ----------------------------
# Helper Functions
# ----------------------------
def get_base_url() -> str:
    return f"http://{st.session_state.server_host}:{st.session_state.server_port}"

def make_post_request(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{get_base_url()}{endpoint}"
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}

def upload_file_with_retry(file, max_retries=3):
    url = f"{get_base_url()}/documents/upload"

    for attempt in range(max_retries):
        try:
            files = {
                "file": (
                    file.name,
                    file.getvalue(),
                    "application/pdf"
                )
            }

            response = requests.post(url, files=files, timeout=60)

            if response.status_code == 200:
                return {"success": True, "data": response.json()}

        except Exception as e:
            if attempt == max_retries - 1:
                return {"success": False, "error": str(e)}

        time.sleep(1 * (attempt + 1))

    return {"success": False, "error": "Upload failed"}

def show_thinking_animation(placeholder):
    """Display an animated thinking indicator in the given placeholder."""
    placeholder.markdown(
        """
        <div class="thinking-container">
            <span class="thinking-emoji">🤔</span>
            <span>Thinking</span>
            <span class="thinking-dot">.</span>
            <span class="thinking-dot">.</span>
            <span class="thinking-dot">.</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------
# SIDEBAR
# ----------------------------
with st.sidebar:
    st.title("⚙️ Server Config")

    st.session_state.server_host = st.text_input("Host", st.session_state.server_host)
    st.session_state.server_port = st.number_input("Port", value=st.session_state.server_port)

    st.caption(f"URL: {get_base_url()}")

    st.divider()

    # ----------------------------
    # 📄 FILE UPLOAD IN SIDEBAR
    # ----------------------------
    st.subheader("📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Drag & drop or select a PDF",
        type=["pdf"],
        key="sidebar_uploader"
    )

    if uploaded_file is not None:
        st.caption(f"Selected: {uploaded_file.name}")

        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("File too large (max 10MB)")
        else:
            if st.button("📤 Upload", use_container_width=True):
                with st.spinner("Uploading..."):
                    result = upload_file_with_retry(uploaded_file)

                    if result["success"]:
                        st.success("Uploaded!")
                        st.json(result["data"])

                        # Add to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"📄 Uploaded document: {uploaded_file.name}"
                        })
                    else:
                        st.error(result["error"])

    st.divider()

    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------------------
# MAIN CHAT UI
# ----------------------------
st.title("🤖 AI AGENT CHATBOT")

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------------------
# CHAT INPUT
# ----------------------------
user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Show animated thinking indicator while waiting for response
        thinking_placeholder = st.empty()
        show_thinking_animation(thinking_placeholder)

        result = make_post_request("/agent/", {
            "message": user_input,
            "k": 5,
            "session_id": st.session_state.session_id
        })

        # Clear the thinking animation and show the actual response
        thinking_placeholder.empty()

        if result["success"]:
            data = result["data"]

            reply = f"""
**Analysis:** {data.get("analysis")}

**Route:** {data.get("route")}

**Result:** {data.get("results")}
"""
            st.markdown(reply)

            st.session_state.messages.append({
                "role": "assistant",
                "content": reply
            })
        else:
            st.error(result["error"])