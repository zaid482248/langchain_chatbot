"""
Professional AI Chatbot Interface using Streamlit
RAG Chatbot with LangChain and Ollama
"""

import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime


# Page Configuration
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container styling */
    .stChatMessage {
        border-radius: 12px;
        padding: 12px;
        margin: 8px 0;
    }

    /* User message styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #f0f2f6;
    }

    /* Assistant message styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #e8f4fd;
    }

    /* Chat input styling */
    .stChatInput textarea {
        border-radius: 12px;
        border: 1px solid #ddd;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }

    /* Header styling */
    .header-container {
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        margin-bottom: 20px;
        color: white;
    }

    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }

    .header-subtitle {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 8px;
    }

    /* Stats cards */
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 20px 0;
    }

    .stat-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        flex: 1;
        margin: 0 5px;
    }

    .stat-number {
        font-size: 1.5rem;
        font-weight: bold;
        color: #667eea;
    }

    .stat-label {
        font-size: 0.85rem;
        color: #666;
    }

    /* Hide default Streamlit footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def init_llm():
    """Initialize the LLM model"""
    return ChatOllama(
        model=st.session_state.get("selected_model", "qwen2.5-coder:1.5b"),
        temperature=st.session_state.get("temperature", 0.7)
    )


def init_prompt():
    """Initialize the prompt template"""
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert AI assistant in {topic}. Provide concise, accurate, and helpful answers."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ]
    )


def get_chain(llm, prompt):
    """Create the LCEL chain"""
    return prompt | llm | StrOutputParser()


def initialize_session_state():
    """Initialize session state variables"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_count" not in st.session_state:
        st.session_state.conversation_count = 0
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "qwen2.5-coder:1.5b"
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    if "topic" not in st.session_state:
        st.session_state.topic = "AI and Technology"


def render_sidebar():
    """Render the sidebar with controls"""
    with st.sidebar:
        # Header
        st.markdown("### ⚙️ Settings")
        st.markdown("---")

        # Model selection
        available_models = [
            "qwen2.5-coder:1.5b",
            "qwen2.5-coder:3b",
            "llama3.2",
            "mistral",
            "gemma2"
        ]
        st.session_state.selected_model = st.selectbox(
            "🤖 Model",
            options=available_models,
            index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
            help="Select the AI model to use"
        )

        # Temperature slider
        st.session_state.temperature = st.slider(
            "🌡️ Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Higher values make output more random, lower values more deterministic"
        )

        # Topic selection
        st.session_state.topic = st.text_input(
            "📚 Topic/Domain",
            value=st.session_state.topic,
            help="Set the expertise domain for the AI assistant"
        )

        st.markdown("---")

        # Conversation stats
        st.markdown("### 📊 Statistics")
        st.metric("Messages", len(st.session_state.messages))
        st.metric("Topic", st.session_state.topic)

        st.markdown("---")

        # Actions
        st.markdown("### 🛠️ Actions")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.messages = []
                st.session_state.conversation_count = 0
                st.rerun()

        with col2:
            if st.button("💾 Export", use_container_width=True):
                export_chat()

        st.markdown("---")

        # Info section
        st.markdown("### ℹ️ About")
        st.markdown("""
        **AI Chatbot** powered by:
        - LangChain
        - Ollama
        - Streamlit

        Built for professional conversations.
        """)


def export_chat():
    """Export chat history as text"""
    if st.session_state.messages:
        chat_text = "AI Chatbot Conversation History\n"
        chat_text += "=" * 40 + "\n"
        chat_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        chat_text += f"Topic: {st.session_state.topic}\n\n"

        for msg in st.session_state.messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            chat_text += f"{role}: {msg['content']}\n"
            chat_text += "-" * 40 + "\n"

        st.download_button(
            label="📥 Download Chat History",
            data=chat_text,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )


def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🤖 AI Assistant</h1>
        <p class="header-subtitle">Professional RAG-powered Chatbot</p>
    </div>
    """, unsafe_allow_html=True)


def render_chat():
    """Render the chat interface"""
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything...", key="chat_input"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    llm = init_llm()
                    prompt_template = init_prompt()
                    chain = get_chain(llm, prompt_template)

                    # Convert messages to LangChain format
                    chat_history = []
                    for msg in st.session_state.messages[:-1]:  # Exclude current user message
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        else:
                            chat_history.append(AIMessage(content=msg["content"]))

                    response = chain.invoke({
                        "topic": st.session_state.topic,
                        "question": prompt,
                        "chat_history": chat_history
                    })

                    st.markdown(response)

                    # Add assistant message
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.chat_history.append(HumanMessage(content=prompt))
                    st.session_state.chat_history.append(AIMessage(content=response))
                    st.session_state.conversation_count += 1

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


def main():
    """Main application"""
    # Initialize session state
    initialize_session_state()

    # Render sidebar
    render_sidebar()

    # Render main content
    render_header()
    render_chat()


if __name__ == "__main__":
    main()
