import streamlit as st
import os
from pathlib import Path
from src.youtube_scraper import YoutubeScraper
from src.youtube_rag import YoutubeRAG
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import (
    StreamlitChatMessageHistory,
)
from langsmith import Client
import uuid

st.title("📣🤖 Your Marketing AI Copilot")

# Initialize session state for API keys
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "youtube_api_key" not in st.session_state:
    st.session_state.youtube_api_key = ""
if "api_key_configured" not in st.session_state:
    st.session_state.api_key_configured = False
if "langsmith_configured" not in st.session_state:
    st.session_state.langsmith_configured = False
if "run_id" not in st.session_state:
    st.session_state.run_id = str(uuid.uuid4())

# API key input section
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input(
        "Enter your OpenAI API key:", type="password", key="api_key_input"
    )

    youtube_api_key = st.text_input(
        "Enter your YouTube API key:", type="password", key="youtube_api_key_input"
    )

    # LangSmith API key input
    langsmith_api_key = st.text_input(
        "Enter your LangSmith API key (optional):",
        type="password",
        key="langsmith_api_key_input",
    )

    langsmith_project_name = st.text_input(
        "LangSmith Project Name:",
        value="marketing-ai-copilot",
        key="langsmith_project_name",
    )

    if st.button("Set API Keys"):
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            st.session_state.openai_api_key = openai_api_key
            st.session_state.api_key_configured = True
            st.success("OpenAI API key set successfully!")
        else:
            st.error("Please enter a valid OpenAI API key.")

        if youtube_api_key:
            os.environ["YOUTUBE_API_KEY"] = youtube_api_key
            st.session_state.youtube_api_key = youtube_api_key
            st.success("YouTube API key set successfully!")
        else:
            st.warning("YouTube API key not provided. Some features may be limited.")

        if langsmith_api_key:
            os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = langsmith_project_name
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            st.session_state.langsmith_configured = True
            st.success("LangSmith configured successfully!")

# Only proceed if API key is configured
if not st.session_state.api_key_configured:
    st.info("Please enter your OpenAI API key in the sidebar to get started.")
    st.stop()

# Check if YouTube API key is present
if not st.session_state.youtube_api_key:
    st.sidebar.warning("YouTube API key not found. Some features may be limited.")

# Show LangSmith status
if st.session_state.langsmith_configured:
    st.sidebar.success("✅ LangSmith monitoring enabled")
else:
    st.sidebar.info(
        "ℹ️ LangSmith monitoring disabled. Add your LangSmith API key for tracing and monitoring."
    )

TOPIC = "Social Media Brand Strategy Tips"
LLM_OPTIONS = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini"]

# Ensure data directory exists
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, "youtube_videos.csv")
DB_PATH = os.path.join(DATA_DIR, "youtube_videos_db")

model_option = st.sidebar.selectbox("Choose OpenAI model", LLM_OPTIONS, index=0)

LLM = model_option

# Create a new session state to track initialization
if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = False

msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")


def create_chain():
    # Scrape videos
    if not Path(CSV_PATH).exists():
        try:
            st.info(f"Fetching YouTube videos...")
            yt_scraper = YoutubeScraper(topic=TOPIC)
            yt_scraper.search_videos()

            if not yt_scraper.video_ids:
                st.error(
                    f"No videos found for topic '{TOPIC}'. Please check your YouTube API key or try a different topic."
                )
                return None

            yt_scraper.load_videos_data(add_video_info=True)

            if yt_scraper.videos_df.empty:
                st.error(
                    "Failed to load any video data. Please check your API credentials and internet connection."
                )
                return None

            yt_scraper.save_videos_data(CSV_PATH)
            st.success(f"Successfully scraped {len(yt_scraper.videos_df)} videos!")
        except Exception as e:
            st.error(f"Error creating video database: {str(e)}")
            return None

    # Create RAG pipeline
    try:
        st.info("Loading YouTube video data for RAG pipeline...")
        yt_rag = YoutubeRAG(csv_path=CSV_PATH, db_path=DB_PATH, llm_model=LLM)
        st.session_state.rag_initialized = True
        st.success("RAG pipeline initialized successfully!")

        chain = RunnableWithMessageHistory(
            runnable=yt_rag.rag_pipeline,
            get_session_history=lambda session_id: msgs,
            input_messages_key="question",
            history_messages_key="history",
            output_messages_key="response",
        )

        return chain
    except Exception as e:
        st.error(f"Error initializing RAG pipeline: {str(e)}")
        return None


def reset_rag():
    if Path(CSV_PATH).exists():
        try:
            Path(CSV_PATH).unlink()
            st.sidebar.success("YouTube data file removed.")
        except Exception as e:
            st.sidebar.error(f"Error removing CSV file: {str(e)}")

    st.session_state.rag_initialized = False
    st.session_state.pop("langchain_messages", None)
    st.rerun()


# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).markdown(msg.content)

# Process user input
if question := st.chat_input("Enter your Marketing question here:"):
    st.chat_message("human").markdown(question)
    with st.chat_message("ai"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            chain = create_chain()
            if chain is None:
                message_placeholder.markdown(
                    "Sorry, I couldn't access the video database. Please check the error message above."
                )
            else:
                # Debug information
                st.info("Retrieving information from YouTube videos...")

                # Add LangSmith trace tags for better tracking
                run_id = st.session_state.run_id
                metadata = {
                    "conversation_id": run_id,
                    "user_question": question,
                    "model": LLM,
                }

                # Configure the run with LangSmith metadata if enabled
                config = {
                    "configurable": {"session_id": run_id},
                }

                if st.session_state.langsmith_configured:
                    config["metadata"] = metadata
                    config["tags"] = ["streamlit", "marketing-copilot", LLM]

                response = chain.invoke(
                    {"question": question},
                    config=config,
                )

                # Display the context used if in debug mode
                if st.session_state.get("debug_mode", False):
                    with st.expander("Retrieved YouTube context"):
                        st.markdown(
                            f"**Query used:** {response.get('context_question', 'N/A')}"
                        )
                        st.markdown("**Sources:**")
                        for doc in response.get("context", []):
                            st.markdown(
                                f"- {doc.metadata.get('title', 'Unknown')} (from {doc.metadata.get('channel', 'Unknown')})"
                            )

                    # If LangSmith is configured, show a link to the run
                    if st.session_state.langsmith_configured:
                        langsmith_project = (
                            langsmith_project_name  # Use the input value, not env var
                        )
                        langsmith_url = f"https://smith.langchain.com/p/{langsmith_project}/r/{run_id}"
                        st.markdown(f"[View this trace in LangSmith]({langsmith_url})")

                message_placeholder.markdown(response["response"])

# Add debug toggle to sidebar
st.sidebar.divider()
st.session_state.debug_mode = st.sidebar.checkbox("Show debug information", value=False)

# Show database info
if st.session_state.rag_initialized:
    st.sidebar.success("✅ YouTube data loaded successfully")
else:
    st.sidebar.warning("⚠️ YouTube data not yet loaded")

# Add reset button
st.sidebar.divider()
if st.sidebar.button("Reset RAG System"):
    reset_rag()
