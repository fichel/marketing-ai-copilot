# 🚀 Marketing AI Copilot

Your awesome AI sidekick that scans YouTube videos and delivers marketing insights! Ask it anything about marketing and watch it work its magic.

## ✨ What This Tool Can Do

- 🔍 Find the best marketing content on YouTube
- 🧠 Process video transcripts efficiently
- 💾 Store knowledge in a vector database for quick retrieval
- 💬 Answer your marketing questions with relevant information
- 🧵 Remember your conversation for contextual responses

## 🛠️ Getting Started

1. Clone this repo
2. Install the dependencies:
```bash
uv sync
```
3. Create a `.env` file with your keys:
```
OPENAI_API_KEY=your_openai_key
YOUTUBE_API_KEY=your_youtube_key
LANGCHAIN_API_KEY=your_langsmith_key (optional)
LANGCHAIN_PROJECT=marketing-ai-copilot (optional)
LANGCHAIN_TRACING_V2=true (optional)
```

## 🏃‍♀️ Fire It Up!

1. Launch the app:
```bash
streamlit run app.py
```
2. Enter your OpenAI API key in the sidebar
3. Optionally, add your LangSmith API key for enhanced features
4. Start asking marketing questions!

## 🔬 LangSmith Integration

This app comes with LangSmith integration for advanced monitoring and debugging:

### 🎁 Cool Features
- 📊 Visualize the RAG pipeline in real-time
- 💰 Monitor token usage and costs
- 🔢 Track conversations with unique IDs
- 🐞 Debug retrieval quality
- 📈 Analyze system performance

### 🔧 Setting Up LangSmith

1. Create a LangSmith account at [smith.langchain.com](https://smith.langchain.com/)
2. Generate an API key in your settings
3. Add the key to the app sidebar
4. Customize your project name if desired (default: marketing-ai-copilot)

### 🧪 Using LangSmith Effectively

With LangSmith configured, each interaction is traced, allowing you to:

1. View detailed execution logs in the LangSmith UI
2. See the context being used for responses
3. Evaluate the quality of generated content
4. Track token usage across different models

Enable debug mode to get a direct link to the current trace in the LangSmith dashboard.

## 🛠️ Customization Options

Make it your own:

- Modify the default topic in `app.py` (TOPIC variable)
- Adjust retrieval parameters in `youtube_rag.py`
- Customize the RAG prompts in `youtube_rag.py`
