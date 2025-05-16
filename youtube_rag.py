# Youtube RAG class that will load a csv of video metadata and transcripts,
# load the data into a vector database, and create a pipeline to answer
# questions about the data

from operator import itemgetter
import os
from pathlib import Path

from dataclasses import dataclass, field
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableSerializable,
    RunnableParallel,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader

import pandas as pd


@dataclass
class YoutubeRAG:
    """
    Youtube RAG class that will load a csv of video metadata and transcripts,
    load the data into a vector database, and create a pipeline to answer
    questions about the data
    """

    csv_path: str
    db_path: str
    videos_df: pd.DataFrame = field(init=False)
    embedding_function: OpenAIEmbeddings = field(init=False)
    vectorstore: Chroma = field(init=False)
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    embedding_model: str = "text-embedding-3-small"

    def __post_init__(self):
        # Ensure environment variables are loaded
        load_dotenv()

        # Verify API key exists
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

        self.videos_df = self._load_videos_df()
        self.embedding_function = OpenAIEmbeddings(model=self.embedding_model)
        self.vectorstore = self._load_vectorstore()

    def _load_videos_df(self) -> pd.DataFrame:
        """Load video data from CSV file into a pandas DataFrame."""
        return pd.read_csv(self.csv_path)

    def __inject_metadata(self, documents: list[Document]) -> list[Document]:
        """
        Inject metadata (title, author, video URL) into document content.

        Args:
            documents: List of Document objects containing video information

        Returns:
            List of Document objects with injected metadata
        """
        injected_docs = []
        print("Length of documents:", len(documents))
        for doc in documents:
            title = doc.metadata.get("title", "Unknown Title")
            channel = doc.metadata.get("channel", "Unknown Channel")
            video_url = doc.metadata.get("video_url", "Unknown URL")

            # print("Document:", title, channel, video_url)
            # print("Metadata:", doc.metadata)

            # Create a summary header with metadata
            metadata_header = (
                f"Title: {title}\nChannel: {channel}\nVideo URL: {video_url}\n\n"
            )

            # Add metadata to document content
            updated_content = metadata_header + doc.page_content

            # Preserve all metadata
            metadata = doc.metadata

            injected_docs.append(
                Document(page_content=updated_content, metadata=metadata)
            )
        return injected_docs

    def __load_documents(self) -> list[Document]:
        """Load documents from DataFrame and inject metadata."""
        loader = DataFrameLoader(self.videos_df, page_content_column="page_content")
        docs = loader.load()

        # Inject title, channel and URL into content to make it visible to LLM
        docs = self.__inject_metadata(docs)

        return docs

    def _load_vectorstore(self) -> Chroma:
        """
        Load or create Chroma vector store from documents.

        Returns:
            Chroma vector store instance with embedded documents
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Check if existing DB can be loaded
        if os.path.exists(self.db_path) and os.listdir(self.db_path):
            try:
                vectorstore = Chroma(
                    embedding_function=self.embedding_function,
                    persist_directory=self.db_path,
                )
                # Check if the database has documents
                if vectorstore._collection.count() > 0:
                    print(
                        f"Loaded existing vectorstore with {vectorstore._collection.count()} documents"
                    )
                    return vectorstore
                else:
                    print("Vectorstore exists but is empty, creating new one...")
            except Exception as e:
                print(
                    f"Error loading existing vectorstore: {str(e)}, creating new one..."
                )

        # Create new vectorstore
        print("Creating new vectorstore...")
        raw_docs = self.__load_documents()
        splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=1000)
        docs = splitter.split_documents(raw_docs)
        print(f"Split documents into {len(docs)} chunks")

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding_function,
            persist_directory=self.db_path,
        )

        # No need to call persist() - Chroma automatically persists when persist_directory is provided
        print(f"Created vectorstore with {vectorstore._collection.count()} documents")

        return vectorstore

    def is_vectorstore_empty(self) -> bool:
        """
        Check if the vector store is empty.

        Returns:
            bool: True if the vector store is empty, False otherwise
        """
        if not hasattr(self, "vectorstore") or self.vectorstore is None:
            return True

        # Get the count of documents in the collection
        count = self.vectorstore._collection.count()
        return count == 0

    @property
    def rag_pipeline(self) -> RunnableSerializable:
        """
        Create RAG pipeline for question answering.

        Returns:
            Runnable pipeline that processes questions using chat history and retrieved context
        """
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("history"),
                ("human", "{question}"),
            ]
        )

        qa_system_prompt = """You are a marketing AI assistant powered by YouTube video content.
        Use the following pieces of retrieved YouTube video content to answer questions about marketing.
        
        Be specific in your answers and cite information from the videos when possible. Mention the video titles or channels
        if it helps provide authority to your answer. Don't make up information - if the context doesn't contain relevant
        information, acknowledge that limitation.
        
        Make sure to ALWAYS reference the video urls in your response. Always respond in markdown format.
        
        Remember you are a helpful marketing advisor who uses YouTube content to provide value.
        
        Here is the retrieved context from relevant YouTube videos:
        {context}"""

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("history"),
                ("human", "{question}"),
            ]
        )

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        llm = ChatOpenAI(model=self.llm_model, temperature=self.llm_temperature)

        # Named components for better tracing in LangSmith
        contextualize_question_chain = (
            contextualize_q_prompt | llm | StrOutputParser()
        ).with_config({"run_name": "contextualize_question"})

        retriever_chain = retriever.with_config(
            {"run_name": "retrieve_youtube_content"}
        )

        response_chain = (qa_prompt | llm | StrOutputParser()).with_config(
            {"run_name": "generate_response"}
        )

        return (
            RunnableParallel(
                context_question=contextualize_question_chain,
                question=itemgetter("question") | RunnablePassthrough(),
                history=itemgetter("history") | RunnablePassthrough(),
            )
            | {
                "context": itemgetter("context_question") | retriever_chain,
                "question": itemgetter("question") | RunnablePassthrough(),
                "context_question": itemgetter("context_question")
                | RunnablePassthrough(),
                "history": itemgetter("history") | RunnablePassthrough(),
            }
            | {
                "response": response_chain,
                "question": itemgetter("question"),
                "context": itemgetter("context"),
                "context_question": itemgetter("context_question"),
            }
        )

    def clear_vectorstore(self):
        """
        Clear all documents from the vector store without deleting the directory.
        This allows for reusing the same database location while starting fresh.
        """
        if hasattr(self, "vectorstore") and self.vectorstore is not None:
            try:
                # Access the underlying Chroma collection and delete all documents
                self.vectorstore._collection.delete(where={})
                # No need to call persist() - Chroma handles this automatically
                return True
            except Exception as e:
                print(f"Error clearing vector store: {str(e)}")
                return False
        return False

    def rebuild_vectorstore(self):
        """
        Rebuild the vector store from scratch using the loaded documents.
        Use this when you need to force a complete refresh of the embeddings.

        Returns:
            bool: True if rebuild was successful, False otherwise
        """
        try:
            # Clear existing data first
            if hasattr(self, "vectorstore") and self.vectorstore is not None:
                self.clear_vectorstore()

            # Process documents
            raw_docs = self.__load_documents()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=15000, chunk_overlap=1000
            )
            docs = splitter.split_documents(raw_docs)
            print(f"Split documents into {len(docs)} chunks for rebuild")

            # Create new vectorstore at the same location
            self.vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embedding_function,
                persist_directory=self.db_path,
            )

            # No need to call persist() - Chroma automatically persists when persist_directory is provided
            count = self.vectorstore._collection.count()
            print(f"Rebuilt vectorstore with {count} documents")
            return count > 0

        except Exception as e:
            print(f"Error rebuilding vector store: {str(e)}")
            return False


if __name__ == "__main__":
    # example usage
    yt_rag = YoutubeRAG(
        csv_path="data/youtube_videos.csv",
        db_path="data/youtube_videos_db",
    )

    # Check if the vector store is empty
    is_empty = yt_rag.is_vectorstore_empty()
    print(f"Vector store is empty: {is_empty}")

    if is_empty:
        print("Vector store is empty, rebuilding...")
        rebuild_success = yt_rag.rebuild_vectorstore()
        print(f"Rebuild {'successful' if rebuild_success else 'failed'}")
        # Check again after rebuild
        is_empty = yt_rag.is_vectorstore_empty()
        print(f"Vector store is empty after rebuild: {is_empty}")

    # Test the vectorstore to see if it's working
    results = yt_rag.vectorstore.similarity_search(
        "What is Day Trading Attention and how does it impact trading strategies?"
    )
    print(f"Found {len(results)} results")

    # If results are found, print the first one to verify content
    if results:
        print(f"First result: {results[0].page_content[:100]}...")
    else:
        print("No results found. Vector store may still be empty.")

    # Test the RAG pipeline
    response = yt_rag.rag_pipeline.invoke(
        {
            "question": "How do I implement Day Trading Attention?",
            "history": [],
        }
    )
    print("RAG pipeline response:", response)
