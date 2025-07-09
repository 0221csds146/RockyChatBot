import os
import streamlit as st
import time
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader

# Load environment variables from .env
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# Streamlit UI setup
st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# URL input
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.9,
    max_tokens=500
)

if process_url_clicked and urls:
    # Load documents from URLs using WebBaseLoader
    loader = WebBaseLoader(urls)
    main_placeholder.text("🔄 Loading data from URLs...")
    data = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("📚 Splitting text into chunks...")
    docs = text_splitter.split_documents(data)

    # Generate embeddings and store in FAISS
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
    
    main_placeholder.text("✅ Vectorstore created and saved!")

# Input query and show answer
query = main_placeholder.text_input("Ask a question about the articles:")
if query:
    if st.session_state.vectorstore is not None:
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever()
        )
        result = chain({"question": query}, return_only_outputs=True)

        st.header("📌 Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("🔗 Sources:")
            for source in sources.split("\n"):
                st.write(source)
    else:
        st.warning("Please process some URLs first!")