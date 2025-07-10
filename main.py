import os
import streamlit as st
import time
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.schema import Document

# Page configuration
st.set_page_config(
    page_title="RockyBot - Interactive News Research Tool",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .success-message {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
    .url-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
    .url-success {
        background: #d4edda;
        color: #155724;
    }
    .url-error {
        background: #f8d7da;
        color: #721c24;
    }
    .chat-bubble {
    background: #e3f2fd;
    color: #000000;  /* Make question text visible */
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    border-left: 4px solid #2196f3;
    font-weight: 500;
}

.answer-bubble {
    background: #f3e5f5;
    color: #000000;  /* Make answer text visible */
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    border-left: 4px solid #9c27b0;
    font-weight: 500;
}


</style>
""", unsafe_allow_html=True)

# Load API key
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except:
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def extract_content_with_requests(url):
    """Enhanced content extraction with better error handling"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
            element.decompose()
        
        # Try to find main content areas
        content_selectors = [
            'article', '[role="main"]', '.content', '.article-content', 
            '.post-content', '.entry-content', '.story-body', 'main', '.main-content'
        ]
        
        content_text = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                content_text = ' '.join([elem.get_text(strip=True) for elem in elements])
                break
        
        if not content_text or len(content_text) < 100:
            paragraphs = soup.find_all('p')
            content_text = ' '.join([p.get_text(strip=True) for p in paragraphs])
        
        if not content_text or len(content_text) < 100:
            content_text = soup.get_text()
        
        content_text = ' '.join(content_text.split())
        
        return content_text, True
    
    except Exception as e:
        return f"Error extracting content from {url}: {str(e)}", False

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'preset_question' not in st.session_state:
    st.session_state.preset_question = ""
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
if 'processed_urls' not in st.session_state:
    st.session_state.processed_urls = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'url_status' not in st.session_state:
    st.session_state.url_status = {}

# Main header
st.markdown('<div class="main-header"><h1>🤖 RockyBot: Interactive News Research Tool</h1></div>', unsafe_allow_html=True)

# Create columns for better layout
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📊 Dashboard")
    
    # Progress metrics
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        st.metric("Queries Used", f"{st.session_state.query_count}", delta=f"{50 - st.session_state.query_count} remaining")
    with col1_2:
        st.metric("URLs Processed", len(st.session_state.processed_urls))
    
    # Progress bar for API usage
    progress_percentage = min(st.session_state.query_count / 50, 1.0)
    st.progress(progress_percentage)
    
    if progress_percentage > 0.8:
        st.warning("⚠️ API usage high!")
    elif progress_percentage >= 1.0:
        st.error("🚫 Daily limit reached!")
    
    # URL Status visualization
    if st.session_state.url_status:
        status_data = pd.DataFrame([
            {"Status": "Success", "Count": sum(1 for v in st.session_state.url_status.values() if v)},
            {"Status": "Failed", "Count": sum(1 for v in st.session_state.url_status.values() if not v)}
        ])
        
        if status_data['Count'].sum() > 0:
            fig = px.pie(status_data, values='Count', names='Status', 
                        title="URL Processing Status",
                        color_discrete_map={'Success': '#4CAF50', 'Failed': '#F44336'})
            st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 🔗 URL Processing")
    
    # Enhanced URL input with real-time validation
    urls = []
    url_containers = []
    
    for i in range(3):
        with st.container():
            col2_1, col2_2 = st.columns([4, 1])
            with col2_1:
                url = st.text_input(f"URL {i+1}", key=f"url_{i}", placeholder="Enter a news article URL...")
                if url:
                    urls.append(url)
                    # Basic URL validation
                    if not url.startswith(('http://', 'https://')):
                        st.error("❌ Invalid URL format. Please include http:// or https://")
                    else:
                        st.success("✅ Valid URL format")
            
            with col2_2:
                if url:
                    if st.button(f"Test {i+1}", key=f"test_{i}"):
                        with st.spinner("Testing..."):
                            try:
                                response = requests.head(url, timeout=5)
                                if response.status_code == 200:
                                    st.success("✅ OK")
                                else:
                                    st.error(f"❌ {response.status_code}")
                            except:
                                st.error("❌ Failed")

    # Processing button with enhanced feedback
    if st.button("🚀 Process URLs", type="primary"):
        if urls:
            # Create a progress container
            progress_container = st.container()
            with progress_container:
                st.markdown("### 🔄 Processing Status")
                
                # Initialize progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                documents = []
                total_urls = len(urls)
                
                for i, url in enumerate(urls):
                    # Update progress
                    progress = (i + 1) / total_urls
                    progress_bar.progress(progress)
                    status_text.text(f"Processing URL {i+1}/{total_urls}: {url[:50]}...")
                    
                    # Process URL
                    try:
                        # Try WebBaseLoader first
                        loader = WebBaseLoader(
                            [url],
                            header_template={
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                            }
                        )
                        docs = loader.load()
                        content = docs[0].page_content if docs else ""
                        success = True
                    except:
                        content = ""
                        success = False
                    
                    # If WebBaseLoader failed, try alternative method
                    if not content or len(content.strip()) < 100:
                        content, success = extract_content_with_requests(url)
                    
                    # Store URL status
                    st.session_state.url_status[url] = success and len(content.strip()) > 50
                    
                    # Create document if successful
                    if success and len(content.strip()) > 50:
                        doc = Document(page_content=content, metadata={"source": url})
                        documents.append(doc)
                        st.session_state.processed_urls.append(url)
                        
                        # Show real-time success message
                        st.success(f"✅ URL {i+1} processed successfully ({len(content)} characters)")
                    else:
                        st.error(f"❌ Failed to process URL {i+1}")
                
                # Final processing
                if documents:
                    status_text.text("📚 Creating text chunks...")
                    text_splitter = RecursiveCharacterTextSplitter(
                        separators=['\n\n', '\n', '.', ','],
                        chunk_size=1000,
                        chunk_overlap=100
                    )
                    docs = text_splitter.split_documents(documents)
                    
                    status_text.text("🧠 Generating embeddings...")
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Processing complete!")
                    
                    # Success summary
                    st.balloons()
                    st.markdown(f"""
                    <div class="success-message">
                        <h3>🎉 Processing Complete!</h3>
                        <p>Successfully processed {len(documents)} articles with {len(docs)} text chunks.</p>
                        <p>You can now ask questions about the content!</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("❌ No content could be extracted from any URLs.")
        else:
            st.warning("⚠️ Please enter at least one URL to process.")

# Chat Interface
st.markdown("### 💬 Ask Questions")

# Chat history display
if st.session_state.chat_history:
    st.markdown("#### Previous Conversations:")
    for i, (question, answer) in enumerate(st.session_state.chat_history[-3:]):  # Show last 3 conversations
        st.markdown(f"""
        <div class="chat-bubble">
            <strong>Q{len(st.session_state.chat_history) - 2 + i}:</strong> {question}
        </div>
        <div class="answer-bubble">
            <strong>A:</strong> {answer[:200]}{'...' if len(answer) > 200 else ''}
        </div>
        """, unsafe_allow_html=True)

# Question input with suggestions
col3, col4 = st.columns([3, 1])

with col3:
    query = st.text_input("Ask a question about the articles:",
                          value=st.session_state.preset_question,
                          placeholder="e.g., What are the main topics discussed?",
                          key="question_input")



with col4:
    if st.button("🎲 Random Question"):
        sample_questions = [
            "What are the main topics discussed in these articles?",
            "Summarize the key findings from these news articles",
            "What are the different perspectives mentioned?",
            "Are there any conflicting viewpoints in these articles?",
            "What are the implications of the events described?"
        ]
        import random
        st.session_state.preset_question = random.choice(sample_questions)
        st.rerun()


# Quick question buttons
st.markdown("#### Quick Questions:")
quick_questions = [
    "Summarize the articles",
    "What are the key points?",
    "Any conflicting views?",
    "What's the timeline?",
    "Who are the key people mentioned?"
]

cols = st.columns(len(quick_questions))
for i, question in enumerate(quick_questions):
    with cols[i]:
        if st.button(question, key=f"quick_{i}"):
            st.session_state.preset_question = question
            st.rerun()


# Process query
if query:
    if st.session_state.vectorstore is not None:
        if st.session_state.query_count < 50:
            try:
                with st.spinner("🤔 Analyzing content..."):
                    # Show thinking animation
                    thinking_placeholder = st.empty()
                    for i in range(3):
                        thinking_placeholder.text(f"Processing{'.' * (i + 1)}")
                        time.sleep(0.5)
                    thinking_placeholder.empty()
                    
                    chain = RetrievalQAWithSourcesChain.from_llm(
                        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.9, max_tokens=500),
                        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                    )
                    
                    result = chain({"question": query}, return_only_outputs=True)
                    st.session_state.query_count += 1
                    
                    # Store in chat history
                    st.session_state.chat_history.append((query, result["answer"]))

                    # ✅ Reset the preset_question input
                    st.session_state.preset_question = ""
                    
                    # Display answer with enhanced formatting
                    st.markdown("### 📌 Answer")
                    st.markdown(f"""
                    <div class="answer-bubble">
                        {result["answer"]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Enhanced sources display
                    sources = result.get("sources", "")
                    if sources:
                        st.markdown("### 🔗 Sources")
                        source_list = [s.strip() for s in sources.split("\n") if s.strip()]
                        for j, source in enumerate(source_list, 1):
                            st.markdown(f"**{j}.** {source}")
                    
                    # Feedback buttons
                    col5, col6, col7 = st.columns([1, 1, 4])
                    with col5:
                        if st.button("👍 Helpful"):
                            st.success("Thank you for your feedback!")
                    with col6:
                        if st.button("👎 Not helpful"):
                            st.info("We'll work on improving our responses!")
                    
                    # Follow-up suggestions
                    st.markdown("#### 🔄 Follow-up Questions:")
                    followup_questions = [
                        "Can you elaborate on this?",
                        "What are the implications?",
                        "Are there any counterarguments?",
                        "What's the broader context?"
                    ]
                    
                    follow_cols = st.columns(len(followup_questions))
                    for k, fq in enumerate(followup_questions):
                        with follow_cols[k]:
                            if st.button(fq, key=f"followup_{k}"):
                                st.session_state.preset_question = fq
                                st.rerun()

                    
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    st.error("🚫 **API Quota Exceeded**: Daily limit reached!")
                    st.info("💡 Try again in 24 hours or upgrade your API plan.")
                else:
                    st.error(f"❌ Error: {error_msg}")
        else:
            st.error("🚫 Daily query limit reached! Please wait 24 hours.")
    else:
        st.warning("⚠️ Please process some URLs first!")

# Sidebar enhancements
with st.sidebar:
    st.markdown("### 🔧 Settings")
    
    # Temperature control
    temperature = st.slider("AI Temperature", 0.0, 1.0, 0.9, 0.1)
    
    # Max tokens
    max_tokens = st.slider("Max Response Length", 100, 1000, 500, 50)
    
    # Clear data button
    if st.button("🗑️ Clear All Data"):
        st.session_state.vectorstore = None
        st.session_state.processed_urls = []
        st.session_state.chat_history = []
        st.session_state.url_status = {}
        st.success("All data cleared!")
        st.rerun()
    
    # Export chat history
    if st.session_state.chat_history:
        chat_export = json.dumps(st.session_state.chat_history, indent=2)
        st.download_button(
            label="📥 Export Chat History",
            data=chat_export,
            file_name=f"rockybot_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    # Statistics
    st.markdown("### 📈 Statistics")
    st.write(f"Total questions asked: {len(st.session_state.chat_history)}")
    st.write(f"URLs processed: {len(st.session_state.processed_urls)}")
    st.write(f"Success rate: {(sum(st.session_state.url_status.values()) / len(st.session_state.url_status) * 100):.1f}%" if st.session_state.url_status else "N/A")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🤖 RockyBot - Enhanced Interactive News Research Tool</p>
    <p>Powered by Google Gemini AI | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)