# 🧠 RockyBot: Research Tool (AI-Powered Q&A from Web URLs using Streamlit + Gemini)

**RockyBot** is an AI-powered research assistant that takes web page URLs (news articles, Wikipedia entries, blogs, etc.) and enables users to ask intelligent questions about the content. It uses Google’s Gemini language model, LangChain, FAISS for vector search, and a Streamlit-based interface to deliver contextual answers along with source references.

---

## 📂 Dataset Used

- **Unstructured Web Content** — Loaded from user-provided URLs (e.g., news sites, Wikipedia, blogs) using `UnstructuredURLLoader`.

---

## 📊 Key Features

- **Web-Based Research**: Input up to 3 URLs from sources like news websites, Wikipedia, or blogs.
- **Text Chunking**: Splits large documents into chunks for better processing.
- **AI Q&A**: Ask questions and get intelligent, context-aware answers.
- **Source Attribution**: Each response includes source references for transparency.
- **Streamlit Interface**: Clean, user-friendly interface for fast interaction.

---

## 🛠️ Tools and Technologies Used

| Technology               | Purpose                                       |
|--------------------------|-----------------------------------------------|
| **Streamlit**            | UI development for interactive web app       |
| **LangChain**            | LLM orchestration and retrieval management   |
| **Google Gemini**        | LLM for answering queries contextually       |
| **Google AI Embeddings** | Converts text to embeddings for FAISS search |
| **FAISS**                | Efficient vector similarity search           |
| **Python**               | Core backend logic and orchestration         |
| **dotenv**               | Secure handling of API keys and configs      |

---

## 🔄 Process Flow

### 1. **Data Collection**
- User provides URLs from any supported web source (news, Wikipedia, blogs, etc.).

### 2. **Data Extraction**
- Content is loaded and cleaned using `UnstructuredURLLoader`.

### 3. **Text Splitting**
- Content is chunked using `RecursiveCharacterTextSplitter` for optimal embedding.

### 4. **Embedding and Indexing**
- Chunks are embedded with **Google Generative AI Embeddings** and indexed in **FAISS**.

### 5. **Q&A Retrieval**
- User asks a question; the tool retrieves relevant chunks and uses Gemini to generate the answer.

### 6. **Result Display**
- The answer and relevant source(s) are shown in the Streamlit UI.

---

## 💡 Use Case Scenarios

- Summarize or explore news articles.
- Research Wikipedia topics more efficiently.
- Analyze and compare blogs or opinion pieces.
- Academic research across online sources.

---

## 📸 Screenshots

| Feature        | Screenshot |
|----------------|------------|
| URL Input & Querying | ![URL Input](screenshots/url_input.png) |
| Answer with Sources  | ![Answer](screenshots/answer_output.png) |

---

## 🚀 Future Enhancements

- Support for PDF/Doc upload.
- Conversational memory for multi-turn interactions.
- Voice input and multi-language support.
- Exporting answers and citations to PDF or Markdown.
