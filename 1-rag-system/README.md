# 📚 RAG System for Technical Documents

A Retrieval-Augmented Generation (RAG) system built with LangChain and FAISS for semantic search and question-answering over technical documents.

## 🎯 Project Overview

This project demonstrates the implementation of a production-ready RAG system that:
- Loads and processes PDF documents
- Creates semantic embeddings using OpenAI
- Stores vectors in FAISS for efficient retrieval
- Answers natural language questions with context from documents
- Provides source attribution for transparency

## 🛠️ Tech Stack

- **LangChain**: Framework for LLM applications
- **FAISS**: Vector database for similarity search
- **OpenAI**: Embeddings and LLM
- **Python 3.8+**: Core implementation

## 📋 Features

✅ PDF document ingestion  
✅ Semantic chunking with overlap  
✅ Vector embeddings with OpenAI  
✅ FAISS vector store for fast retrieval  
✅ Custom prompt templates  
✅ Source document attribution  
✅ Persistent vector store (save/load)  

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
OpenAI API key
```

### Installation

```bash
# Clone the repository
git clone https://github.com/MFV-PS/genai-portfolio.git
cd genai-portfolio/1-rag-system

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Set your OpenAI API key:
```python
# In rag_system.py
api_key = "your-openai-api-key-here"
```

Or use environment variable:
```bash
export OPENAI_API_KEY="your-key-here"
```

2. Add PDF documents to `./sample_documents/` directory

### Usage

Basic usage:
```python
from rag_system import RAGSystem

# Initialize system
rag = RAGSystem(api_key="your-key", docs_path="./documents")

# Load and process documents
documents = rag.load_documents()
rag.create_vectorstore(documents)
rag.setup_qa_chain()

# Query the system
result = rag.query("What are the main findings?")
print(result['answer'])
```

Run the example:
```bash
python rag_system.py
```

## 📊 How It Works

```
┌─────────────┐
│   PDFs      │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Text Splitter  │  (Chunks: 1000 chars, overlap: 200)
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│  OpenAI Embed.   │  (Convert text → vectors)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  FAISS Store     │  (Store vectors for retrieval)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  User Query      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Retrieval       │  (Find top-k relevant chunks)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  LLM + Context   │  (Generate answer)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Answer + Sources│
└──────────────────┘
```

## 🎓 Key Concepts Demonstrated

1. **Document Loading**: Handling multiple PDFs from directory
2. **Text Chunking**: Splitting documents with optimal size and overlap
3. **Embeddings**: Converting text to semantic vectors
4. **Vector Store**: FAISS for efficient similarity search
5. **Retrieval**: Finding relevant context for queries
6. **Chain Construction**: Building QA pipeline with LangChain
7. **Prompt Engineering**: Custom templates for better responses

## 📈 Performance

- **Chunk size**: 1000 characters (optimized for technical content)
- **Overlap**: 200 characters (maintains context continuity)
- **Retrieval**: Top-3 most relevant chunks per query
- **Response time**: ~2-3 seconds per query

## 🔧 Customization

### Adjust chunk size:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Increase for longer context
    chunk_overlap=300
)
```

### Change retrieval parameters:
```python
retriever=self.vectorstore.as_retriever(
    search_kwargs={"k": 5}  # Retrieve top-5 instead of top-3
)
```

### Modify prompt template:
```python
prompt_template = """Custom instructions here...
Context: {context}
Question: {question}
Answer:"""
```

## 📝 Example Output

```
Q: What methodologies are mentioned in the documents?
A: The documents discuss several methodologies including finite element 
   analysis, numerical simulation techniques, and statistical modeling 
   approaches for analyzing porous media systems.

Sources: 3 documents referenced
```

## 🔐 Security Note

- Never commit your OpenAI API key to version control
- Use environment variables or `.env` files (add to `.gitignore`)
- Monitor API usage to avoid unexpected costs

## 🚧 Future Enhancements

- [ ] Support for multiple document formats (DOCX, TXT, Markdown)
- [ ] Web interface using Streamlit
- [ ] Multiple vector store backends (Pinecone, Chroma)
- [ ] Hybrid search (keyword + semantic)
- [ ] Conversation memory for follow-up questions

## 📄 License

MIT License - feel free to use this code for your projects

## 👤 Author

**Moisés Franco-Villegas**
- LinkedIn: [linkedin.com/in/mfvps](https://linkedin.com/in/mfvps)
- GitHub: [github.com/MFV-PS](https://github.com/MFV-PS)

## 🙏 Acknowledgments

- LangChain documentation and community
- OpenAI for embeddings and LLM APIs
- FAISS team for efficient vector search

---

*Built as part of AI/ML engineering portfolio demonstrating practical GenAI applications*
