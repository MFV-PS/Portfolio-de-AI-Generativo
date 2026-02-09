# 🚀 GenAI Portfolio

AI/ML Engineering portfolio showcasing practical implementations of Large Language Model applications using LangChain and OpenAI.

**Author:** Moisés Franco-Villegas  
**LinkedIn:** [linkedin.com/in/mfvps](https://linkedin.com/in/mfvps)  
**GitHub:** [github.com/MFV-PS](https://github.com/MFV-PS)

---

## 📂 Projects Overview

### 1. 📚 RAG System for Technical Documents
**Technologies:** LangChain, FAISS, OpenAI Embeddings

A Retrieval-Augmented Generation system that enables semantic search and Q&A over PDF documents.

**Key Features:**
- PDF document ingestion
- Vector embeddings with FAISS
- Semantic search and retrieval
- Context-aware answers with source attribution

[View Project →](./1-rag-system/)

---

### 2. 💬 Conversational Chatbot with Memory
**Technologies:** LangChain, OpenAI GPT, Conversation Memory

An intelligent chatbot that maintains context across multiple conversation turns.

**Key Features:**
- Conversation memory (buffer and summary modes)
- Multiple personality templates
- Save/load conversation histories
- Conversation analytics

[View Project →](./2-chatbot-memory/)

---

### 3. 🤖 Data Analysis Agent
**Technologies:** LangChain Agents, Pandas, OpenAI

An autonomous agent that analyzes datasets and generates insights using natural language queries.

**Key Features:**
- Natural language data queries
- Automated statistical analysis
- Visualization generation
- ReAct reasoning pattern

[View Project →](./3-data-agent/)

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Frameworks** | LangChain, OpenAI API |
| **Data** | Pandas, FAISS |
| **Visualization** | Matplotlib, Seaborn |
| **Languages** | Python 3.8+ |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

```bash
# Clone the repository
git clone https://github.com/MFV-PS/genai-portfolio.git
cd genai-portfolio

# Choose a project
cd 1-rag-system  # or 2-chatbot-memory or 3-data-agent

# Install dependencies
pip install -r requirements.txt

# Run the project
python main_file.py  # See each project's README for specific instructions
```

### Setting Up API Key

```bash
# Option 1: Environment variable
export OPENAI_API_KEY="your-key-here"

# Option 2: .env file (recommended)
echo "OPENAI_API_KEY=your-key-here" > .env
```

---

## 📊 What I Learned

Through building these projects, I gained hands-on experience with:

✅ **LangChain Framework**
- Chains, Agents, and Memory components
- Prompt engineering and template design
- Integration with LLM APIs

✅ **Retrieval-Augmented Generation (RAG)**
- Document embedding and chunking strategies
- Vector databases (FAISS)
- Semantic search and retrieval

✅ **Agent-Based Systems**
- ReAct reasoning pattern
- Tool integration and function calling
- Autonomous decision-making

✅ **Production Considerations**
- Error handling and edge cases
- API cost optimization
- Code modularity and reusability

---

## 🎯 Use Cases

These projects demonstrate capabilities applicable to:

- **Enterprise Knowledge Management**: RAG for internal documentation
- **Customer Support**: Conversational AI for automated assistance
- **Business Intelligence**: Natural language data analysis
- **Research Tools**: Q&A systems for scientific papers
- **Personal Assistants**: Context-aware AI helpers

---

## 📈 Project Stats

| Project | Lines of Code | Key Concepts | Complexity |
|---------|--------------|--------------|------------|
| RAG System | ~300 | Embeddings, Retrieval | Medium |
| Chatbot | ~250 | Memory, Prompts | Low-Medium |
| Data Agent | ~280 | Agents, Tools | Medium-High |

---

## 🔮 Future Enhancements

- [ ] Web interfaces using Streamlit/Gradio
- [ ] Support for additional LLM providers (Anthropic, Cohere)
- [ ] Multi-agent collaboration systems
- [ ] Integration with cloud vector databases (Pinecone, Weaviate)
- [ ] Fine-tuning pipelines for domain-specific tasks

---

## 📝 Documentation

Each project includes:
- ✅ Detailed README with examples
- ✅ Well-commented code
- ✅ Requirements file
- ✅ Usage instructions
- ✅ Troubleshooting tips

---

## 🤝 Contributing

This is a personal portfolio, but feedback and suggestions are welcome!

Feel free to:
- Open issues for questions
- Suggest improvements
- Share your own implementations

---

## 📄 License

MIT License - feel free to use these projects for learning and reference.

---

## 📧 Contact

**Moisés Franco-Villegas**
- **Email:** moi.14896@outlook.com
- **LinkedIn:** [linkedin.com/in/mfvps](https://linkedin.com/in/mfvps)
- **GitHub:** [github.com/MFV-PS](https://github.com/MFV-PS)

---

## 🙏 Acknowledgments

- **LangChain** team for excellent documentation
- **OpenAI** for accessible LLM APIs
- **Python community** for amazing libraries

---

*Built with ❤️ as part of my journey into AI/ML engineering*

**Last Updated:** January 2025
