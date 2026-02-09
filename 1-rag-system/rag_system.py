"""
RAG System for Technical Documents
A Retrieval-Augmented Generation system using LangChain and FAISS

Author: Moisés Franco-Villegas
Portfolio Project for AI Engineer Position
"""

import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

class RAGSystem:
    """
    RAG (Retrieval-Augmented Generation) System
    
    This system allows users to:
    1. Load PDF documents
    2. Create embeddings and vector store
    3. Query documents using natural language
    4. Get contextually relevant answers
    """
    
    def __init__(self, api_key: str, docs_path: str = "./documents"):
        """
        Initialize RAG System
        
        Args:
            api_key: OpenAI API key
            docs_path: Path to documents directory
        """
        os.environ["OPENAI_API_KEY"] = api_key
        self.docs_path = docs_path
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.qa_chain = None
        
    def load_documents(self):
        """Load PDF documents from specified directory"""
        print(f"Loading documents from {self.docs_path}...")
        
        # Load PDFs
        loader = DirectoryLoader(
            self.docs_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        
        print(f"Loaded {len(documents)} document pages")
        return documents
    
    def create_vectorstore(self, documents):
        """
        Create vector store from documents
        
        Args:
            documents: List of loaded documents
        """
        print("Splitting documents into chunks...")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        print(f"Created {len(chunks)} text chunks")
        print("Creating embeddings and vector store...")
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        print("Vector store created successfully!")
        
    def setup_qa_chain(self):
        """Setup Question-Answering chain"""
        if self.vectorstore is None:
            raise ValueError("Vector store not created. Run create_vectorstore first.")
        
        # Custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer in a clear and concise way:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0),
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        print("QA chain setup complete!")
    
    def query(self, question: str):
        """
        Query the RAG system
        
        Args:
            question: Natural language question
            
        Returns:
            dict: Answer and source documents
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not setup. Run setup_qa_chain first.")
        
        result = self.qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }
    
    def save_vectorstore(self, path: str = "./vectorstore"):
        """Save vector store to disk"""
        if self.vectorstore is None:
            raise ValueError("No vector store to save")
        
        self.vectorstore.save_local(path)
        print(f"Vector store saved to {path}")
    
    def load_vectorstore(self, path: str = "./vectorstore"):
        """Load vector store from disk"""
        self.vectorstore = FAISS.load_local(path, self.embeddings)
        print(f"Vector store loaded from {path}")


def main():
    """Example usage of RAG System"""
    
    # Initialize system (replace with your API key)
    api_key = "your-openai-api-key-here"
    rag = RAGSystem(api_key=api_key, docs_path="./sample_documents")
    
    # Load and process documents
    documents = rag.load_documents()
    rag.create_vectorstore(documents)
    rag.setup_qa_chain()
    
    # Example queries
    questions = [
        "What are the main topics discussed in these documents?",
        "Can you summarize the key findings?",
        "What methodologies are mentioned?"
    ]
    
    print("\n" + "="*60)
    print("QUERYING RAG SYSTEM")
    print("="*60 + "\n")
    
    for question in questions:
        print(f"Q: {question}")
        result = rag.query(question)
        print(f"A: {result['answer']}\n")
        print(f"Sources: {len(result['source_documents'])} documents referenced\n")
        print("-"*60 + "\n")
    
    # Save vector store for later use
    rag.save_vectorstore()


if __name__ == "__main__":
    main()
