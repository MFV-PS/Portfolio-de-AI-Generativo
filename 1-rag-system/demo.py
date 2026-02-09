"""
Simple example demonstrating RAG system usage
Perfect for testing and quick demos
"""

from rag_system import RAGSystem

def quick_demo():
    """Quick demo of RAG system capabilities"""
    
    print("="*60)
    print("RAG SYSTEM - QUICK DEMO")
    print("="*60)
    
    # Step 1: Initialize
    print("\n[1/4] Initializing RAG System...")
    api_key = input("Enter your OpenAI API key: ")
    rag = RAGSystem(api_key=api_key)
    
    # Step 2: Load documents
    print("\n[2/4] Loading documents...")
    try:
        documents = rag.load_documents()
        print(f"✓ Successfully loaded {len(documents)} pages")
    except Exception as e:
        print(f"✗ Error loading documents: {e}")
        print("Make sure you have PDF files in ./documents/ folder")
        return
    
    # Step 3: Create vector store
    print("\n[3/4] Creating vector store (this may take a minute)...")
    rag.create_vectorstore(documents)
    rag.setup_qa_chain()
    print("✓ System ready!")
    
    # Step 4: Interactive queries
    print("\n[4/4] You can now ask questions!")
    print("(Type 'quit' to exit)\n")
    
    while True:
        question = input("Your question: ")
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not question.strip():
            continue
        
        print("\nThinking...")
        result = rag.query(question)
        
        print(f"\n📝 Answer:\n{result['answer']}\n")
        print(f"📚 Based on {len(result['source_documents'])} source documents\n")
        print("-"*60 + "\n")

if __name__ == "__main__":
    quick_demo()
