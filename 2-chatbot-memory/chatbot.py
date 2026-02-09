"""
Conversational AI Chatbot with Memory
Built with LangChain - Maintains context across conversations

Author: Moisés Franco-Villegas
Portfolio Project for AI Engineer Position
"""

from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from datetime import datetime
import json

class ConversationalChatbot:
    """
    Chatbot with conversation memory
    
    Features:
    - Remembers conversation history
    - Maintains context across multiple turns
    - Custom personality via prompt templates
    - Save/load conversation history
    """
    
    def __init__(self, api_key: str, personality: str = "helpful", memory_type: str = "buffer"):
        """
        Initialize chatbot
        
        Args:
            api_key: OpenAI API key
            personality: Chatbot personality ('helpful', 'technical', 'friendly')
            memory_type: Type of memory ('buffer' or 'summary')
        """
        self.llm = OpenAI(
            api_key=api_key,
            temperature=0.7,
            model_name="gpt-3.5-turbo-instruct"
        )
        
        # Setup memory
        if memory_type == "buffer":
            self.memory = ConversationBufferMemory()
        else:
            self.memory = ConversationSummaryMemory(llm=self.llm)
        
        # Setup prompt based on personality
        self.prompt = self._create_prompt(personality)
        
        # Create conversation chain
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self.prompt,
            verbose=False
        )
        
        self.conversation_log = []
        
    def _create_prompt(self, personality: str) -> PromptTemplate:
        """Create custom prompt template based on personality"""
        
        personalities = {
            "helpful": """You are a helpful AI assistant. You provide clear, accurate, 
            and detailed responses. You remember previous parts of the conversation 
            and use that context to give better answers.
            
            Current conversation:
            {history}
            
            Human: {input}
            AI:""",
            
            "technical": """You are a technical AI assistant specialized in software 
            engineering, data science, and AI/ML. You provide precise, code-focused 
            responses with examples when appropriate. You remember technical details 
            from earlier in the conversation.
            
            Current conversation:
            {history}
            
            Human: {input}
            AI:""",
            
            "friendly": """You are a friendly and conversational AI assistant. You use 
            a warm, approachable tone while still being informative and helpful. You 
            remember personal details shared in the conversation.
            
            Current conversation:
            {history}
            
            Human: {input}
            AI:"""
        }
        
        template = personalities.get(personality, personalities["helpful"])
        
        return PromptTemplate(
            input_variables=["history", "input"],
            template=template
        )
    
    def chat(self, user_input: str) -> str:
        """
        Send message to chatbot and get response
        
        Args:
            user_input: User's message
            
        Returns:
            str: Chatbot's response
        """
        response = self.conversation.predict(input=user_input)
        
        # Log conversation
        self.conversation_log.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "bot": response
        })
        
        return response
    
    def get_conversation_history(self) -> str:
        """Get formatted conversation history"""
        return self.memory.load_memory_variables({})["history"]
    
    def save_conversation(self, filepath: str = "conversation_history.json"):
        """Save conversation log to file"""
        with open(filepath, 'w') as f:
            json.dump(self.conversation_log, f, indent=2)
        print(f"Conversation saved to {filepath}")
    
    def load_conversation(self, filepath: str = "conversation_history.json"):
        """Load conversation log from file"""
        try:
            with open(filepath, 'r') as f:
                self.conversation_log = json.load(f)
            print(f"Conversation loaded from {filepath}")
            
            # Rebuild memory from log
            for entry in self.conversation_log:
                self.memory.save_context(
                    {"input": entry["user"]},
                    {"output": entry["bot"]}
                )
        except FileNotFoundError:
            print(f"File {filepath} not found")
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        self.conversation_log = []
        print("Memory cleared")
    
    def get_stats(self) -> dict:
        """Get conversation statistics"""
        return {
            "total_messages": len(self.conversation_log),
            "user_messages": len([e for e in self.conversation_log]),
            "conversation_start": self.conversation_log[0]["timestamp"] if self.conversation_log else None,
            "last_message": self.conversation_log[-1]["timestamp"] if self.conversation_log else None
        }


def interactive_demo():
    """Interactive demo of the chatbot"""
    
    print("="*60)
    print("CONVERSATIONAL AI CHATBOT - INTERACTIVE DEMO")
    print("="*60)
    
    # Get API key
    api_key = input("\nEnter your OpenAI API key: ")
    
    # Choose personality
    print("\nChoose chatbot personality:")
    print("1. Helpful (default)")
    print("2. Technical")
    print("3. Friendly")
    
    choice = input("Enter choice (1-3): ").strip() or "1"
    personalities = {"1": "helpful", "2": "technical", "3": "friendly"}
    personality = personalities.get(choice, "helpful")
    
    # Initialize chatbot
    print(f"\n✓ Initializing {personality} chatbot...")
    bot = ConversationalChatbot(api_key=api_key, personality=personality)
    
    print("\n" + "="*60)
    print("Chatbot ready! Type your messages below.")
    print("Commands: 'history' = show history, 'clear' = clear memory, 'quit' = exit")
    print("="*60 + "\n")
    
    # Conversation loop
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() == 'quit':
            # Save conversation before exiting
            save = input("\nSave conversation? (y/n): ").lower()
            if save == 'y':
                bot.save_conversation()
            
            stats = bot.get_stats()
            print(f"\nConversation stats:")
            print(f"  Total messages: {stats['total_messages']}")
            print(f"  Duration: {stats['conversation_start']} to {stats['last_message']}")
            print("\nGoodbye!")
            break
        
        elif user_input.lower() == 'history':
            print("\n--- Conversation History ---")
            print(bot.get_conversation_history())
            print("----------------------------\n")
            continue
        
        elif user_input.lower() == 'clear':
            bot.clear_memory()
            print("✓ Memory cleared\n")
            continue
        
        # Get chatbot response
        try:
            response = bot.chat(user_input)
            print(f"\nBot: {response}\n")
        except Exception as e:
            print(f"\n✗ Error: {e}\n")


def automated_demo():
    """Automated demo showing chatbot capabilities"""
    
    print("="*60)
    print("CHATBOT DEMO - Automated Conversation Example")
    print("="*60 + "\n")
    
    # Simulated conversation
    conversation = [
        "Hi! My name is Alex.",
        "I'm learning about machine learning.",
        "What are the main types of ML algorithms?",
        "Which one should I start with as a beginner?",
        "Do you remember my name?"  # Tests memory
    ]
    
    # Initialize (you would use real API key here)
    api_key = "your-api-key-here"
    bot = ConversationalChatbot(api_key=api_key, personality="technical")
    
    # Run conversation
    for user_msg in conversation:
        print(f"User: {user_msg}")
        response = bot.chat(user_msg)
        print(f"Bot: {response}\n")
        print("-"*60 + "\n")
    
    # Show memory works
    print("="*60)
    print("CONVERSATION HISTORY:")
    print("="*60)
    print(bot.get_conversation_history())
    
    # Show stats
    print("\n" + "="*60)
    print("STATISTICS:")
    print("="*60)
    stats = bot.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    # Run interactive demo
    interactive_demo()
    
    # Uncomment to run automated demo instead:
    # automated_demo()
