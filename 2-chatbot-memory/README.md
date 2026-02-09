# 💬 Conversational AI Chatbot with Memory

An intelligent chatbot built with LangChain that maintains conversation context and remembers previous interactions.

## 🎯 Project Overview

This chatbot demonstrates advanced conversational AI capabilities:
- **Contextual awareness**: Remembers earlier parts of the conversation
- **Multiple personalities**: Choose between helpful, technical, or friendly modes
- **Persistent memory**: Save and load conversation histories
- **Conversation analytics**: Track message counts and timestamps

## 🛠️ Tech Stack

- **LangChain**: Conversation management
- **OpenAI GPT**: Language model
- **Python 3.8+**: Implementation

## 📋 Features

✅ Conversation memory  
✅ Custom personalities  
✅ Context-aware responses  
✅ Save/load conversations  
✅ Statistics tracking  
✅ Interactive CLI  

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python chatbot.py
```

## 💡 Usage

```python
from chatbot import ConversationalChatbot

bot = ConversationalChatbot(api_key="key", personality="helpful")

bot.chat("Hi! My name is Sarah.")
bot.chat("What's my name?")  # Bot remembers!
```

## 📊 Demo Conversation

```
You: Hi! My name is Alex.
Bot: Hello Alex! Nice to meet you...

You: I'm learning Python
Bot: That's great, Alex! Python is...

You: What was I learning?
Bot: You mentioned you're learning Python.
```

✓ Memory working correctly!

## 🎨 Personalities

- **helpful**: Clear, detailed responses
- **technical**: Code-focused, engineering
- **friendly**: Warm, conversational

## 📁 File Structure

```
2-chatbot-memory/
├── chatbot.py           # Main implementation
├── README.md            # This file
├── requirements.txt     # Dependencies
└── demo.py              # Example usage
```

## 👤 Author

Moisés Franco-Villegas - [LinkedIn](https://linkedin.com/in/mfvps)

---

*Part of AI/ML Engineering Portfolio*
