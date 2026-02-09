# 🤖 Data Analysis Agent with LLM

An autonomous AI agent that analyzes datasets and answers questions in natural language using LangChain and pandas.

## 🎯 Project Overview

This agent demonstrates AI-powered data analysis:
- Ask questions about your data in plain English
- Get statistical insights automatically
- Generate visualizations on command
- Autonomous decision-making using ReAct pattern

## 🛠️ Tech Stack

- **LangChain**: Agent framework
- **OpenAI**: LLM reasoning
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization

## 📋 Features

✅ Natural language data queries  
✅ Automated statistical analysis  
✅ Visualization generation  
✅ Correlation analysis  
✅ Missing data detection  
✅ Multi-column operations  

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python data_agent.py
```

## 💡 Example Queries

```python
from data_agent import DataAnalysisAgent

agent = DataAnalysisAgent(api_key="key", csv_path="data.csv")

# Natural language questions
agent.query("What is the average sales by region?")
agent.query("Which product has the highest profit margin?")
agent.query("Show me the top 5 customers by revenue")
agent.query("What percentage of orders were delivered late?")
```

## 📊 Agent Capabilities

**Statistical Analysis:**
- Averages, medians, standard deviations
- Counts and percentages
- Group-by operations
- Trend identification

**Data Exploration:**
- Column types and ranges
- Missing value detection
- Unique value counts
- Data shape and structure

**Visualizations:**
```python
# Generate histogram
agent.visualize_column('Sales', kind='hist')

# Create correlation heatmap
agent.correlation_analysis()
```

## 🧠 How It Works

```
User Question: "What's the average sales?"
        │
        ▼
┌──────────────────┐
│   LLM Reasoning  │  "I need to calculate mean of Sales column"
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Pandas Tool     │  df['Sales'].mean()
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Result: 15,234  │
└──────────────────┘
```

The agent uses **ReAct** (Reasoning + Acting):
1. **Reason**: Understand what operation is needed
2. **Act**: Execute pandas code
3. **Observe**: Check result
4. **Repeat** if needed

## 🎓 Key Concepts

- **Agent**: Autonomous decision-maker
- **Tools**: Pandas functions the agent can use
- **ReAct**: Reasoning pattern for LLMs
- **Zero-shot**: No examples needed

## 📁 File Structure

```
3-data-agent/
├── data_agent.py        # Main implementation
├── README.md            # This file
├── requirements.txt     # Dependencies
└── sample_data.csv      # Example dataset
```

## ⚠️ Important Note

This agent executes code dynamically. Only use with trusted data sources.

## 👤 Author

Moisés Franco-Villegas - [LinkedIn](https://linkedin.com/in/mfvps)

---

*Part of AI/ML Engineering Portfolio*
