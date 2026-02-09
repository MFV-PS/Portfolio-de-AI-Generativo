"""
Data Analysis Agent with LLM
An autonomous agent that analyzes datasets using pandas and generates insights

Author: Moisés Franco-Villegas
Portfolio Project for AI Engineer Position
"""

from langchain.agents import create_pandas_dataframe_agent, AgentType
from langchain.llms import OpenAI
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class DataAnalysisAgent:
    """
    AI Agent for automated data analysis
    
    Capabilities:
    - Load and explore datasets
    - Answer natural language questions about data
    - Generate statistical summaries
    - Create visualizations
    - Provide insights and recommendations
    """
    
    def __init__(self, api_key: str, csv_path: str = None):
        """
        Initialize Data Analysis Agent
        
        Args:
            api_key: OpenAI API key
            csv_path: Optional path to CSV file
        """
        self.llm = OpenAI(
            api_key=api_key,
            temperature=0,  # Deterministic for data analysis
            model_name="gpt-3.5-turbo-instruct"
        )
        
        self.df = None
        self.agent = None
        
        if csv_path:
            self.load_data(csv_path)
    
    def load_data(self, csv_path: str):
        """Load CSV data and create agent"""
        try:
            self.df = pd.read_csv(csv_path)
            print(f"✓ Loaded dataset: {csv_path}")
            print(f"  Shape: {self.df.shape}")
            print(f"  Columns: {list(self.df.columns)}")
            
            # Create pandas dataframe agent
            self.agent = create_pandas_dataframe_agent(
                self.llm,
                self.df,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                allow_dangerous_code=True  # Required for pandas operations
            )
            
            print("✓ Agent ready for analysis")
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
    
    def query(self, question: str) -> str:
        """
        Ask agent a question about the data
        
        Args:
            question: Natural language question
            
        Returns:
            str: Agent's answer
        """
        if self.agent is None:
            return "Error: No data loaded. Use load_data() first."
        
        try:
            response = self.agent.run(question)
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_summary(self) -> dict:
        """Get comprehensive dataset summary"""
        if self.df is None:
            return {"error": "No data loaded"}
        
        summary = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "numeric_summary": self.df.describe().to_dict() if len(self.df.select_dtypes(include='number').columns) > 0 else None
        }
        
        return summary
    
    def visualize_column(self, column: str, kind: str = 'auto'):
        """
        Create visualization for a column
        
        Args:
            column: Column name
            kind: Type of plot ('auto', 'hist', 'box', 'bar')
        """
        if self.df is None:
            print("Error: No data loaded")
            return
        
        if column not in self.df.columns:
            print(f"Error: Column '{column}' not found")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Auto-detect plot type
        if kind == 'auto':
            if pd.api.types.is_numeric_dtype(self.df[column]):
                kind = 'hist'
            else:
                kind = 'bar'
        
        # Create plot
        if kind == 'hist':
            plt.hist(self.df[column].dropna(), bins=30, edgecolor='black')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.title(f'Distribution of {column}')
        
        elif kind == 'box':
            plt.boxplot(self.df[column].dropna())
            plt.ylabel(column)
            plt.title(f'Box Plot of {column}')
        
        elif kind == 'bar':
            value_counts = self.df[column].value_counts().head(10)
            plt.bar(range(len(value_counts)), value_counts.values)
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.title(f'Top 10 values in {column}')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{column}_{kind}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename)
        print(f"✓ Plot saved as {filename}")
        plt.close()
    
    def correlation_analysis(self):
        """Generate correlation heatmap for numeric columns"""
        if self.df is None:
            print("Error: No data loaded")
            return
        
        numeric_cols = self.df.select_dtypes(include='number').columns
        
        if len(numeric_cols) < 2:
            print("Error: Need at least 2 numeric columns for correlation")
            return
        
        plt.figure(figsize=(12, 8))
        correlation_matrix = self.df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        
        filename = f"correlation_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename)
        print(f"✓ Correlation heatmap saved as {filename}")
        plt.close()


def interactive_demo():
    """Interactive demo of data analysis agent"""
    
    print("="*60)
    print("DATA ANALYSIS AGENT - INTERACTIVE DEMO")
    print("="*60)
    
    # Get API key
    api_key = input("\nEnter your OpenAI API key: ")
    
    # Get CSV path
    csv_path = input("Enter path to CSV file: ")
    
    # Initialize agent
    print("\n" + "="*60)
    print("Initializing agent...")
    print("="*60 + "\n")
    
    agent = DataAnalysisAgent(api_key=api_key, csv_path=csv_path)
    
    if agent.df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Show summary
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    summary = agent.get_summary()
    print(f"Shape: {summary['shape']}")
    print(f"Columns: {summary['columns']}")
    print(f"\nMissing values:")
    for col, count in summary['missing_values'].items():
        if count > 0:
            print(f"  {col}: {count}")
    
    # Interactive query loop
    print("\n" + "="*60)
    print("Ask questions about your data!")
    print("Commands: 'viz <column>' for visualization, 'corr' for correlation, 'quit' to exit")
    print("="*60 + "\n")
    
    while True:
        question = input("Your question: ").strip()
        
        if not question:
            continue
        
        if question.lower() == 'quit':
            print("\nGoodbye!")
            break
        
        # Handle visualization command
        if question.lower().startswith('viz '):
            column = question[4:].strip()
            agent.visualize_column(column)
            continue
        
        # Handle correlation command
        if question.lower() == 'corr':
            agent.correlation_analysis()
            continue
        
        # Query agent
        print("\nAnalyzing...")
        response = agent.query(question)
        print(f"\n📊 Answer: {response}\n")
        print("-"*60 + "\n")


def automated_demo():
    """Automated demo with example dataset"""
    
    print("="*60)
    print("AUTOMATED DEMO - Example Analysis")
    print("="*60 + "\n")
    
    # Create sample dataset
    print("Creating sample dataset...")
    data = {
        'Product': ['A', 'B', 'C', 'A', 'B', 'C'] * 10,
        'Sales': [100, 150, 120, 110, 160, 130] * 10,
        'Profit': [20, 30, 25, 22, 32, 28] * 10,
        'Region': ['North', 'South', 'East', 'West', 'North', 'South'] * 10
    }
    df = pd.DataFrame(data)
    df.to_csv('sample_data.csv', index=False)
    
    # Initialize agent
    api_key = "your-openai-key-here"
    agent = DataAnalysisAgent(api_key=api_key, csv_path='sample_data.csv')
    
    # Example questions
    questions = [
        "What is the average sales value?",
        "Which product has the highest total sales?",
        "What is the correlation between Sales and Profit?",
        "How many unique regions are there?",
        "What is the total profit by product?"
    ]
    
    for question in questions:
        print(f"Q: {question}")
        answer = agent.query(question)
        print(f"A: {answer}\n")
        print("-"*60 + "\n")


if __name__ == "__main__":
    # Run interactive demo
    interactive_demo()
    
    # Uncomment for automated demo:
    # automated_demo()
