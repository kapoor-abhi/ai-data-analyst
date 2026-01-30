import os
import io
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Annotated, TypedDict, List, Dict, Any
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from contextlib import redirect_stdout
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    groq_api_key=GROQ_API_KEY
)

class PythonREPL:
    def run(self, code: str) -> str:
        obs = io.StringIO()
        try:
            with redirect_stdout(obs):
                exec(code, globals())
            return obs.getvalue()
        except Exception as e:
            return f"Error: {str(e)}"

repl = PythonREPL()

class AgentState(TypedDict):
    csv_path: str
    df_info: str
    analysis_plan: str
    current_code: str
    execution_error: str
    charts_generated: List[str]
    messages: List[BaseMessage]
    iteration_count: int

def profiler_node(state: AgentState):
    print("--- PROFILING DATA ---")
    path = state["csv_path"]
    
    try:
        df = pd.read_csv(path)
        
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        
        description = df.describe(include='all').to_string()
        sample = df.head(5).to_string()
        
        fingerprint = f"""
        DATASET INFO:
        {info_str}
        
        STATISTICAL SUMMARY:
        {description}
        
        SAMPLE ROWS:
        {sample}
        """
        return {"df_info": fingerprint, "messages": [HumanMessage(content="Data profiled successfully.")]}
    
    except Exception as e:
        return {"execution_error": f"Error reading CSV: {str(e)}"}

def planner_node(state: AgentState):
    print("--- PLANNING ANALYSIS ---")
    
    prompt = f"""
    You are an expert Data Scientist. Based on the dataset profile below, identify the most important columns for analysis.
    
    {state['df_info']}
    
    Your task:
    1. Identify 2-3 most important numerical columns for Univariate analysis (histograms).
    2. Identify 1-2 important categorical columns for Univariate analysis (bar charts).
    3. Identify 2 pairs of columns for Bivariate analysis (scatter plots or box plots).
    
    Output a clear plan.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "analysis_plan": response.content,
        "messages": [AIMessage(content=response.content)],
        "iteration_count": 0
    }

import re

def programmer_node(state: AgentState):
    print("--- GENERATING CODE ---")
    
    error_feedback = ""
    if state.get("execution_error"):
        error_feedback = f"\n\nYour previous code failed with this error: {state['execution_error']}\nPlease fix the code and try again."

    prompt = f"""
    You are a Python Data Visualization expert. Write a script to perform the following analysis plan:
    
    PLAN:
    {state['analysis_plan']}
    
    DATA METADATA:
    {state['df_info']}
    
    FILE PATH: The CSV is located at '{state['csv_path']}'
    
    REQUIREMENTS:
    1. Use pandas, matplotlib.pyplot, and seaborn.
    2. Save each chart as a separate PNG file.
    3. Use plt.close().
    4. Provide ONLY the Python code.
    
    {error_feedback}
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    code = response.content.replace("```python", "").replace("```", "").strip()
    
    return {
        "current_code": code,
        "iteration_count": state.get("iteration_count", 0) + 1
    }

def executor_node(state: AgentState):
    print("--- EXECUTING CODE ---")
    code = state["current_code"]
    
    result = repl.run(code)
    
    if "Error:" in result:
        print(f"Execution Failed: {result}")
        return {
            "execution_error": result,
            "messages": [HumanMessage(content=f"Execution failed: {result}")]
        }
    else:
        print("Execution Successful!")
        generated_files = re.findall(r"['\"](.+?\.png)['\"]", code)
        
        return {
            "execution_error": "",
            "charts_generated": list(set(generated_files)),
            "messages": [HumanMessage(content="Code executed and charts saved.")]
        }

def router(state: AgentState):
    if state.get("execution_error") and state.get("iteration_count", 0) < 3:
        return "retry"
    elif state.get("execution_error"):
        return "failed"
    else:
        return "success"

workflow = StateGraph(AgentState)

workflow.add_node("profiler", profiler_node)
workflow.add_node("planner", planner_node)
workflow.add_node("programmer", programmer_node)
workflow.add_node("executor", executor_node)

workflow.set_entry_point("profiler")
workflow.add_edge("profiler", "planner")
workflow.add_edge("planner", "programmer")
workflow.add_edge("programmer", "executor")

workflow.add_conditional_edges(
    "executor",
    router,
    {
        "retry": "programmer",
        "success": END,
        "failed": END
    }
)

app = workflow.compile()

def main(csv_file_path: str):
    initial_state = {
        "csv_path": csv_file_path,
        "messages": [HumanMessage(content=f"Analyze the dataset: {csv_file_path}")],
        "iteration_count": 0,
        "charts_generated": []
    }

    print(f"Starting Analysis for: {csv_file_path}")
    
    final_output = app.invoke(initial_state)

    if final_output.get("execution_error"):
        print(f"Status: Failed after {final_output['iteration_count']} attempts.")
        print(f"Final Error: {final_output['execution_error']}")
    else:
        print(f"Status: Success! Generated {len(final_output['charts_generated'])} charts.")
        print(final_output["analysis_plan"])

    return final_output

if __name__ == "__main__":
    test_csv = "test_data.csv"
    if not os.path.exists(test_csv):
        import numpy as np
        data = {
            'Revenue': np.random.randint(1000, 5000, 100),
            'Marketing_Spend': np.random.randint(100, 1000, 100),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
            'Customer_Satisfaction': np.random.uniform(1, 5, 100)
        }
        pd.DataFrame(data).to_csv(test_csv, index=False)

    result = main(test_csv)

class AgentState(TypedDict):
    csv_path: str
    df_info: str
    user_input: str
    analysis_plan: str
    current_code: str
    execution_error: str
    charts_generated: List[str]
    messages: Annotated[List[BaseMessage], "The conversation history"]
    iteration_count: int
    next_step: str

def intent_router_node(state: AgentState):
    print("--- ROUTING USER INTENT ---")
    
    prompt = f"""
    You are a Data Orchestrator.
    
    USER MESSAGE: "{state['user_input']}"
    DATA INFO: {state['df_info']}
    
    Respond with ONLY one word: profile, query, or visualize.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    intent = response.content.lower().strip().replace("'", "").replace(".", "")
    
    if intent not in ['profile', 'query', 'visualize']:
        intent = 'query'
        
    print(f"Detected Intent: {intent}")
    return {"next_step": intent}

import duckdb

def data_query_node(state: AgentState):
    print("--- QUERYING DATA (SQL) ---")
    
    sql_prompt = f"""
    You are a SQL expert.
    
    DATA SCHEMA:
    {state['df_info']}
    
    USER QUESTION:
    "{state['user_input']}"
    
    Only return SQL.
    """
    
    sql_response = llm.invoke([HumanMessage(content=sql_prompt)])
    sql_query = sql_response.content.replace("```sql", "").replace("```", "").strip()
    
    try:
        query_result = duckdb.query(sql_query).to_df().to_string()
        error_msg = ""
    except Exception as e:
        query_result = ""
        error_msg = str(e)

    if error_msg:
        return {
            "execution_error": error_msg,
            "messages": [HumanMessage(content=f"SQL Query failed: {error_msg}")]
        }

    summary_prompt = f"""
    USER QUESTION: {state['user_input']}
    SQL RESULT:
    {query_result}
    """
    
    summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
    
    return {
        "messages": state["messages"] + [AIMessage(content=summary_response.content)],
        "execution_error": ""
    }

def visualizer_node(state: AgentState):
    print("--- GENERATING DYNAMIC VISUALIZATION ---")
    
    error_feedback = ""
    if state.get("execution_error"):
        error_feedback = f"\n\nYour previous code failed with this error: {state['execution_error']}\nPlease fix the code and try again."

    prompt = f"""
    USER REQUEST: "{state['user_input']}"
    
    DATA SCHEMA:
    {state['df_info']}
    
    FILE PATH: '{state['csv_path']}'
    
    Provide ONLY Python code.
    
    {error_feedback}
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    code = response.content.replace("```python", "").replace("```", "").strip()
    
    return {
        "current_code": code,
        "iteration_count": state.get("iteration_count", 0) + 1
    }