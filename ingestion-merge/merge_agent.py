import pandas as pd
from typing import List, Dict, Any, TypedDict, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
import re

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

class MergeState(TypedDict):
    """
    State specifically for the Merging Phase.
    """
    dfs: Dict[str, Any]
    merged_df: Optional[Any]
    suggestion: str
    user_feedback: str
    python_code: str
    error: str

def get_dataframe_schema(dfs: Dict[str, pd.DataFrame]) -> str:
    """Extracts column names to help LLM find common keys."""
    schema_info = ""
    for name, df in dfs.items():
        schema_info += f"\n--- File: {name} ---\n"
        schema_info += f"Columns: {list(df.columns)}\n"
        schema_info += f"Shape: {df.shape}\n"
    return schema_info

def analyze_merge_node(state: MergeState):
    """
    LLM analyzes columns to find common keys for merging.
    """
    print("--- 1. ANALYZING DATASETS FOR MERGE KEYS ---")
    dfs = state['dfs']
    
    if len(dfs) < 2:
        return {"suggestion": "Only one file provided. No merge needed.", "error": "Not enough files"}

    schema = get_dataframe_schema(dfs)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Data Integration Expert.
        Analyze the schemas of the provided DataFrames.
        Identify the most likely column(s) to merge these datasets on.
        
        Rules:
        1. Look for columns with identical or similar names (e.g., 'ID' vs 'Client_ID').
        2. If more than 2 files, try to find a common key across all.
        3. Output a SHORT suggestion string explaining the strategy. 
           Example: "Merge 'bank.csv' and 'data.csv' on column 'HNUM'."
        """),
        ("user", "Schemas:\n{schema}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"schema": schema})
    
    print(f"Agent Suggestion: {response.content}")
    return {"suggestion": response.content, "error": None}

def human_strategy_node(state: MergeState):
    """
    Presents suggestion to user and gets their specific merge instruction.
    """
    print("\n--- 2. MERGE STRATEGY SELECTION ---")
    suggestion = state['suggestion']
    
    print(f"Agent Suggestion: {suggestion}")
    print("-" * 40)
    print("Options:")
    print("1. Accept Suggestion")
    print("2. Enter Custom Merge Instruction (e.g., 'Merge on Date and ID')")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        return {"user_feedback": suggestion}
    else:
        custom_prompt = input("Enter your specific merge instruction: ")
        return {"user_feedback": custom_prompt}
        
def generate_merge_code_node(state: MergeState):
    """
    Generates Python code to perform the merge.
    """
    print("--- 3. GENERATING MERGE CODE ---")
    schema = get_dataframe_schema(state['dfs'])
    instruction = state['user_feedback']
    
    error_context = f"Previous Error: {state['error']}" if state.get("error") else ""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Python Pandas Expert.
        
        CRITICAL CONTEXT:
        1. You are running inside a function where a dictionary named `dfs` IS ALREADY DEFINED.
        2. `dfs` keys are filename strings (e.g. 'file1.csv').
        3. Do NOT try to read csv files (pd.read_csv). USE the `dfs` variable directly.
        
        Task: 
        Generate Python code to merge dataframes from `dfs` into a single variable `merged_df`.
        
        Rules:
        1. Return ONLY valid Python code inside Markdown blocks (```python ... ```).
        2. Do NOT write any explanations or text outside the code block.
        3. Use `pd.merge()`.
        4. Assign the result to `merged_df`.
        """),
        ("user", "Schemas:\n{schema}\n\nUser Instruction: {instruction}\n{error_context}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "schema": schema,
        "instruction": instruction,
        "error_context": error_context
    })
    
    raw_content = response.content
    
    match = re.search(r"```python(.*?)```", raw_content, re.DOTALL)
    if match:
        code = match.group(1).strip()
    else:
        code = raw_content.strip()
        
    print(f"Generated Merge Code:\n{code}")
    
    return {"python_code": code}

def execute_merge_node(state: MergeState):
    print("--- 4. EXECUTING MERGE ---")
    dfs = state['dfs']
    code = state['python_code']
    
    local_scope = {"dfs": dfs, "pd": pd}
    
    try:
        exec(code, {}, local_scope)
        
        result_df = local_scope.get("merged_df")
        if result_df is None:
            raise ValueError("Code ran, but 'merged_df' variable was not created.")
            
        print(f"Merge Successful. Result Shape: {result_df.shape}")
        return {"merged_df": result_df, "error": None}
        
    except Exception as e:
        print(f"Merge Failed: {e}")
        return {"error": str(e)}

def route_merge_retry(state: MergeState):
    if state.get("error"):
        return "retry"
    return "success"

workflow = StateGraph(MergeState)
workflow.add_node("analyze", analyze_merge_node)
workflow.add_node("human_strategy", human_strategy_node)
workflow.add_node("generate", generate_merge_code_node)
workflow.add_node("execute", execute_merge_node)

workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "human_strategy")
workflow.add_edge("human_strategy", "generate")
workflow.add_edge("generate", "execute")

workflow.add_conditional_edges(
    "execute",
    route_merge_retry,
    {
        "retry": "generate",
        "success": END
    }
)
app = workflow.compile()

if __name__ == "__main__":
    print("--- STARTING MERGE AGENT TEST ---")
    
    df1 = pd.DataFrame({
        'HNUM': ['A101', 'B202', 'C303'],
        'Age': [34, 45, 22],
        'Date': ['2023-01-01', '2023-01-01', '2023-01-02']
    })
    
    df2 = pd.DataFrame({
        'HNUM': ['A101', 'B202', 'D404'],
        'Diagnosis': ['Flu', 'Cold', 'Fever'],
        'Date': ['2023-01-01', '2023-01-05', '2023-01-02']
    })
    
    print(f"Dataset 1 Shape: {df1.shape}")
    print(f"Dataset 2 Shape: {df2.shape}")
    
    initial_state = {
        "dfs": {
            "patient_data": df1, 
            "medical_data": df2
        },
        "error": None
    }
  
    result = app.invoke(initial_state)
    
    if result.get("merged_df") is not None:
        print("\n--- FINAL MERGED DATASET ---")
        print(result["merged_df"])
        print("\nSuccess! The datasets are merged.")
    else:
        print("\n--- MERGE FAILED ---")
        print(f"Last Error: {result.get('error')}")