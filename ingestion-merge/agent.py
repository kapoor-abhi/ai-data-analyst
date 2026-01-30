import os
import time
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, TypedDict
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

class AgentState(TypedDict):
    file_paths: List[str]
    dfs: Dict[str, Any]
    user_query: str
    python_code: str
    error: str
    is_large_dataset: bool

def get_dataframe_schema(dfs: Dict[str, pd.DataFrame]) -> str:
    schema_info = ""
    for name, df in dfs.items():
        schema_info += f"\n--- File: {name} ---\n"
        schema_info += f"Columns: {list(df.columns)}\n"
        schema_info += f"Data Types: {list(df.dtypes)}\n"
        schema_info += f"Shape: {df.shape}\n"
        schema_info += f"Sample: {df.head(2).values.tolist()}\n"
    return schema_info

def ingest_data_node(state: AgentState):
    print("--- 1. INGESTING DATA ---")
    file_paths = state['file_paths']
    dfs = {}
    try:
        for path in file_paths:
            filename = os.path.basename(path)
            df = pd.read_csv(path)
            dfs[filename] = df
            print(f"Loaded {filename}")
        return {"dfs": dfs, "error": None}
    except Exception as e:
        return {"error": str(e)}

def optimize_data_node(state: AgentState):
    print("--- 2. CHECKING DATA SIZE & OPTIMIZING ---")
    dfs = state['dfs']
    is_large = False
    THRESHOLD = 50000
    
    for name, df in dfs.items():
        rows = len(df)
        print(f"File '{name}' has {rows:,} rows.")
        
        if rows > THRESHOLD:
            is_large = True
            print(f" LARGE DATASET DETECTED. Applying optimizations to '{name}'")
            start_mem = df.memory_usage(deep=True).sum() / 1024**2
            
            for col in df.select_dtypes(include=['object']).columns:
                num_unique = len(df[col].unique())
                num_total = len(df)
                if num_unique / num_total < 0.5:
                    df[col] = df[col].astype('category')
            
            end_mem = df.memory_usage(deep=True).sum() / 1024**2
            print(f"    Memory reduced from {start_mem:.2f} MB to {end_mem:.2f} MB")
            dfs[name] = df

    return {"dfs": dfs, "is_large_dataset": is_large}

def column_selection_node(state: AgentState):
    print("--- 3. GENERATING CODE ---")
    schema = get_dataframe_schema(state['dfs'])
    is_large = state.get('is_large_dataset', False)
    performance_instruction = ""
    if is_large:
        performance_instruction = """
        PERFORMANCE MODE ACTIVE (Large Dataset):
        1. DO NOT use loops (iterrows). Use vectorized Pandas operations only.
        2. The data is optimized (strings -> category).
        3. Use `pivot_table` with `aggfunc='first'` (or similar) to handle duplicate entries safely.
        """
    error_context = f"\nPREVIOUS ERROR: {state['error']}" if state.get("error") else ""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a High-Performance Python Data Expert. 
        You have a dictionary `dfs` where Keys=Filenames and Values=DataFrames.
        
        Task: Write Python code to update the DataFrames in `dfs` based on the User Query.
        
        {performance_instruction}
        
        CRITICAL RULES:
        1. Return ONLY valid Python code inside ```python ... ``` blocks.
        2. **NEVER assume the filename is 'file.csv'.** 3. **YOU MUST OVERWRITE THE DATAFRAME IN THE DICTIONARY.**
        4. If pivoting, ALWAYS use `pivot_table` and specify `aggfunc='first'`.
        """),
        ("user", "User Query: {query}\n{error_context}")
    ])
    
    chain = prompt_template | llm
    response = chain.invoke({
        "schema": schema,
        "query": state['user_query'],
        "performance_instruction": performance_instruction,
        "error_context": error_context
    })
    
    raw = response.content
    match = re.search(r"```python(.*?)```", raw, re.DOTALL)
    code = match.group(1).strip() if match else raw.strip()
    
    print(f"Generated Code:\n{code}")
    return {"python_code": code, "error": None}

def execute_code_node(state: AgentState):
    print("--- 4. EXECUTING CODE ---")
    code = state['python_code']
    dfs = state['dfs']
    local_scope = {"dfs": dfs, "pd": pd, "np": np}
    
    try:
        exec(code, {}, local_scope)
        print("Execution Successful.")
        return {"dfs": local_scope["dfs"], "error": None}
    except Exception as e:
        print(f"Execution Failed: {e}")
        return {"error": str(e)}

def human_review_node(state: AgentState):
    print("\n--- 5. HUMAN REVIEW ---")
    dfs = state['dfs']
    for name, df in dfs.items():
        print(f"\n[PREVIEW] File: {name}")
        print(df.head(10))
        print(f"Shape: {df.shape}")
    
    user_choice = input("\nAre you happy with this transformation? (yes/no): ").lower().strip()
    if user_choice == "yes":
        return {"error": None}
    else:
        new_instruction = input("Please describe what changes you want: ")
        return {"user_query": new_instruction, "error": "User rejected the previous output."}

def router_logic(state: AgentState):
    if state.get("error") and "User rejected" not in state.get("error", ""):
        return "retry_llm"
    return "review"

def review_router(state: AgentState):
    if state.get("error"):
        return "retry_llm"
    return "end"

workflow = StateGraph(AgentState)
workflow.add_node("ingest_data", ingest_data_node)
workflow.add_node("optimize_data", optimize_data_node)
workflow.add_node("reasoning", column_selection_node)
workflow.add_node("execution", execute_code_node)
workflow.add_node("human_review", human_review_node)

workflow.set_entry_point("ingest_data")
workflow.add_edge("ingest_data", "optimize_data")
workflow.add_edge("optimize_data", "reasoning")
workflow.add_edge("reasoning", "execution")

workflow.add_conditional_edges(
    "execution",
    router_logic,
    {"retry_llm": "reasoning", "review": "human_review"}
)
workflow.add_conditional_edges(
    "human_review",
    review_router,
    {"retry_llm": "reasoning", "end": END}
)

app = workflow.compile()

if __name__ == "__main__":
    import os
    FILENAME = "litmus-test.csv"
    if not os.path.exists(FILENAME):
        print(f" Error: '{FILENAME}' was not found in the directory.")
        print("Please make sure the file is in the same folder as agent.py")
    else:
        user_prompt = "I want unique HNUM as rows and COMPNAME as columns, with RESULT as values."
        print(user_prompt)
        
        initial_state = {
            "file_paths": [FILENAME],
            "user_query": user_prompt,
            "dfs": {},
            "error": None
        }

        print(f"\n--- STARTING AGENT ON: {FILENAME} ---")
        start_time = time.time()
        
        app.invoke(initial_state)
        print(f"\n--- TOTAL EXECUTION TIME: {time.time() - start_time:.2f} seconds ---")