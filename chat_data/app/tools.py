import sys
import io
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool

# Set the backend for Matplotlib to non-interactive (Agg)
# This prevents errors in environments without a display (like Docker)
plt.switch_backend('Agg')

@tool
def python_repl(code: str):
    """
    A Python shell to analyze data. Use this to execute python code. 
    The input should be a valid python command. 
    If you want to see a value, you should print it with `print(...)`.
    If you create a plot, save it as 'app/sandbox/temp_plot.png'.
    """
    # Change directory to sandbox to ensure file isolation and path consistency
    # In Docker, this is /code/app/sandbox
    sandbox_path = "/code/app/sandbox"
    orig_dir = os.getcwd()
    
    if not os.path.exists(sandbox_path):
        os.makedirs(sandbox_path, exist_ok=True)
        
    os.chdir(sandbox_path)
    
    # Standard output capture
    stdout = io.StringIO()
    
    # Provide the environment with access to essential libraries
    # We include 'plt' so the LLM can call plt.savefig()
    local_vars = {"pd": pd, "plt": plt, "sns": sns}
    
    try:
        # Clear any previous plots to avoid overlapping data
        plt.clf()
        plt.close('all')
        
        # Redirect stdout to our buffer
        old_stdout = sys.stdout
        sys.stdout = stdout
        
        try:
            # Execute the code provided by the LLM
            exec(code, {}, local_vars)
        finally:
            # Always restore stdout
            sys.stdout = old_stdout
        
        output = stdout.getvalue()
        return output if output else "Code executed successfully (no output)."
        
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        # Always return to the original working directory
        os.chdir(orig_dir)

@tool
def get_csv_schema(file_path: str):
    """
    Returns the first 5 rows and the column names of a CSV file.
    Use this at the beginning of a conversation to understand the data.
    """
    try:
        # Ensure path is handled correctly if LLM passes relative path
        df = pd.read_csv(file_path)
        info = {
            "columns": df.columns.tolist(),
            "sample_data": df.head(3).to_dict(orient='records'),
            "row_count": len(df),
            "data_types": df.dtypes.apply(lambda x: str(x)).to_dict()
        }
        return str(info)
    except Exception as e:
        return f"Error reading CSV: {str(e)}"