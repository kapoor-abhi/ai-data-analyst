import sys
import io
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool



plt.switch_backend('Agg')

@tool
def python_repl(code: str):
    """
    A Python shell to analyze data. Use this to execute python code. 
    The input should be a valid python command. 
    If you want to see a value, you should print it with `print(...)`.
    If you create a plot, save it as 'app/sandbox/temp_plot.png'.
    """


    sandbox_path = "/code/app/sandbox"
    orig_dir = os.getcwd()
    
    if not os.path.exists(sandbox_path):
        os.makedirs(sandbox_path, exist_ok=True)
        
    os.chdir(sandbox_path)
    

    stdout = io.StringIO()
    


    local_vars = {"pd": pd, "plt": plt, "sns": sns}
    
    try:

        plt.clf()
        plt.close('all')
        

        old_stdout = sys.stdout
        sys.stdout = stdout
        
        try:

            exec(code, {}, local_vars)
        finally:

            sys.stdout = old_stdout
        
        output = stdout.getvalue()
        return output if output else "Code executed successfully (no output)."
        
    except Exception as e:
        return f"Error: {str(e)}"
    finally:

        os.chdir(orig_dir)

@tool
def get_csv_schema(file_path: str):
    """
    Returns the first 5 rows and the column names of a CSV file.
    Use this at the beginning of a conversation to understand the data.
    """
    try:

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