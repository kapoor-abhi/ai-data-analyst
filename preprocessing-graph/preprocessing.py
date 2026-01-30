import os
import json
import re
import traceback
import pandas as pd
import numpy as np
from typing import List, Literal, Optional, Dict
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# --- UTILS ---

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super(NpEncoder, self).default(obj)

def clean_json_markdown(text: str) -> str:
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return text.strip()

def clean_python_code(text: str) -> str:
    text = re.sub(r"^```python\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

# --- 1. DEEP SCANNER ---

def analyze_column(series: pd.Series):
    clean_series = series.dropna()
    total_count = len(series)
    non_null_count = len(clean_series)
    
    report = {
        "logical_type": "Unknown",
        "null_count": int(total_count - non_null_count),
        "pct_missing": round((1 - (non_null_count / total_count)) * 100, 2) if total_count > 0 else 0,
        "issues": []
    }

    if non_null_count == 0:
        report["logical_type"] = "Empty"
        return report

    numeric_converted = pd.to_numeric(clean_series, errors='coerce')
    num_valid_count = numeric_converted.notna().sum()
    
    if num_valid_count / non_null_count > 0.7:
        report["logical_type"] = "Numeric"
        
        if num_valid_count < non_null_count:
            report["logical_type"] = "Numeric (Dirty)"
            mask_failures = numeric_converted.isna()
            bad_values = clean_series[mask_failures].unique().tolist()
            report["issues"].append("Contains non-numeric characters")
            report["pollution_values"] = bad_values[:10]
            
        valid_nums = numeric_converted.dropna()
        if not valid_nums.empty:
            Q1 = valid_nums.quantile(0.25)
            Q3 = valid_nums.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            outliers = valid_nums[(valid_nums < lower) | (valid_nums > upper)]
            
            report["stats"] = {
                "min": valid_nums.min(),
                "max": valid_nums.max(),
                "mean": round(valid_nums.mean(), 2),
                "median": round(valid_nums.median(), 2),
                "std": round(valid_nums.std(), 2),
                "outlier_count": int(len(outliers))
            }
            
            if len(outliers) > 0:
                report["issues"].append("Has Outliers")
        
        return report

    unique_count = clean_series.nunique()
    
    if unique_count < 50 or (unique_count / total_count < 0.2):
        report["logical_type"] = "Categorical"
        
        lower_unique = clean_series.astype(str).str.lower().nunique()
        
        if lower_unique < unique_count:
            report["issues"].append("Inconsistent Casing")
            report["inconsistent_casing"] = True
            
        report["top_values"] = clean_series.value_counts().head(5).to_dict()
        return report

    report["logical_type"] = "Text/ID"
    return report

def run_deep_scan(file_path: str):
    print(f"Scanning file: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return {"error": f"Failed to load CSV: {str(e)}"}

    full_report = {
        "file_summary": {
            "rows": len(df),
            "cols": len(df.columns),
            "columns_list": df.columns.tolist()
        },
        "columns": {}
    }

    for col in df.columns:
        print(f"   > Analyzing {col}...")
        col_report = analyze_column(df[col])
        full_report["columns"][col] = col_report

    print("Scan Complete.")
    return json.dumps(full_report, cls=NpEncoder, indent=2)

# --- 2. STRATEGIST ---

class CleaningStep(BaseModel):
    step_type: Literal["impute", "drop_rows", "type_cast", "string_norm", "replace_value"] = Field(
        description="Category of action: 'impute', 'drop_rows', 'type_cast', 'string_norm', 'replace_value'"
    )
    function_name: str = Field(description="Pandas function: 'fillna', 'dropna', 'astype', 'to_numeric', 'to_datetime', 'replace'")
    parameters: dict = Field(description="Kwargs for the function, e.g. {'value': 'median'}")
    explanation: str = Field(description="Short rationale")

class ColumnStrategy(BaseModel):
    column_name: str
    logical_type: str
    steps: List[CleaningStep]

class DatasetCleaningPlan(BaseModel):
    strategies: List[ColumnStrategy]

def generate_cleaning_plan(profile_json_str: str, api_key: str):
    print("\n--- STEP 2: STRATEGIST ---")
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, api_key=api_key)
    parser = PydanticOutputParser(pydantic_object=DatasetCleaningPlan)

    example_json = """
    {
        "strategies": [
            {
                "column_name": "signup_date",
                "logical_type": "Date (Dirty)",
                "steps": [
                    { "step_type": "type_cast", "function_name": "to_datetime", "parameters": {"errors": "coerce"}, "explanation": "Convert" },
                    { "step_type": "impute", "function_name": "ffill", "parameters": {}, "explanation": "Forward fill missing dates" }
                ]
            }
        ]
    }
    """

    system_prompt = f"""You are a Principal Data Engineer.
    Input: Data Profile (JSON).
    Output: Cleaning Plan (JSON) that STRICTLY matches the schema.
    
    CRITICAL LOGIC RULES:
    1. NUMERICS:
       - If 'pollution_values' exist -> Replace with "np.nan" -> Convert to Numeric.
       - If 'outlier_count' > 0 -> Impute "median". Else -> "mean".
       
    2. DATES:
       - Step 1: Convert using 'to_datetime' (errors='coerce').
       - Step 2: NEVER DROP ROWS. Impute using 'ffill' (Forward Fill) followed by 'bfill' (Back Fill) to handle edges.
       
    3. TEXT / CATEGORICAL:
       - 'inconsistent_casing' -> Apply 'str.lower'.
       - Missing Values -> Fill with "Unknown" or the Mode.
       - NEVER DROP ROWS for text columns.
    
    4. IDS (user_id):
       - Do nothing.
    
    JSON FORMAT INSTRUCTIONS:
    {parser.get_format_instructions()}
    
    EXAMPLE OUTPUT STRUCTURE:
    {example_json}
    """

    user_message = f"""
    DATA PROFILE:
    {profile_json_str}
    
    Generate the plan now.
    """

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ])
        
        clean_text = clean_json_markdown(response.content)
        plan_obj = parser.parse(clean_text)
        
        print("Plan Validated & Parsed Successfully.")
        return plan_obj.model_dump_json(indent=2)

    except Exception as e:
        print(f" Error: {e}")
        return None

# --- 3. ENGINEER ---

def generate_python_code(csv_path: str, cleaning_plan_json: str, api_key: str) -> str:
    print("\n--- STEP 3: ENGINEER ---")
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, api_key=api_key)
    
    system_prompt = """You are a Senior Python Data Engineer.
    Task: Write a Python script to clean a dataset based on a specific JSON Plan.
    
    INPUT:
    1. CSV Path: '{csv_path}'
    2. Cleaning Plan (JSON)
    
    STRICT CODING RULES:
    1. WRITE LINEAR, EXPLICIT CODE. No loops over the JSON.
    2. FORBIDDEN SYNTAX (Deprecated):
       - DO NOT use `inplace=True`.
       - DO NOT use `fillna(method='ffill')` or `fillna(method='bfill')`.
       
    3. REQUIRED SYNTAX (Modern Pandas):
       - Impute Median: `df['col'] = df['col'].fillna(df['col'].median())`
       - Forward Fill:  `df['col'] = df['col'].ffill()`
       - Back Fill:     `df['col'] = df['col'].bfill()`
       - Both:          `df['col'] = df['col'].ffill().bfill()`
    
    4. HANDLING 'np.nan':
       - Use `np.nan` (object) not the string "np.nan".
       
    5. STRUCTURE:
       - Import pandas, numpy.
       - Load CSV.
       - [Code Block for Column 1]
       - [Code Block for Column 2]
       - ...
       - Save to 'cleaned_data.csv'.
       - Wrap in try/except.
    
    OUTPUT:
    - Return ONLY the Python Code.
    """

    user_message = f"""
    Generate the code for this plan:
    {cleaning_plan_json}
    """

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt.format(csv_path=csv_path)),
            HumanMessage(content=user_message)
        ])
        
        code = clean_python_code(response.content)
        print("Explicit Python Script Generated.")
        return code

    except Exception as e:
        print(f"Error generating code: {e}")
        return ""

# --- MAIN PIPELINE ---

def main_data_cleaning_pipeline(csv_file_path: str):
    print(f"\n========================================")
    print(f"STARTING AI CLEANING PIPELINE")
    print(f"File: {csv_file_path}")
    print(f"========================================\n")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        api_key = input("Enter Groq API Key: ").strip()
        if not api_key: 
            print("Error: API Key is required."); return

    print("\n[1/3] Running Deep Scanner...")
    profile_json = run_deep_scan(csv_file_path)
    if not profile_json or "error" in profile_json:
        print("Profiling Failed."); return
    print("Profile Generated.")

    print("\n[2/3] Generating Cleaning Strategy...")
    cleaning_plan_json = generate_cleaning_plan(profile_json, api_key)
    if not cleaning_plan_json:
        print("Strategy Generation Failed."); return
    print("Plan Created.")

    print("\n[3/3] Writing & Executing Code...")
    python_code = generate_python_code(csv_file_path, cleaning_plan_json, api_key)
    
    if not python_code:
        print("Code Generation Failed."); return

    with open("generated_cleaning_script.py", "w") as f:
        f.write(python_code)
    print("   -> Code saved to 'generated_cleaning_script.py'")

    print("\nExecuting Cleaning Script...")
    try:
        exec_globals = {}
        exec(python_code, exec_globals)
        
        if os.path.exists("cleaned_data.csv"):
            print(f"\n========================================")
            print(f"SUCCESS! Data Cleaned.")
            print(f"========================================")
            
            df_orig = pd.read_csv(csv_file_path)
            df_clean = pd.read_csv("cleaned_data.csv")
            print(f"Original Shape: {df_orig.shape}")
            print(f"Cleaned Shape:  {df_clean.shape}")
            print("\nPreview of Cleaned Data:")
            print(df_clean.head())
        else:
            print("Error: Script ran but 'cleaned_data.csv' was not created.")
            
    except Exception as e:
        print(f" Execution Runtime Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    np.random.seed(42)
    num_rows = 200

    t_ids = range(1001, 1001 + num_rows)
    cats = ['Electronics', 'electronics', 'Clothing', 'clothing', 'Home', 'HOME', 'Toys']
    category_col = np.random.choice(cats, num_rows)

    price_col = np.random.uniform(10, 500, num_rows).astype(object)
    price_col[0] = 5000000 
    price_col[10] = 2000000
    price_col[20:30] = "Unknown"
    price_col[40:45] = "Not Listed"

    date_rng = pd.date_range(start='2025-01-01', periods=num_rows, freq='D')
    date_col = pd.Series(date_rng).astype(object)
    date_col.iloc[50:60] = "Invalid Date"
    date_col.iloc[100:110] = np.nan

    ship_methods = ['Standard', 'Express', 'Overnight', np.nan]
    ship_col = np.random.choice(ship_methods, num_rows, p=[0.5, 0.3, 0.1, 0.1])

    df_big = pd.DataFrame({
        "transaction_id": t_ids,
        "category": category_col,
        "price": price_col,
        "transaction_date": date_col,
        "shipping_method": ship_col
    })
    
    big_filename = "retail_sales_messy.csv"
    df_big.to_csv(big_filename, index=False)
    print(f"Created '{big_filename}' with shape {df_big.shape}")

    main_data_cleaning_pipeline(big_filename)