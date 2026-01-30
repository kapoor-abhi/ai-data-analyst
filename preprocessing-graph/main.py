from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

import os
import sys
import json
import io
import pandas as pd
import numpy as np
import traceback
from typing import TypedDict, Optional, List, Any, Dict

class AgentState(TypedDict):
    csv_file_path: str
    deep_profile_report: str
    cleaning_plan: Optional[str]
    generated_code: Optional[str]
    execution_result: Optional[str]
    audit_verdict: Optional[str]
    revision_count: int

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

def infer_and_scan_column(series: pd.Series, col_name: str) -> Dict[str, Any]:
    stats = {}
    clean_series = series.dropna()
    total_count = len(series)
    non_null_count = len(clean_series)
    
    stats["null_count"] = int(series.isnull().sum())
    stats["pct_missing"] = round((stats["null_count"] / total_count) * 100, 2)
    stats["physical_dtype"] = str(series.dtype)
    
    if non_null_count == 0:
        stats["logical_type"] = "Empty"
        return stats

    is_id_by_name = any(x in col_name.lower() for x in ['id', 'uuid', 'pk', 'key'])
    unique_ratio = clean_series.nunique() / non_null_count
    
    if is_id_by_name and unique_ratio > 0.9:
        stats["logical_type"] = "ID_Column"
        return stats

    numeric_converted = pd.to_numeric(clean_series, errors='coerce')
    num_success_count = numeric_converted.notna().sum()
    
    if num_success_count / non_null_count > 0.6:
        mask_dirty = numeric_converted.isna()
        dirty_values = clean_series[mask_dirty].unique().tolist()
        
        stats["logical_type"] = "Numeric (Dirty)" if dirty_values else "Numeric"
        stats["pollution"] = dirty_values[:10]

        valid_nums = numeric_converted.dropna()
        if not valid_nums.empty:
            stats["median"] = valid_nums.median()
            stats["mean"] = valid_nums.mean()
            if abs(stats["mean"] - stats["median"]) > (0.5 * abs(stats["median"])):
                stats["has_outliers"] = True
            else:
                stats["has_outliers"] = False
        return stats

    try:
        date_converted = pd.to_datetime(clean_series, errors='coerce')
        if date_converted.notna().sum() / non_null_count > 0.6:
            mask_dirty = date_converted.isna()
            dirty_values = clean_series[mask_dirty].unique().tolist()
            stats["logical_type"] = "Date (Dirty)" if dirty_values else "Date"
            stats["pollution"] = dirty_values[:10]
            return stats
    except:
        pass

    if unique_ratio < 0.2:
        stats["logical_type"] = "Categorical"
        lower_unique = clean_series.astype(str).str.lower().nunique()
        stats["inconsistent_casing"] = clean_series.nunique() > lower_unique
    else:
        stats["logical_type"] = "Text"
    
    return stats

def profiler_node(state: AgentState):
    print("\n--- STEP 1: DEEP PROFILER ---")
    file_path = state["csv_file_path"]
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except:
        df = pd.read_csv(file_path, encoding='latin-1')

    profile_report = {"columns": {}}
    for col in df.columns:
        profile_report["columns"][col] = infer_and_scan_column(df[col], col)

    report_json = json.dumps(profile_report, cls=NpEncoder, indent=2)
    return {"deep_profile_report": report_json}