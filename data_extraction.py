import os
import re
from typing import List
import numpy as pd
import pandas as pd
import numpy as np

# ========== USER CONFIG ==========
INPUT_DIR = "data"   # folder with your Excel files
VALID_EXTS = {".xlsx", ".xlsm", ".xls"}  # add .xlsb if needed (requires pyxlsb)
SCAN_ROWS = 20  # how many rows to search for header
TARGET_COLUMNS = [
        'JobNumber', 'Office', 'Office (Div)', 'ProjectTitle', 'Client', 
        'Location (Country)', 'Gross Fee (USD)', 'Fee Earned (USD)', 
        'Gross Fee Yet To Be Earned (USD)', 'Currency', 'GrossFee', 
        'GrossFeeEarned', 'GrossFeeYetToBeEarned', 'Status', 'NewProject', 
        'StartDate', 'Anticipated EndDate', 'ProjectType'
    ]
# =================================

def list_excel_files(folder: str, exts=VALID_EXTS) -> List[str]:
    files = []
    for root, _, filenames in os.walk(folder):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in exts:
                files.append(os.path.join(root, fn))
    return files

def normalize(name: str) -> str:
    """
    Normalize column names so:
    - 'Project Title' -> 'projecttitle'
    - 'ProjectTitle'  -> 'projecttitle'
    - 'Project_Type'  -> 'projecttype'
    - 'Client'        -> 'client'
    - 'Gross Fee Yet To Be Earned (USD)' -> 'grossfeeyettobeearnedusd'
    """
    if name is None or type(name) is not str:
        return ""
    s = name.strip().lower()
    s = re.sub(r"[\s_\-]+", "", s)     # remove spaces/underscores/dashes
    s = re.sub(r"[^\w]", "", s)        # remove other punctuation
    return s

def find_header_row(df_no_header):
    """
    Return the earliest (0-based) row index where BOTH 'Client' and 'Currency' appear.
    Case-insensitive; whitespace/punct ignored.
    """
    max_r = min(len(df_no_header), SCAN_ROWS)
    for r in range(max_r):
        row = [normalize(v) for v in df_no_header.iloc[r].tolist()]
        if "jobnumber" in row and "currency" in row:
            return r
    return None  # not found

def _to_number(series: pd.Series) -> pd.Series:
    # Keep digits, sign, and decimal; turn "(123)" into "-123"
    s = series.astype(str).fillna("")
    s = s.str.replace(r"\(([\d\.,]+)\)", r"-\1", regex=True)
    s = s.str.replace(r"[^\d\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def read_sheet(path, sheet):
    raw = pd.read_excel(path, sheet_name=sheet, header=None, dtype=str, engine=None)
    print(f"Processing sheet : {sheet}")
    if raw.empty:
        return pd.DataFrame()

    hdr = find_header_row(raw)
    if hdr is None:
        print(f"no headers found for {sheet} - skipping..")
        return pd.DataFrame()

    # set header and slice data rows
    cols = raw.iloc[hdr].tolist()
    
    # Clean up na column names and duplicates
    cleaned_cols = []
    seen_cols = {}
    
    for i, col in enumerate(cols):
        # Handle None/NaN/empty columns
        if col is None or (isinstance(col, float) and pd.isna(col)) or str(col).strip() == '':
            clean_col = f"unnamed_col_{i}"
        else:
            clean_col = str(col).strip()
        
        # Handle duplicates by appending suffix
        if clean_col in seen_cols:
            seen_cols[clean_col] += 1
            clean_col = f"{clean_col}_duplicate_{seen_cols[clean_col]}"
        else:
            seen_cols[clean_col] = 0
        
        cleaned_cols.append(clean_col)
    
    data = raw.iloc[hdr+1:].reset_index(drop=True)
    data.columns = cleaned_cols  # Use cleaned column names
    
    print(f"Columns in sheet {sheet} : {cleaned_cols}")
    out = pd.DataFrame()
    
    # Map each target column to the actual column in the data
    for target_col in TARGET_COLUMNS:
        if target_col in data.columns:
            out[target_col] = data[target_col]
            print(f"✓ Found: {target_col}")
        else:
            print(f"✗ Missing: {target_col}")
    
    # Ensure the DataFrame has ONLY the target columns in the exact order
    out = out.reindex(columns=TARGET_COLUMNS)
    
    # Filter out rows with empty ProjectTitle
    # if 'ProjectTitle' in out.columns and not out['ProjectTitle'].isna().all():
    #     valid_mask = out['ProjectTitle'].astype(str).str.strip().replace({"": None, "nan": None}).notna()
    #     out = out[valid_mask].reset_index(drop=True)

    # else:
    #     print("No ProjectTitle column or all values are empty - returning empty DataFrame")
    #     return pd.DataFrame(columns=TARGET_COLUMNS)
    
    if out.empty:
        print(f"No valid data found in {sheet}")
        return pd.DataFrame(columns=TARGET_COLUMNS)

    # Convert numeric columns
    numeric_columns = [
        'Gross Fee (USD)', 'Fee Earned (USD)', 'Gross Fee Yet To Be Earned (USD)',
        'GrossFee', 'GrossFeeEarned', 'GrossFeeYetToBeEarned'
    ]
    
    for col in numeric_columns:
        if col in out.columns and not out[col].isna().all():
            try:
                out[col] = _to_number(out[col])
            except:
                pass  # Keep as string if conversion fails

    print(f"Successfully processed {sheet}: {len(out)} rows")
    return out

def read_file(path):
    try:
        xl = pd.ExcelFile(path, engine=None)
    except Exception:
        print("Could not read file")
        return pd.DataFrame()
    frames = []
    for s in xl.sheet_names:
        try:
            frames.append(read_sheet(path, s))
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True)

def extract_all(input_dir=INPUT_DIR):
    dfs = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in VALID_EXTS:
                dfs.append(read_file(os.path.join(root, f)))
    return pd.concat(dfs, ignore_index=True)
