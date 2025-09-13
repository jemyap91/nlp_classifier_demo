# import ollama
import os
import re
from typing import Dict, List, Optional
import pandas as pd

# ========== USER CONFIG ==========
INPUT_DIR = "data"   # folder with your Excel files
VALID_EXTS = {".xlsx", ".xlsm", ".xls"}  # add .xlsb if needed (requires pyxlsb)
TITLE_STD = "ProjectTitle"
TYPE_STD  = "ProjectType"
SCAN_ROWS = 20  # how many rows to search for header
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
    - 'Client'       -> 'client'
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
        if "client" in row and "currency" in row:
            return r
    return None  # not found

def read_sheet(path, sheet):
    raw = pd.read_excel(path, sheet_name=sheet, header=None, dtype=str, engine=None)
    if raw.empty:
        return pd.DataFrame(columns=["project_title", "project_type", "client", "source_file", "sheet_name", "excel_row"])

    hdr = find_header_row(raw)
    if hdr is None:
        return pd.DataFrame(columns=["project_title", "project_type", "client", "source_file", "sheet_name", "excel_row"])

    # set header and slice data rows
    cols = raw.iloc[hdr].tolist()
    data = raw.iloc[hdr+1:].reset_index(drop=True)
    data.columns = cols

    # find title/type columns accepting both spaced and nospace variants
    # (we normalize names to match)
    title_col = None
    type_col  = None
    client_col = None
    for c in data.columns:
        nc = normalize(c)
        if title_col is None and nc in {"projecttitle"}:
            title_col = c
        if type_col is None and nc in {"projecttype"}:
            type_col = c
        if client_col is None and nc in {"client"}:
            client_col = c

    if title_col is None:
        return pd.DataFrame(columns=["project_title", "project_type", "client", "source_file", "sheet_name", "excel_row"])

    out = pd.DataFrame()
    out["project_title"] = data[title_col].astype(str).str.strip()
    out["project_type"]  = data[type_col].astype(str).str.strip() if type_col in data.columns else None
    out["client"]  = data[client_col].astype(str).str.strip() if client_col in data.columns else None

    # keep non-empty titles only
    out = out[out["project_title"].replace({"": None, "nan": None}).notna()]

    # provenance; Excel’s visible row number = header_row + 2 + df index
    out["excel_row"]   = out.index + (hdr + 2)
    out["source_file"] = os.path.basename(path)
    out["sheet_name"]  = sheet
    return out

def read_file(path):
    try:
        xl = pd.ExcelFile(path, engine=None)
    except Exception:
        return pd.DataFrame(columns=["project_title", "project_type", "client", "source_file", "sheet_name", "excel_row"])
    frames = []
    for s in xl.sheet_names:
        try:
            frames.append(read_sheet(path, s))
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["project_title", "project_type", "client", "source_file", "sheet_name", "excel_row"]
    )

def extract_all(input_dir=INPUT_DIR):
    dfs = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in VALID_EXTS:
                dfs.append(read_file(os.path.join(root, f)))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(
        columns=["project_title", "project_type", "client", "source_file", "sheet_name", "excel_row"]
    )

# def main():
#     raw_data = extract_all(INPUT_DIR)
#     columns = ["project_title", "client", "project_type"]
#     df = raw_data[columns]
#     df["project_title_and_client"] = df['project_title'] + " " + df['client']
#     return df

# if __name__ == "__main__":
#     df = main()
#     print(df.head())