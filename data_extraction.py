import os
import re
from typing import List
import numpy as pd
import pandas as pd
import numpy as np

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
    - 'Client'        -> 'client'
    - 'Gross Fee (USD)' -> 'grossfeeusd'
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

# --- NEW: aliases + parser for Project Value (Gross Fee, USD)
FEE_ALIASES_USD = {
    "grossfeeusd", "projectvalueusd", "contractvalueusd", "feeusd",
    "grossfeesusd", "grossfeeinusd", "grossfeeus$", "grossfee$", "usdfee"
}
FEE_ALIASES_GENERIC = {"grossfee", "projectvalue", "contractvalue", "fee"}

def _to_number(series: pd.Series) -> pd.Series:
    # Keep digits, sign, and decimal; turn "(123)" into "-123"
    s = series.astype(str).fillna("")
    s = s.str.replace(r"\(([\d\.,]+)\)", r"-\1", regex=True)
    s = s.str.replace(r"[^\d\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def read_sheet(path, sheet):
    raw = pd.read_excel(path, sheet_name=sheet, header=None, dtype=str, engine=None)
    if raw.empty:
        return pd.DataFrame(columns=["project_title", "project_type", "client", "gross_fee_usd",
                                     "source_file", "sheet_name", "excel_row"])

    hdr = find_header_row(raw)
    if hdr is None:
        return pd.DataFrame(columns=["project_title", "project_type", "client", "gross_fee_usd",
                                     "source_file", "sheet_name", "excel_row"])

    # set header and slice data rows
    cols = raw.iloc[hdr].tolist()
    data = raw.iloc[hdr+1:].reset_index(drop=True)
    data.columns = cols

    # find title/type/client columns
    title_col = None
    type_col  = None
    client_col = None
    fee_col = None
    currency_col = None

    for c in data.columns:
        nc = normalize(c)
        if title_col is None and nc == "projecttitle":
            title_col = c
        if type_col is None and nc == "projecttype":
            type_col = c
        if client_col is None and nc == "client":
            client_col = c
        # detect fee & currency
        if fee_col is None and (nc in FEE_ALIASES_USD or nc in FEE_ALIASES_GENERIC):
            fee_col = c
        if currency_col is None and nc in {"currency", "curr"}:
            currency_col = c

    if title_col is None:
        return pd.DataFrame(columns=["project_title", "project_type", "client", "gross_fee_usd",
                                     "source_file", "sheet_name", "excel_row"])

    out = pd.DataFrame()
    out["project_title"] = data[title_col].astype(str).str.strip()
    out["project_type"]  = data[type_col].astype(str).str.strip() if type_col in data.columns else None
    out["client"]        = data[client_col].astype(str).str.strip() if client_col in data.columns else None

    # keep non-empty titles only
    out = out[out["project_title"].replace({"": None, "nan": None}).notna()]

    # provenance; Excel’s visible row number = header_row + 2 + df index
    out["excel_row"]   = out.index + (hdr + 2)
    out["source_file"] = os.path.basename(path)
    out["sheet_name"]  = sheet

    # --- NEW: derive numeric Project Value in USD
    out["gross_fee_usd"] = np.nan
    if fee_col is not None:
        fees = _to_number(data[fee_col])
        fee_name_norm = normalize(fee_col)
        if ("usd" in fee_name_norm) or (fee_name_norm in FEE_ALIASES_USD):
            out["gross_fee_usd"] = fees
        elif currency_col is not None:
            curr = data[currency_col].astype(str).str.upper()
            is_usd = curr.str.contains("USD") | curr.str.contains("US$")
            out["gross_fee_usd"] = fees.where(is_usd, np.nan)
        else:
            # Currency unknown: keep numeric value (you can change to np.nan if you want strict USD only)
            out["gross_fee_usd"] = fees

    return out

def read_file(path):
    try:
        xl = pd.ExcelFile(path, engine=None)
    except Exception:
        return pd.DataFrame(columns=["project_title", "project_type", "client", "gross_fee_usd",
                                     "source_file", "sheet_name", "excel_row"])
    frames = []
    for s in xl.sheet_names:
        try:
            frames.append(read_sheet(path, s))
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["project_title", "project_type", "client", "gross_fee_usd",
                 "source_file", "sheet_name", "excel_row"]
    )

def extract_all(input_dir=INPUT_DIR):
    dfs = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in VALID_EXTS:
                dfs.append(read_file(os.path.join(root, f)))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(
        columns=["project_title", "project_type", "client", "gross_fee_usd",
                 "source_file", "sheet_name", "excel_row"]
    )
