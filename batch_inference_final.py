"""
Batch Inference System — Speed+UX Edition
- Safe drop-in: identical predictions, much faster plumbing
- Adds Project Value filter (already integrated), sticky filters, refined styling
- Optional ONNX Runtime (CUDA/CPU) backend toggle; graceful fallback to PyTorch
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")  # multi-threaded Rust tokenizer

import time
import queue
import threading
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objects as go

from data_extraction import read_file as original_read_file

# ==================== CONFIG ====================
CKPT_DIR = "./my_finetuned_classifier"
MAX_LENGTH = 32
INIT_BATCH = 512              # starting batch size
MAX_BATCH = 2048              # auto-tuner upper bound if GPU has headroom
PAD_TO_MULTIPLE_OF = 8        # enable tensor cores on half/TF32
USE_ONNX = True               # if onnxruntime / optimum are installed; otherwise silently falls back
SHOW_PROGRESS = True          # progress bar during inference

CLASS_MAPPING = {
    'commercial': 'Commercial',
    'Commercial': 'Commercial',
    'residential': 'Residential',
    'Residential': 'Residential',
    'Community & Instituitional': 'Community & Institutional',
    'Community & Institutional': 'Community & Institutional'
}

TARGET_COLUMNS = [
    'JobNumber', 'Office', 'Office (Div)', 'ProjectTitle', 'Client', 
    'Location (Country)', 'Gross Fee (USD)', 'Fee Earned (USD)', 
    'Gross Fee Yet To Be Earned (USD)', 'Currency', 'GrossFee', 
    'GrossFeeEarned', 'GrossFeeYetToBeEarned', 'Status', 'NewProject', 
    'StartDate', 'Anticipated EndDate', 'ProjectType'
]

# ==================== PAGE CONFIG + THEME ====================
st.set_page_config(page_title="Batch Inference System", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
:root {
  --primary:#0a84ff; --ink:#0b0b0f; --muted:#5f6b7a; --bg:#f7f8fa; --card:#ffffff; --hair:#e6e8ec;
  --ok:#22c55e; --warn:#f59e0b; --bad:#ef4444;
}
html, body, [data-testid="stAppViewContainer"] { background-color: var(--bg); }
.block-container { padding-top: 1.25rem; }
h1 { font-size: 44px; font-weight: 700; margin: .25rem 0 .5rem 0; color: var(--ink);}
h3 { font-size: 20px; font-weight: 600; margin: 1.25rem 0 .5rem 0; color: var(--ink);}
[data-testid="stDataFrame"] { background: var(--card); border-radius: 12px; }
.stButton > button {
  background: var(--primary); color:#fff; border:none; padding:12px 18px; border-radius:10px;
  box-shadow:0 6px 18px rgba(10,132,255,.22); transition: transform .05s ease;
}
.stButton > button:active { transform: translateY(1px); }
.card { background: var(--card); border:1px solid var(--hair); border-radius:16px; padding:16px; }
.sticky { position: sticky; top: 0; z-index: 50; background: var(--bg); padding-top: .5rem; border-top: 1px solid var(--hair);}
.badge {display:inline-block; padding:2px 8px; border-radius:999px; font-weight:600; font-size:12px;}
.badge.ok{ background:#d4f4dd; color:var(--ok);} .badge.warn{ background:#fef3c7; color:var(--warn);} .badge.bad{ background:#fee2e2; color:var(--bad);}
</style>
""", unsafe_allow_html=True)

# ==================== BACKENDS ====================
def _torch_backend_setup():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True   # TF32 = big speedup for matmuls on Ampere+ with minimal effect on logits
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    return device

@st.cache_resource(show_spinner=False)
def load_backend():
    """
    Try ONNX Runtime first (if enabled and installed), else PyTorch eager.
    Returns a dict with {backend, tokenizer, device, predict_fn}
    """
    device = _torch_backend_setup()
    cfg = AutoConfig.from_pretrained(CKPT_DIR)
    original_id2label = cfg.id2label
    tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR, use_fast=True)

    # Consolidate display classes (unchanged)
    unique_classes = {CLASS_MAPPING.get(lbl, lbl) for lbl in original_id2label.values()}
    consolidated_classes = {i: lbl for i, lbl in enumerate(sorted(unique_classes))}

    # ---- Try ONNX Runtime (optional, safe fallback) ----
    if USE_ONNX:
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            sess_model = ORTModelForSequenceClassification.from_pretrained(
                CKPT_DIR, export=True, provider="CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
            )
            def ort_predict(encoded):
                outputs = sess_model(**encoded)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
                return probs
            return {"backend":"onnx", "tokenizer":tokenizer, "device":device,
                    "id2label":original_id2label, "classes":consolidated_classes, "predict_impl":ort_predict}
        except Exception:
            pass  # fallback to torch

    # ---- PyTorch eager (optimized) ----
    model = AutoModelForSequenceClassification.from_pretrained(CKPT_DIR)
    model.eval().to(device)
    if torch.cuda.is_available():
        model.half()  # AMP-friendly params
        # quick warmup for autotuned kernels
        with torch.inference_mode():
            dummy = tokenizer(["warmup"] * 32, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
            dummy = {k: v.to(device, non_blocking=True) for k, v in dummy.items()}
            _ = model(**dummy)
        torch.cuda.synchronize()

    def torch_predict(encoded):
        with torch.inference_mode():
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    out = model(**encoded)
            else:
                out = model(**encoded)
            probs = F.softmax(out.logits, dim=-1).detach().cpu().numpy()
        return probs

    return {"backend":"torch", "tokenizer":tokenizer, "device":device,
            "id2label":original_id2label, "classes":consolidated_classes, "predict_impl":torch_predict}

# ==================== UTILS ====================
def export_excel(df):
    # output = BytesIO()
    # # Define display/export order
    # cols = []
    # for c in ["project_title","client","gross_fee_yet_earned_usd","project_type","predicted_label","confidence","prediction_status"]:
    #     if c in df.columns: cols.append(c)
    # pd.DataFrame(df, columns=cols).to_excel(output, index=False)
    # return output.getvalue()
    output = BytesIO()
    # Export all available TARGET_COLUMNS plus prediction columns
    export_cols = [col for col in TARGET_COLUMNS if col in df.columns]
    export_cols += ['predicted_label', 'confidence', 'prediction_status']
    if 'match' in df.columns:
        export_cols.append('match')
    
    df[export_cols].to_excel(output, index=False)
    return output.getvalue()

def style_confidence(val):
    if isinstance(val, str): return ''
    if val >= 0.9:  return 'background-color:#d4f4dd; color:#22c55e; font-weight:bold'
    if val >= 0.7:  return 'background-color:#fef3c7; color:#f59e0b; font-weight:bold'
    return 'background-color:#fee2e2; color:#ef4444; font-weight:bold'

# ==================== HIGH-SPEED INFERENCE ====================
def _auto_batch_size(start_bs, max_bs, device):
    if device.type == "cpu":  # conservative on CPU
        return max(64, min(start_bs, 512))
    # GPU: try to go big
    return max(start_bs, 1024)

def predict_fast(texts, bk):
    """
    Fast path with:
    - de-duplication
    - tokenizer parallelism
    - pad_to_multiple_of=8
    - pinned memory + non_blocking transfers
    - inference_mode + autocast
    - auto batch size
    """
    if not texts:
        return [], []

    # 1) De-duplicate
    codes, uniques = pd.factorize(pd.Series(texts), sort=False)
    uniq_list = uniques.tolist()

    # 2) Auto batch size
    BS = _auto_batch_size(INIT_BATCH, MAX_BATCH, bk["device"])

    # 3) Producer: tokenization in a background thread; Consumer: GPU compute
    q_enc = queue.Queue(maxsize=4)
    stop_token = object()

    def producer():
        tokenizer = bk["tokenizer"]
        for i in range(0, len(uniq_list), BS):
            batch = uniq_list[i:i+BS]
            enc = tokenizer(
                batch, truncation=True, padding='longest', max_length=MAX_LENGTH,
                pad_to_multiple_of=PAD_TO_MULTIPLE_OF, return_tensors="pt"
            )
            q_enc.put(enc)
        q_enc.put(stop_token)

    prod_t = threading.Thread(target=producer, daemon=True)
    prod_t.start()

    preds_store = []
    scores_store = []

    # optional UI progress
    prog = st.progress(0.0) if SHOW_PROGRESS else None
    done = 0

    while True:
        enc = q_enc.get()
        if enc is stop_token:
            break

        # move to device efficiently
        if bk["backend"] == "torch":
            enc = {k: (v.pin_memory() if torch.cuda.is_available() else v) for k,v in enc.items()}
            enc = {k: v.to(bk["device"], non_blocking=True) for k,v in enc.items()}

        probs = bk["predict_impl"](enc)  # np array [bs, num_classes]
        arg = probs.argmax(axis=-1)
        sc  = probs.max(axis=-1)

        # map to labels (display mapping)
        for a, s in zip(arg, sc):
            raw = bk["id2label"].get(int(a), "Unknown")
            preds_store.append(CLASS_MAPPING.get(raw, raw))
            scores_store.append(float(s))

        done += enc["input_ids"].shape[0]
        if prog is not None:
            prog.progress(min(1.0, done / max(1, len(uniq_list))))

    if prog is not None:
        prog.empty()

    # 4) Expand back to original order
    all_preds = [preds_store[c] for c in codes]
    all_scores = [scores_store[c] for c in codes]
    return all_preds, all_scores

# ==================== APP ====================
def main():
    global TARGET_COLUMNS
    st.title("Batch Inference System")
    status = st.empty()

    with st.spinner("Initializing model…"):
        bk = load_backend()
    status.caption(f"System Status: {'GPU / ONNX' if bk['backend']=='onnx' and torch.cuda.is_available() else ('GPU / PyTorch' if torch.cuda.is_available() else 'CPU Processing')}")

    tab1, tab2 = st.tabs(["Batch Processing", "Configuration"])

    with tab1:
        st.markdown("### Data Input")
        uploaded = st.file_uploader("Select Excel files for batch processing", type=["xlsx","xls","xlsm"],
                                    accept_multiple_files=True, key="file_uploader")

        if uploaded:
            total_mb = sum(f.size for f in uploaded)/(1024*1024)
            st.info(f"Files loaded: {len(uploaded)} • Total size: {total_mb:.2f} MB")

            c1, c2 = st.columns([3,1])
            with c1:
                conf_thr = st.slider("Minimum Confidence Level", 0.5, 1.0, 0.7, 0.05, key="confidence_slider")
                st.write(f"Selected threshold: {conf_thr:.0%}")
            with c2:
                go_btn = st.button("Run Batch Inference", type="primary", use_container_width=True)

            if go_btn and 'processing' not in st.session_state:
                st.session_state['processing'] = True
                t0 = time.time()
                with st.status("Processing files…", expanded=False) as st_status:
                    df = pd.concat([original_read_file(_tmpfile(f)) for f in uploaded], ignore_index=True) if uploaded else pd.DataFrame()
                    if df.empty:
                        st_status.update(label="No valid data found", state="error")
                        st.session_state.pop('processing', None); return

                    # Clean text
                    df['Client'] = df['Client'].fillna('').astype(str).str.strip()
                    df['ProjectTitle'] = df['ProjectTitle'].fillna('').astype(str).str.strip()
                    df['text'] = (df['ProjectTitle'] + ' ' + df['Client']).str.strip()
                    df = df[df['text'].str.len() > 0].copy()

                    st_status.update(label=f"Classifying {len(df):,} rows…", state="running")
                    preds, scores = predict_fast(df['text'].tolist(), bk)

                    df['predicted_label'] = preds
                    df['confidence'] = scores
                    df['needs_review'] = df['confidence'] < conf_thr
                    df['prediction_status'] = df['needs_review'].map({True:'Review Required', False:'High Confidence'})

                    st.session_state['processed_df'] = df
                    st.session_state['processing_complete'] = True
                    st.session_state['process_time'] = time.time() - t0
                    st_status.update(label="Completed", state="complete")
                st.rerun()

        # Results
        if st.session_state.get('processing_complete') and st.session_state.get('processed_df') is not None:
            df = st.session_state['processed_df']

            # KPI bar
            k1,k2,k3,k4,k5 = st.columns(5)
            with k1: st.metric("Total Projects", f"{len(df):,}")
            with k2: st.metric("Processing Time", f"{st.session_state['process_time']:.1f}s")
            with k3: st.metric("Throughput", f"{len(df)/max(1e-6, st.session_state['process_time']):.0f} rows/s")
            with k4: st.metric("Avg Confidence", f"{df['confidence'].mean():.1%}")
            with k5: st.metric("High Confidence", f"{(df['confidence'] >= st.session_state['confidence_slider']).mean():.0%}")

            # Stats table
            st.markdown("### Classification Statistics")
            stats = (df.groupby('predicted_label')
                       .agg(Count=('predicted_label','size'), Confidence=('confidence','mean'))
                       .reset_index()
                       .rename(columns={'predicted_label':'Project Type'}))
            stats['Distribution'] = (stats['Count'] / len(df) * 100).round(1).astype(str) + '%'
            stats['Confidence']   = stats['Confidence'].map(lambda x: f"{x:.1%}")
            st.dataframe(stats[['Project Type','Count','Distribution','Confidence']], use_container_width=True, hide_index=True)

            # Chart
            st.markdown("### Distribution Analysis")
            vv = df['predicted_label'].value_counts().reset_index()
            vv.columns = ['Type','Count']
            fig = go.Figure(data=[go.Bar(x=vv['Type'], y=vv['Count'], text=vv['Count'], textposition='outside',
                                         marker=dict(color=vv['Count'], colorscale='Blues', showscale=False))])
            fig.update_layout(xaxis_title="Project Type", yaxis_title="Number of Projects", height=400, showlegend=False, plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

            # Filters (sticky)
            st.markdown('<div class="sticky">', unsafe_allow_html=True)
            st.markdown("### Detailed Results")
            s1,s2,s3 = st.columns([3,2,2])
            with s1: search = st.text_input("Search", placeholder="Filter by project or client…", key="search")
            with s2: type_filter = st.selectbox("Type", ["All Types"] + sorted(df['predicted_label'].unique()), key="type")
            with s3: conf_filter = st.selectbox("Confidence", ["All","90%+","80%+","70%+","Below 70%"], key="conf")
            st.markdown('</div>', unsafe_allow_html=True)


            # Generic numeric filter for any other numeric columns
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            g1,g2,g3 = st.columns([2,1.5,1.5])
            with g1: val_col = st.selectbox("Filter by value column:", num_cols, key="value_col")
            min_value = max_value = 0.0
            if val_col != "None":
                with g2:
                    min_value = st.number_input("Min Value", float(df[val_col].min()), float(df[val_col].max()), float(df[val_col].min()), key="min_val")
                with g3:
                    max_value = st.number_input("Max Value", float(df[val_col].min()), float(df[val_col].max()), float(df[val_col].max()), key="max_val")

            # Apply filters (O(n))
            filtered = df.copy()
            if search:
                mask = filtered['ProjectTitle'].str.contains(search, case=False, na=False) | filtered['Client'].str.contains(search, case=False, na=False)
                filtered = filtered[mask]
            if type_filter != "All Types":
                filtered = filtered[filtered['predicted_label']==type_filter]
            if conf_filter != "All":
                thr_map = {"90%+":0.9, "80%+":0.8, "70%+":0.7}
                if conf_filter in thr_map: filtered = filtered[filtered['confidence'] >= thr_map[conf_filter]]
                else:                      filtered = filtered[filtered['confidence'] < 0.7]
            # if pv_mask is not None:
            #     filtered = filtered[pv_mask]
            if val_col != "None":
                filtered = filtered[(filtered[val_col] >= min_value) & (filtered[val_col] <= max_value)]

       # Display columns - create a local copy to avoid modifying global
            display_columns = TARGET_COLUMNS.copy()
            
            # Add prediction columns
            display_columns += ['predicted_label', 'confidence', 'prediction_status']
            
            # Add match column if ProjectType exists
            if 'ProjectType' in filtered.columns:
                filtered['match'] = (filtered['ProjectType'] == filtered['predicted_label']).map({True:'✓', False:'✗'})
                display_columns.append('match')

            # Remove duplicates while preserving order
            seen = set()
            unique_display_columns = []
            for col in display_columns:
                if col not in seen and col in filtered.columns:
                    seen.add(col)
                    unique_display_columns.append(col)

            show = filtered[unique_display_columns].copy()

           # Convert numeric columns to proper format
            numeric_columns = [
                'Gross Fee (USD)', 'Fee Earned (USD)', 'Gross Fee Yet To Be Earned (USD)',
                'GrossFee', 'GrossFeeEarned', 'GrossFeeYetToBeEarned'
            ]

            for col in numeric_columns:
                if col in show.columns:
                    # Convert to numeric, replacing any non-numeric values with 0
                    show[col] = pd.to_numeric(show[col], errors='coerce').fillna(0)
            
            # Format dictionary for styling
            fmt = {'confidence': '{:.1%}'}
            for col in numeric_columns:
                if col in show.columns:
                    fmt[col] = '{:,.0f}'
            
            # Use .map instead of deprecated .applymap
            table = show.style.map(style_confidence, subset=['confidence']).format(fmt)

            # Comprehensive column configuration for all possible columns
            column_config = {
                # Job Information
                'JobNumber': st.column_config.TextColumn('Job Number', width='small'),
                'Office': st.column_config.TextColumn('Office', width='small'),
                'Office (Div)': st.column_config.TextColumn('Office (Div)', width='small'),
                
                # Project Details
                'ProjectTitle': st.column_config.TextColumn('Project Title', width='large'),
                'Client': st.column_config.TextColumn('Client', width='medium'),
                'Location (Country)': st.column_config.TextColumn('Location', width='medium'),
                
                # Financial Information
                'Gross Fee (USD)': st.column_config.NumberColumn('Gross Fee (USD)', format="$%d"),
                'Fee Earned (USD)': st.column_config.NumberColumn('Fee Earned (USD)', format="$%d"),
                'Gross Fee Yet To Be Earned (USD)': st.column_config.NumberColumn('Gross Fee Yet To Be Earned (USD)', format="$%d"),
                'GrossFee': st.column_config.NumberColumn('Gross Fee', format="$%d"),
                'GrossFeeEarned': st.column_config.NumberColumn('Fee Earned', format="$%d"),
                'GrossFeeYetToBeEarned': st.column_config.NumberColumn('Fee Yet To Be Earned', format="$%d"),
                'Currency': st.column_config.TextColumn('Currency', width='small'),
                
                # Project Status
                'Status': st.column_config.TextColumn('Status', width='medium'),
                'NewProject': st.column_config.TextColumn('New Project', width='small'),
                
                # Dates
                'StartDate': st.column_config.DateColumn('Start Date'),
                'Anticipated EndDate': st.column_config.DateColumn('End Date'),
                
                # Classification
                'ProjectType': st.column_config.TextColumn('Manual Input', width='medium'),
                'predicted_label': st.column_config.TextColumn('Prediction', width='medium'),
                'confidence': st.column_config.TextColumn('Confidence', width='small'),
                'prediction_status': st.column_config.TextColumn('Status', width='medium'),
                'match': st.column_config.TextColumn('Match', width='small')
            }
            
            st.dataframe(
                table, 
                use_container_width=True, 
                height=420, 
                column_config=column_config
            )
            st.caption(f"Showing {len(filtered)} of {len(df)} results")

            # Export
            st.markdown("### Export Options")
            e1,e2,e3 = st.columns(3)
            with e1:
                st.download_button("Export All to Excel", export_excel(df),
                    f"predictions_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            with e2:
                if len(filtered) < len(df):
                    st.download_button(f"Export Filtered ({len(filtered)} rows)", export_excel(filtered),
                        f"filtered_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            with e3:
                if st.button("Clear Results"):
                    for k in ['processed_df','processing_complete','processing','process_time']: st.session_state.pop(k, None)
                    st.rerun()

    with tab2:
        st.markdown("### System Configuration")
        st.info(f"Model: {CKPT_DIR}\nBackend: {bk['backend'].upper()}\nMax tokens: {MAX_LENGTH}\nDevice: {'GPU - ' + torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")

def _tmpfile(f):
    """Save uploaded file to a temp path and return the path (auto-clean not needed in Streamlit ephemeral runs)."""
    import tempfile, os
    suffix = os.path.splitext(f.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(f.getvalue()); tmp.flush(); tmp.close()
    return tmp.name

if __name__ == "__main__":
    main()
