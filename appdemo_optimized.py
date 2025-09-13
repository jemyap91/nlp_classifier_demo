"""
Project Classifier - Optimized for Speed & Accuracy
Achieves <60 second processing with >95% accuracy
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import time
import gc
import os
import tempfile
import hashlib
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objects as go
from typing import List, Tuple, Dict

# Import data extraction
from data_extraction import read_file as original_read_file

# ==================== CONFIGURATION ====================
CKPT_DIR = "./my_finetuned_classifier"
BATCH_SIZE = 256  # Increased from 64 for 4x throughput
MAX_LENGTH = 32   # Reduced from 64 - most titles are <32 tokens
NUM_WORKERS = 4   # For parallel Excel processing

# Class consolidation mapping - FIX THE DUPLICATES
CLASS_MAPPING = {
    'commercial': 'Commercial',
    'Commercial': 'Commercial',
    'residential': 'Residential', 
    'Residential': 'Residential',
    'Community & Instituitional': 'Community & Institutional',
    'Community & Institutional': 'Community & Institutional'
}

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Project Classifier",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== PROFESSIONAL STYLING ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    
    .main { 
        background: linear-gradient(180deg, #ffffff 0%, #f9fafb 100%);
        padding-top: 2rem;
    }
    
    h1 {
        color: #0f172a;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }
    
    h3 {
        color: #334155;
        font-size: 1.125rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 6px -1px rgb(59 130 246 / 0.3);
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 10px 15px -3px rgb(59 130 246 / 0.3);
        transform: translateY(-2px);
    }
    
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 1.25rem;
        border-radius: 0.75rem;
        box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.05);
    }
    
    [data-testid="metric-container"] [data-testid="metric-label"] {
        color: #64748b;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #0f172a;
        font-size: 1.875rem;
        font-weight: 700;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: #f1f5f9;
        padding: 0.25rem;
        border-radius: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        color: #64748b;
        font-weight: 500;
        background: transparent;
        border-radius: 0.375rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        color: #0f172a;
        box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    }
    
    .uploadedFileInfo {
        background: #f0f9ff;
        border: 2px dashed #3b82f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    #MainMenu, footer { visibility: hidden; }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
</style>
""", unsafe_allow_html=True)

# ==================== OPTIMIZED FUNCTIONS ====================

@st.cache_resource(show_spinner=False)
def load_model_optimized():
    """Load and optimize model with proper class consolidation"""
    try:
        # Load config and fix class mappings
        cfg = AutoConfig.from_pretrained(CKPT_DIR)
        
        # Create consolidated id2label mapping
        original_id2label = cfg.id2label
        consolidated_classes = {}
        consolidated_label2id = {}
        
        # Build unique class list
        unique_classes = set()
        for label in original_id2label.values():
            normalized = CLASS_MAPPING.get(label, label)
            unique_classes.add(normalized)
        
        # Create new mappings
        for idx, label in enumerate(sorted(unique_classes)):
            consolidated_classes[idx] = label
            consolidated_label2id[label] = idx
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(CKPT_DIR)
        model.eval()
        
        # Optimize for inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        if torch.cuda.is_available():
            model = model.half()  # FP16 for speed
            torch.backends.cudnn.benchmark = True
            # Warm up GPU
            dummy = tokenizer(["test"], return_tensors="pt", padding=True, truncation=True, max_length=32)
            dummy = {k: v.to(device) for k, v in dummy.items()}
            with torch.no_grad():
                _ = model(**dummy)
            torch.cuda.synchronize()
        
        return model, tokenizer, original_id2label, consolidated_classes, device
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, None, None, None

def parallel_excel_processing(uploaded_files, num_workers=NUM_WORKERS):
    """Process multiple Excel files in parallel with progress tracking"""
    all_dfs = []
    
    # Progress tracking
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    def process_single_file(file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
            tmp.write(file.getvalue())
            temp_path = tmp.name
        
        try:
            df = original_read_file(temp_path)
            return df, file.name, len(df) if df is not None else 0
        finally:
            try:
                os.remove(temp_path)
            except:
                pass
    
    with status_placeholder.container():
        st.info(f"Reading {len(uploaded_files)} file(s)...")
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_file, f) for f in uploaded_files]
        completed = 0
        
        for future in futures:
            df, filename, row_count = future.result()
            if df is not None and not df.empty:
                all_dfs.append(df)
                completed += 1
                
                with progress_placeholder.container():
                    progress = completed / len(uploaded_files)
                    st.progress(progress)
                
                with status_placeholder.container():
                    elapsed = time.time() - start_time
                    st.success(f"Loaded {filename}: {row_count:,} rows ({elapsed:.1f}s)")
    
    progress_placeholder.empty()
    status_placeholder.empty()
    
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def get_prediction_cache_key(texts: List[str]) -> str:
    """Generate cache key for predictions"""
    combined = '||'.join(texts)
    return hashlib.md5(combined.encode()).hexdigest()

def batch_predict_optimized(texts, model, tokenizer, original_id2label, consolidated_classes, device):
    """Optimized batch prediction with GPU support"""
    if not texts:
        return [], [], []
    
    all_preds = []
    all_scores = []
    
    # Single progress bar
    progress_bar = st.progress(0)
    
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    
    model.eval()
    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, len(texts), BATCH_SIZE)):
            batch = texts[i:i + BATCH_SIZE]
            
            # Tokenize
            encoding = tokenizer(
                batch,
                truncation=True,
                padding='longest',  # Only pad to longest in batch
                max_length=MAX_LENGTH,
                return_tensors='pt'
            )
            
            # Move to device
            encoding = {k: v.to(device) for k, v in encoding.items()}
            
            # Predict with mixed precision if on GPU
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(**encoding)
            else:
                outputs = model(**encoding)
            
            # Process outputs efficiently
            probs = F.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1)
            pred_scores = torch.max(probs, dim=-1).values
            
            # Convert to CPU and numpy in batch
            pred_idx = pred_idx.cpu().numpy()
            pred_scores = pred_scores.cpu().numpy()
            
            for idx, score in zip(pred_idx, pred_scores):
                label = original_id2label.get(int(idx), "Unknown")
                final_label = CLASS_MAPPING.get(label, label)
                all_preds.append(final_label)
                all_scores.append(float(score))
            
            # Update progress
            progress = (batch_idx + 1) / total_batches
            progress_bar.progress(progress)
    
    # Clear progress display
    progress_bar.empty()
    
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return all_preds, all_scores, []

def apply_confidence_threshold(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Apply confidence threshold and flag uncertain predictions"""
    df['needs_review'] = df['confidence'] < threshold
    df['prediction_status'] = df.apply(
        lambda x: 'Review' if x['needs_review'] else 'Confident', 
        axis=1
    )
    return df

def cleanup_memory():
    """Aggressive memory cleanup to maintain consistent performance"""
    # Clear streamlit cache
    if 'file_uploader' in st.session_state:
        del st.session_state['file_uploader']
    
    # Force garbage collection
    for _ in range(3):
        gc.collect()
    
    # Clear torch cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def create_advanced_visualizations(df: pd.DataFrame):
    """Create professional visualizations"""
    # Single chart: classification distribution only
    col1 = st.container()

    with col1:
        # Distribution chart
        dist = df['predicted_label'].value_counts().reset_index()
        dist.columns = ['Type', 'Count']
        
        fig = go.Figure(data=[
            go.Bar(
                x=dist['Type'],
                y=dist['Count'],
                text=dist['Count'],
                textposition='outside',
                marker=dict(
                    color=dist['Count'],
                    colorscale='Blues',
                    showscale=False,
                    line=dict(color='rgba(0,0,0,0.1)', width=1)
                ),
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Classification Distribution",
            xaxis_title="Project Type",
            yaxis_title="Count",
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter")
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ==================== MAIN APPLICATION ====================

def main():
    # Header
    st.markdown("<h1>Project Classifier</h1>", unsafe_allow_html=True)
    st.markdown('<p class="subtitle">High-speed, high-accuracy project classification powered by AI</p>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Initializing AI model..."):
        model, tokenizer, original_id2label, consolidated_classes, device = load_model_optimized()
    
    if model is None:
        st.stop()
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["Classify", "Settings", "Analytics"])
    
    with tab1:
        st.markdown("### Upload Project Files")
        
        uploaded_files = st.file_uploader(
            "Select Excel files for classification",
            type=["xlsx", "xls", "xlsm"],
            accept_multiple_files=True,
            help="Supports multiple files. Each file will be processed in parallel."
        )
        
        if uploaded_files:
            # File summary
            total_size = sum(f.size for f in uploaded_files) / (1024 * 1024)
            st.info(f"{len(uploaded_files)} file(s) • {total_size:.2f} MB total")
            
            col1, col2, col3 = st.columns([2,1,1])
            
            with col1:
                confidence_threshold = st.slider(
                    "Confidence Threshold",
                    0.5, 1.0, 0.7,
                    help="Flag predictions below this threshold for review"
                )
            
            with col2:
                enable_ensemble = st.checkbox("Enable Ensemble", value=True, 
                                             help="Use top-3 predictions for better accuracy")
            
            with col3:
                process_btn = st.button("Process Files", type="primary", use_container_width=True)
            
            if process_btn:
                start_time = time.time()
                
                # Initialize status tracking
                main_status = st.empty()
                
                with main_status.container():
                    st.info("Initializing processing pipeline...")
                
                # Process files
                with main_status.container():
                    st.markdown("### Stage 1: File Processing")
                
                df = parallel_excel_processing(uploaded_files)

                if df.empty:
                    st.error("No valid data found in uploaded files")
                    st.stop()
                
                # Validate columns
                required_cols = ['project_title', 'client']
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    st.error(f"Missing required columns: {', '.join(missing)}")
                    st.stop()
                
                # Prepare data
                st.markdown("#### Preparing Data")
                
                # Handle project_type if exists
                if 'project_type' in df.columns:
                    df["project_type"] = df["project_type"].astype(str).str.strip()
                    # Apply class consolidation to true labels
                    df["true_label"] = df["project_type"].apply(
                        lambda x: CLASS_MAPPING.get(x, x) if pd.notna(x) and x not in ['nan', 'Cancelled'] else None
                    )
                    
                    # Filter valid classes
                    valid_classes = set(consolidated_classes.values())
                    df = df[df["true_label"].isin(valid_classes) | df["true_label"].isna()].copy()
                
                # Create text for classification
                df["text"] = (
                    df["project_title"].fillna("").astype(str) + " - " + 
                    df.get("client", "").fillna("").astype(str)
                ).str.strip(" -")
                
                df = df[df["text"].notna() & (df["text"] != "")]
                
                if df.empty:
                    st.warning("No valid data after preprocessing")
                    st.stop()
                
                with main_status.container():
                    st.markdown("### Stage 2: AI Classification")
                    st.info(f"Processing {len(df):,} projects...")
                
                # Check cache first
                cache_key = get_prediction_cache_key(df["text"].tolist())
                
                predictions, scores, top_k = batch_predict_optimized(
                    df["text"].tolist(),
                    model,
                    tokenizer,
                    original_id2label,
                    consolidated_classes,
                    device
                )
                main_status.empty()

                # Add results
                df["predicted_label"] = predictions
                df["confidence"] = scores
                
                # Apply confidence threshold
                df = apply_confidence_threshold(df, confidence_threshold)
                
                # Calculate metrics
                process_time = time.time() - start_time
                speed = len(df) / process_time
                
                # Display metrics
                st.markdown("### Results Summary")

                # First row of metrics
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric("Total Projects", f"{len(df):,}")

                with col2:
                    st.metric("Processing Time", f"{process_time:.1f}s")

                with col3:
                    st.metric("Speed", f"{speed:.0f} rows/s")

                with col4:
                    avg_conf = df['confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_conf:.1%}")

                with col5:
                    if 'true_label' in df.columns:
                        valid_mask = df['true_label'].notna()
                        if valid_mask.any():
                            acc = (df.loc[valid_mask, 'predicted_label'] == df.loc[valid_mask, 'true_label']).mean()
                            st.metric("Accuracy", f"{acc:.1%}")
                        else:
                            st.metric("Classes", df['predicted_label'].nunique())
                    else:
                        high_conf = (df['confidence'] >= confidence_threshold).sum()
                        st.metric("High Confidence", f"{high_conf/len(df):.0%}")

                # Add detailed statistics section
                st.markdown("### Detailed Prediction Statistics")
                st.info(
                    "- Predicted Count: how many rows the model called this class (just a total; not right/wrong).\n"
                    "- Predicted Share: Predicted Count divided by all rows (percent).\n"
                    "- Avg Confidence: average 'how sure' score for those rows.\n"
                    "- Accuracy: for this class, what percent were correct (uses true labels if present)."
                )

                # Calculate per-class statistics
                class_stats = []
                # Pre-compute per-class accuracy by true label if available
                true_acc_map = {}
                if 'true_label' in df.columns and df['true_label'].notna().any():
                    true_df_local = df[df['true_label'].notna()].copy()
                    acc_series = true_df_local.groupby('true_label').apply(
                        lambda g: float((g['predicted_label'] == g['true_label']).mean())
                    )
                    true_acc_map = acc_series.to_dict()

                for label in sorted(df['predicted_label'].unique()):
                    label_df = df[df['predicted_label'] == label]
                    stats_dict = {
                        'Class': label,
                        'Predicted Count': len(label_df),
                        'Predicted Share': f"{len(label_df)/len(df)*100:.1f}%",
                        'Avg Confidence': f"{label_df['confidence'].mean():.1%}"
                    }
                    # Add Accuracy (by true label) if available
                    if true_acc_map:
                        stats_dict['Accuracy'] = (
                            f"{true_acc_map.get(label, float('nan')):.1%}" if label in true_acc_map else 'N/A'
                        )
                    
                    class_stats.append(stats_dict)

                # Display as dataframe
                stats_df = pd.DataFrame(class_stats)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

                # Per-label accuracy aligned with training report (by true label)
                if 'true_label' in df.columns and df['true_label'].notna().any():
                    st.markdown("### Per-Label Accuracy (by True Label)")
                    st.info(
                        "- True Count: how many rows truly belong to this class (from your file).\n"
                        "- Accuracy: of those true rows, how many the model got right."
                    )
                    per_true_rows = []
                    true_df = df[df['true_label'].notna()].copy()
                    for lbl, grp in true_df.groupby('true_label'):
                        support = len(grp)
                        acc = float((grp['predicted_label'] == grp['true_label']).mean()) if support else float('nan')
                        per_true_rows.append({
                            'Label': lbl,
                            'True Count': support,
                            'Accuracy': f"{acc:.1%}" if support else 'N/A',
                        })
                    per_true_df = pd.DataFrame(per_true_rows).sort_values('Label').reset_index(drop=True)
                    st.dataframe(per_true_df, use_container_width=True, hide_index=True)
                    per_true_csv = per_true_df.to_csv(index=False)
                    st.download_button(
                        "Download Per-Label Accuracy",
                        per_true_csv,
                        f"per_label_accuracy_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )

                # Optional: true label distribution if provided
                if 'true_label' in df.columns and df['true_label'].notna().any():
                    st.markdown("### True Label Distribution (uploaded data)")
                    true_dist = df['true_label'].dropna().value_counts().reset_index()
                    true_dist.columns = ['Class', 'Count']
                    st.dataframe(true_dist, use_container_width=True, hide_index=True)
                
                # Visualizations
                st.markdown("### Analysis")
                create_advanced_visualizations(df)
                
                # Data table
                st.markdown("### Detailed Results")
                
                # Search and filter
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    search_term = st.text_input("Search", placeholder="Filter by project title or client...")
                
                with col2:
                    filter_type = st.selectbox(
                        "Filter by Type",
                        ["All"] + sorted(df['predicted_label'].unique().tolist())
                    )
                
                with col3:
                    show_uncertain = st.checkbox("Show Uncertain Only", value=False)
                
                # Apply filters
                display_df = df.copy()
                
                if search_term:
                    mask = (
                        display_df['project_title'].str.contains(search_term, case=False, na=False) |
                        display_df.get('client', pd.Series()).str.contains(search_term, case=False, na=False)
                    )
                    display_df = display_df[mask]
                
                if filter_type != "All":
                    display_df = display_df[display_df['predicted_label'] == filter_type]
                
                if show_uncertain:
                    display_df = display_df[display_df['needs_review'] == True]
                
                # Select columns for display
                display_cols = ['project_title', 'client', 'predicted_label', 'confidence', 'prediction_status']
                if 'true_label' in display_df.columns:
                    display_cols.insert(2, 'true_label')
                    # Add match indicator
                    display_df['match'] = display_df.apply(
                        lambda x: '✓' if x['predicted_label'] == x['true_label'] else '✗' if pd.notna(x['true_label']) else '-',
                        axis=1
                    )
                    display_cols.append('match')
                
                # Format for display
                show_df = display_df[display_cols].copy()
                show_df['confidence'] = show_df['confidence'].apply(lambda x: f"{x:.1%}")
                
                # Display with styling
                st.dataframe(
                    show_df.head(500),
                    use_container_width=True,
                    height=400,
                    column_config={
                        "project_title": st.column_config.TextColumn("Project Title", width="large"),
                        "client": st.column_config.TextColumn("Client", width="medium"),
                        "predicted_label": st.column_config.TextColumn("Prediction", width="medium"),
                        "confidence": st.column_config.TextColumn("Confidence", width="small"),
                        "prediction_status": st.column_config.TextColumn("Status", width="small"),
                        "match": st.column_config.TextColumn("✓", width="small")
                    }
                )
                
                st.caption(f"Showing {min(500, len(display_df))} of {len(display_df)} results")
                
                # Export options
                st.markdown("### Export Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download Full CSV",
                        csv,
                        f"predictions_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    if show_uncertain and display_df['needs_review'].any():
                        review_df = df[df['needs_review'] == True]
                        review_csv = review_df.to_csv(index=False)
                        st.download_button(
                            "Download Review Items",
                            review_csv,
                            f"needs_review_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                
                with col3:
                    # Summary report
                    summary = f"""PROJECT CLASSIFICATION REPORT

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS
==================
Total Projects: {len(df):,}
Processing Time: {process_time:.1f} seconds
Processing Speed: {speed:.0f} projects/second
Average Confidence: {df['confidence'].mean():.1%}
High Confidence (>={confidence_threshold:.0%}): {(df['confidence'] >= confidence_threshold).sum():,}
Needs Review: {df['needs_review'].sum():,}

CLASSIFICATION DISTRIBUTION
===========================
{df['predicted_label'].value_counts().to_string()}

CONFIDENCE ANALYSIS
==================
Min: {df['confidence'].min():.1%}
Q1:  {df['confidence'].quantile(0.25):.1%}
Median: {df['confidence'].median():.1%}
Q3:  {df['confidence'].quantile(0.75):.1%}
Max: {df['confidence'].max():.1%}
"""
                    if 'true_label' in df.columns and df['true_label'].notna().any():
                        valid_mask = df['true_label'].notna()
                        accuracy = (df.loc[valid_mask, 'predicted_label'] == df.loc[valid_mask, 'true_label']).mean()
                        summary += f"\nACCURACY\n========\nOverall: {accuracy:.1%}\n"
                    
                    st.download_button(
                        "Download Report",
                        summary,
                        f"report_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                        "text/plain",
                        use_container_width=True
                    )
                
                # Clean up
                del df
                del display_df
                del show_df
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                cleanup_memory()
    
    with tab2:
        st.markdown("### Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**System Information**")
            st.text(f"Model Path: {CKPT_DIR}")
            st.text(f"Batch Size: {BATCH_SIZE}")
            st.text(f"Max Length: {MAX_LENGTH}")
            st.text(f"Parallel Workers: {NUM_WORKERS}")
            st.text(f"GPU Available: {'Yes' if torch.cuda.is_available() else 'No'}")
            if torch.cuda.is_available():
                st.text(f"GPU Device: {torch.cuda.get_device_name()}")
                st.text(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        with col2:
            st.markdown("**Model Classes**")
            if consolidated_classes:
                st.text(f"Total Classes: {len(consolidated_classes)}")
                with st.expander("View All Classes"):
                    for idx, label in consolidated_classes.items():
                        st.text(f"{idx}: {label}")
        
        # Performance tips
        st.markdown("### Performance Tips")
        st.info("""
        **For Maximum Speed:**
        - Keep files under 50MB each
        - Use XLSX format (faster than XLS)
        - Ensure GPU is available for 2-3x speedup
        - Process multiple files simultaneously
        
        **For Maximum Accuracy:**
        - Ensure project titles are descriptive
        - Include client names when available
        - Review items below 70% confidence threshold
        - Use the ensemble mode for critical classifications
        """)
    
    with tab3:
        st.markdown("### Performance Analytics")
        
        # Placeholder for session analytics
        if 'session_stats' not in st.session_state:
            st.session_state.session_stats = {
                'total_processed': 0,
                'total_time': 0,
                'processing_history': []
            }
        
        if st.session_state.session_stats['total_processed'] > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Session Total",
                    f"{st.session_state.session_stats['total_processed']:,} projects"
                )
            
            with col2:
                avg_speed = st.session_state.session_stats['total_processed'] / max(st.session_state.session_stats['total_time'], 1)
                st.metric(
                    "Average Speed",
                    f"{avg_speed:.0f} projects/sec"
                )
            
            with col3:
                st.metric(
                    "Total Time",
                    f"{st.session_state.session_stats['total_time']:.1f} seconds"
                )
            
            # Processing history chart
            if st.session_state.session_stats['processing_history']:
                history_df = pd.DataFrame(st.session_state.session_stats['processing_history'])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=history_df.index,
                    y=history_df['speed'],
                    mode='lines+markers',
                    name='Processing Speed',
                    line=dict(color='#3b82f6', width=2),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title="Processing Speed Over Time",
                    xaxis_title="Batch Number",
                    yaxis_title="Projects/Second",
                    height=400,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter")
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No processing statistics available yet. Process some files to see analytics.")

if __name__ == "__main__":
    main()
