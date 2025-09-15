# predict_only.py — run inference on a 70/30 split and write CSVs (no retraining)

import os
import numpy as np
import pandas as pd
from pathlib import Path

from datasets import Dataset, Features, ClassLabel, Value
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoConfig, Trainer, DataCollatorWithPadding
)

from data_extraction_model_training import extract_all  # your existing extractor

CKPT_DIR = "./my_finetuned_classifier"                   # saved model folder
CACHE_PARQUET = Path("cache/cleaned_dataset.parquet")    # your cleaned cache
MODEL_NAME = "distilbert-base-uncased"                   # tokenizer base

def load_dataframe():
    """Load the same data you trained on (prefer parquet cache)."""
    if CACHE_PARQUET.exists():
        df = pd.read_parquet(CACHE_PARQUET)
    else:
        raw = extract_all()
        df = raw[["project_title", "client", "project_type"]].copy()
        df["text"] = (
            df["project_title"].fillna("").astype(str) + " - " +
            df["client"].fillna("").astype(str)
        ).str.strip(" -")
        # match your training filters
        df = df.dropna(subset=["project_type"]).copy()
        df = df[~df["project_type"].isin([None, "nan", "Cancelled"])]
        df["project_type"] = df["project_type"].astype(str)

    if "text" not in df.columns:
        df["text"] = (
            df["project_title"].fillna("").astype(str) + " - " +
            df["client"].fillna("").astype(str)
        ).str.strip(" -")

    df["project_type"] = df["project_type"].astype(str).str.strip()
    return df

def main():
    # 1) Load data
    df = load_dataframe()

    # 2) Load label mapping from checkpoint (SINGLE SOURCE OF TRUTH)
    cfg = AutoConfig.from_pretrained(CKPT_DIR)
    id2label = {int(k): v for k, v in cfg.id2label.items()}
    label2id = {v: int(k) for k, v in id2label.items()}

    # Keep only rows with labels the model knows
    df = df[df["project_type"].isin(label2id)].copy()
    # Assign numeric labels in the model's exact id order
    df["label"] = df["project_type"].map(label2id).astype(int)

    # For stratified split, drop singleton classes if any
    vc = df["label"].value_counts()
    singletons = vc[vc < 2].index.tolist()
    if singletons:
        print(f"⚠️ Removing {len(singletons)} singleton class(es) so stratify can run: {singletons}")
        df = df[~df["label"].isin(singletons)].reset_index(drop=True)

    # Sanity checks
    num_classes = len(id2label)
    assert df["label"].min() >= 0
    assert df["label"].max() <= num_classes - 1

    # 3) Build HF dataset with the SAME class order as the checkpoint
    #    IMPORTANT: ClassLabel names must be in id order 0..N-1
    class_names_in_id_order = [id2label[i] for i in range(num_classes)]
    features = Features({
        "text": Value("string"),
        "label": ClassLabel(names=class_names_in_id_order),
    })

    dataset = Dataset.from_pandas(
        df[["text", "label"]].reset_index(drop=True),
        features=features,
        preserve_index=False
    )

    # 4) 70/30 stratified split
    train_test = dataset.train_test_split(
        test_size=0.3, seed=42, stratify_by_column="label"
    )

    # 5) Tokenize + predict with saved model
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    def enc(b): return tok(b["text"], truncation=True)
    tokenised = train_test.map(enc, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tok)

    model = AutoModelForSequenceClassification.from_pretrained(CKPT_DIR)
    trainer = Trainer(model=model, tokenizer=tok, data_collator=collator)

    print("⏳ Predicting on test split...")
    out = trainer.predict(tokenised["test"])
    logits = out.predictions
    pred_ids = np.argmax(logits, axis=1)
    true_ids = out.label_ids

    texts = train_test["test"]["text"]
    true_labels = [id2label[int(i)] for i in true_ids]
    pred_labels = [id2label[int(i)] for i in pred_ids]

    # 6) Save outputs
    os.makedirs("./results", exist_ok=True)

    # Predictions CSV
    preds_path = "./results/predictions.csv"
    pd.DataFrame({
        "text": texts,
        "original_label": true_labels,
        "predicted_label": pred_labels,
    }).to_csv(preds_path, index=False)

    # Confusion matrix CSV
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_ids, pred_ids, labels=list(range(num_classes)))
    cm_df = pd.DataFrame(
        cm,
        index=[f"true:{id2label[i]}" for i in range(num_classes)],
        columns=[f"pred:{id2label[i]}" for i in range(num_classes)]
    )
    cm_path = "./results/confusion_matrix.csv"
    cm_df.to_csv(cm_path, index=True)

    # Quick demo prints
    print("✅ Saved:", os.path.abspath(preds_path))
    print("✅ Saved:", os.path.abspath(cm_path))
    print("🔎 Sample predictions:")
    for s in ["Hobart Airport Expansion - Planned Performance Pty Ltd",
              "Community hospital redevelopment",
              "Luxury resort in Bali"]:
        enc_single = tok(s, return_tensors="pt", truncation=True)
        pred = model(**enc_single).logits.argmax(dim=1).item()
        print(f"  {s}  ->  {id2label[pred]}")

if __name__ == "__main__":
    main()
