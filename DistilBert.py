#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import evaluate

from data_extraction_model_training import extract_all #importing from data extraction script

from datasets import Dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from transformers import pipeline
from pathlib import Path

def main():
    print("CWD:", Path.cwd().resolve()) #just a check to see my current working directory
    CLEANED_CACHE = Path("cache/cleaned_dataset.parquet") #save cleaned up parquet data
    df = None # Initialize df outside the if/else block (temporary space)

    # ---- FAST PATH: use cached cleaned parquet dataset if it exists ----
    if CLEANED_CACHE.exists():
        print("Loading cleaned dataset from:", CLEANED_CACHE.resolve())
        try:
            df = pd.read_parquet(CLEANED_CACHE) #load it as a pandas df
        except Exception as e:
            print(f"❌ Error loading Parquet file: {e}")
            print("Falling back to full data extraction...")
            df = None # Fallback to the 'else' block
        if "text" not in df.columns:
            df["text"] = (
                df["project_title"].fillna("").astype(str)
                + " - "
                + df["client"].fillna("").astype(str)
            ).str.strip(" -")

    
    # ---- SLOW PATH: if file doesn't exist, do full extraction + cleaning ----
    if df is None:
        print("Step 1: Extracting data from source files:")
        try:
            raw = extract_all() #separate function from data_Extraction
        except Exception as e:
            print(f"Extraction error: {e}")
            return

        if raw.empty:
            print("No data found. Exiting.")
            return

        required = ["project_title", "client", "project_type"]
        missing = [c for c in required if c not in raw.columns]
        if missing:
            print(f"Missing required columns in extracted data: {missing}")
            return

        df = raw[required].copy() #only keeping the columns we need

        #building the text column by concatenating the project title and client
        df["text"] = (
            df["project_title"].fillna("").astype(str)
            + " - "
            + df["client"].fillna("").astype(str)
        ).str.strip(" -")
        
        #cleaning up the df 
        df = df.dropna(subset=["project_type"]).copy()
        df = df[~df["project_type"].isin([None, "nan", "Cancelled"])]
        df["project_type"] = df["project_type"].astype(str)

        # ---- SAVE CLEANED CACHE (only once, after initial cleaning) ----
        try:
            CLEANED_CACHE.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(CLEANED_CACHE, index=False, engine="pyarrow")
            print("✅ Saved cleaned dataset to:", CLEANED_CACHE.resolve())
        except Exception as e:
            print("❌ Parquet save failed:", e)
            return

#TLDR flow -> if cache exists then go thru fast process, load it and ensure text column exists if not then create
#else -> slow process -> extract raw data, validate (Clean up df) columns -> create text column -> clean project type -> mapping -> drop classes not needed and then save to cache

    # --- Canonicalize classes and build aligned ids ---
    #more clean up to convert column to string and then strip white trailing spaces from both ends
    df["project_type"] = df["project_type"].astype(str).str.strip()

    #to make consistent class names
    clean_mapping = {'commercial':'Commercial', 'Community & Instituitional': 'Community & Institutional', 'residential': 'Residential'}
    df["project_type"] = df["project_type"].replace(clean_mapping)

    
    # Drop rare classes before building the class list
    vc = df["project_type"].value_counts() #count how many rows per classes
    keep = vc[vc >= 2].index
    df = df[df["project_type"].isin(keep)].reset_index(drop=True)

    # Build the final, canonical class list
    class_names = sorted(df["project_type"].unique().tolist()) # output is a list and this is the SINGLE SOURCE OF TRUTH -----

    # Assign numeric labels in guaranteed range [0 .. num_classes-1]
    #categorical is to convert a column into categorical data type
    df["label"] = pd.Categorical(
        df["project_type"], categories=class_names, ordered=True
    ).codes

    # Build id maps from the same single source
    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {name: i for i, name in enumerate(class_names)}
    num_labels = len(class_names)

    print(f"Final dataset has {num_labels} classes: {class_names}")

    from datasets import ClassLabel, Features, Value
    
    #using distilbert as model name here
    model_name = "distilbert-base-uncased"

    # Features must use the same ordered class list you built: class_names
    #create the features object - schema dictionary
    features = Features({
        "text": Value("string"), #defining the data type for the column text
        "label": ClassLabel(names=class_names), #categorical column with class_names as the names
    })

    print("DEBUG: num_classes =", len(class_names))
    print("DEBUG: label range =", df['label'].min(), "to", df['label'].max())
    assert df['label'].min() == 0
    assert df['label'].max() == len(class_names)-1

    dataset = Dataset.from_pandas(
        df[["text", "label"]].reset_index(drop=True),
        features=features,
        preserve_index=False,
    )

    #visualise the schema using features that tells what text and label are what data types
    print(dataset.features)

    train_test = dataset.train_test_split(
        test_size=0.3,
        seed=42,
        stratify_by_column="label",
    )

    #load the tokeniser from distilbert
    #convert raw text -> tokens that model understands
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #tokenizes a batch of rows, batch is a dictionary of columns
    def tok(batch):
        return tokenizer(batch["text"], truncation=True) #takes list of strings to convert them into token IDs

    tokenised = train_test.map(tok, batched=True, remove_columns=["text"]) #apply the tokenisation to whole dataset 
    data_collator = DataCollatorWithPadding(tokenizer) #fixed-size tensors per batch
    # num_labels must come from class_names, not any old variable
    num_labels = len(class_names)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    #Training parameters for fine tuning
    training_args = TrainingArguments(
        output_dir="./results", #where metrics get saved
        learning_rate=2e-5, 
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",  # change to "tensorboard" if need to use TB
        logging_steps=50,
        seed=42,
    )

    #acuracy metric
    accuracy = evaluate.load("accuracy") #loads accuracy function

    def compute_metrics(eval_pred):
        logits, labels_np = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=preds, references=labels_np)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # This will save the model files to a directory named 'my_finetuned_classifier'
    trainer.train()
    trainer.save_model("./my_finetuned_classifier")
    print("Model saved to ./my_finetuned_classifier")

    clf = pipeline( #ready-to-use classifier
        "text-classification",
        model="./my_finetuned_classifier",
        tokenizer=tokenizer,
        truncation=True,
    )
    #eg:
    print(clf("130351 - Hobart Airport Expansion - AUQLDPLAN - Planned Performance Pty Ltd"))

    #Final evaluation
    print(trainer.evaluate()) #eval loop on test tokenised data 

    #to compare predicted labels with the original
    print("\nGenerating predictions: ") 

    #predictions for the test set
    predictions = trainer.predict(test_dataset=tokenised["test"])

    #original vs true labels from the test dataset
    original_text = train_test["test"]["text"]
    true_labels_int = predictions.label_ids #get the true labels from predictions output

    #predicted labels by taking the argmax of the logits
    predicted_labels_int = np.argmax(predictions.predictions, axis=1) #guessed labels

    # Mapping the integer labels back to their original  labels
    true_labels_str = [id2label[i] for i in true_labels_int]
    predicted_labels_str = [id2label[i] for i in predicted_labels_int]

    #results df
    results_df = pd.DataFrame({
        "text": original_text,
        "original_label": true_labels_str,
        "predicted_label": predicted_labels_str,
    })

    from sklearn.metrics import confusion_matrix

    # use the same num_labels you computed from class_names
    cm = confusion_matrix(
        true_labels_int,
        predicted_labels_int,
        labels=list(range(num_labels))
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"true:{id2label[i]}" for i in range(num_labels)],
        columns=[f"pred:{id2label[i]}" for i in range(num_labels)]
    )

    cm_output_path = "cache/confusion_matrix.csv"
    cm_df.to_csv(cm_output_path, index=True)
    print("✅ Confusion matrix saved to:", Path(cm_output_path).resolve())

    #show cm
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 10_000)          # big width so it doesn't break lines
    pd.set_option("display.expand_frame_repr", False)  # <- stop wrapping onto multiple lines
    print(cm_df.to_string())

    # results_df already has columns: text + original_label + predicted_label
    correct = (results_df["original_label"] == results_df["predicted_label"]).sum()
    total = len(results_df)

    print(f"{correct}/{total} correct ({correct/total: 2%})")

    # --- Per-label test accuracy and training-set volume ---
    try:
        # Per-label test accuracy
        per_label_rows = []
        for i in range(num_labels):
            name = id2label[i]
            mask = (true_labels_int == i)
            support = int(mask.sum())
            if support > 0:
                acc = float((predicted_labels_int[mask] == true_labels_int[mask]).mean())
            else:
                acc = float('nan')
            per_label_rows.append({
                "label_id": i,
                "label": name,
                "test_support": support,
                "test_accuracy": acc,
            })
        # Sort per-label metrics by highest test accuracy, then by support (descending)
        per_label_df = (
            pd.DataFrame(per_label_rows)
            .sort_values(["test_accuracy", "test_support"], ascending=[False, False], na_position="last")
            .reset_index(drop=True)
        )

        # Training set label volume
        train_labels_int = np.array(train_test["train"]["label"])  # ints in [0..num_labels-1]
        train_counts = np.bincount(train_labels_int, minlength=num_labels)
        train_volume_df = (
            pd.DataFrame({
                "label_id": list(range(num_labels)),
                "label": [id2label[i] for i in range(num_labels)],
                "train_count": train_counts,
            })
            .sort_values("train_count", ascending=False)
            .reset_index(drop=True)
        )

        # Save both reports
        out_dir = Path("cache")
        out_dir.mkdir(parents=True, exist_ok=True)
        per_label_path = out_dir / "per_label_test_metrics.csv"
        train_volume_path = out_dir / "train_label_volume.csv"
        per_label_df.to_csv(per_label_path, index=False)
        train_volume_df.to_csv(train_volume_path, index=False)

        print("\nPer-label test accuracy (saved to):", per_label_path.resolve())
        print(per_label_df.to_string(index=False))

        print("\nTraining set label volume (saved to):", train_volume_path.resolve())
        print(train_volume_df.to_string(index=False))
    except Exception as e:
        print("❌ Failed to compute per-label metrics:", e)

    #output
    output_csv_path = "./predictions.csv"
    results_df.to_csv(output_csv_path, index=False)
    print(f"Prediction results saved to {output_csv_path}")

    # print(results_df.head(5))

if __name__ == "__main__":

    print("RUNNING:", __file__)
    main()
