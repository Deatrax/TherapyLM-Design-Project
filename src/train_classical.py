"""
train_classical.py  --  Step 2 of the pipeline (Sadman)

Creates the stratified 80/20 split, fits the TF-IDF vectoriser, trains the three
classical models (LinearSVC, RandomForest, XGBoost), evaluates them on the full
held-out test set, and saves every artifact Promitee and Maha need.

Written to be IMPORTED from notebooks/02_train_classical_models.ipynb.
Standalone CLI use still works:
    python train_classical.py --input data/processed/clean.csv

OUTPUTS
    models/tfidf_vectorizer.joblib   the fitted TF-IDF vectoriser
    models/svm.joblib                trained LinearSVC
    models/rf.joblib                 trained RandomForest
    models/xgb.joblib                trained XGBoost
    models/label_encoder.joblib      maps label text <-> integer (XGBoost needs ints)
    data/processed/train.csv         the exact training rows
    data/processed/test.csv          the exact held-out test rows (Promitee & Maha use this)
    results/tables/full_test_metrics.csv   accuracy / macro-F1 / weighted-F1 per model
"""

import argparse
import os
import joblib                        # save/load fitted sklearn objects to disk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

SEED = 42                           # fixed random seed => same split every run (reproducibility)


def evaluate(name, model, X_test, y_test, label_encoder=None, verbose=True):
    """Predict on the test set and return the headline metrics (and print them)."""
    preds = model.predict(X_test)                       # model's predicted labels
    if label_encoder is not None:                       # XGBoost returns ints; turn them back into text
        preds = label_encoder.inverse_transform(preds)
        y_true = label_encoder.inverse_transform(y_test)
    else:
        y_true = y_test
    acc = accuracy_score(y_true, preds)                              # overall correct fraction
    macro = f1_score(y_true, preds, average="macro", zero_division=0)     # unweighted mean F1 (imbalance-fair)
    weighted = f1_score(y_true, preds, average="weighted", zero_division=0)  # F1 weighted by class size
    if verbose:
        print(f"\n=== {name} (full test set) ===")
        print(f"Accuracy {acc:.3f} | Macro-F1 {macro:.3f} | Weighted-F1 {weighted:.3f}")
        print(classification_report(y_true, preds, zero_division=0))     # per-class precision/recall/F1
    return {"model": name, "accuracy": acc, "macro_f1": macro, "weighted_f1": weighted}


def prepare_split(df: pd.DataFrame):
    """Stratified 80/20 split. stratify=status keeps each class's proportion
    identical in train and test; random_state=SEED makes it reproducible."""
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["status"], random_state=SEED
    )
    return train_df, test_df


def fit_tfidf(train_texts):
    """Fit the TF-IDF vectoriser with the paper's exact settings."""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # use single words AND word pairs (bigrams)
        max_features=20000,   # keep at most the 20,000 most informative terms
        min_df=3,             # ignore terms appearing in fewer than 3 documents
        max_df=0.95,          # ignore terms appearing in more than 95% of documents
        sublinear_tf=True,    # dampen term-frequency with 1+log(tf) (paper's setting)
    )
    vectorizer.fit(train_texts)
    return vectorizer


def train_all_models(input_path: str):
    """
    Run the full Step 2 pipeline: split -> vectorise -> train SVM/RF/XGBoost ->
    evaluate -> save everything. Returns a dict with the fitted objects and the
    metrics DataFrame, so a notebook can keep working with them in memory
    without immediately re-loading from disk.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    df = pd.read_csv(input_path).dropna(subset=["statement", "status"])

    # ---- Stratified 80/20 split -------------------------------------------------
    train_df, test_df = prepare_split(df)
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    # ---- TF-IDF features ---------------------------------------------------------
    vectorizer = fit_tfidf(train_df["statement"])
    X_train = vectorizer.transform(train_df["statement"])       # transform train with the fitted vocab
    X_test = vectorizer.transform(test_df["statement"])         # transform test with the SAME vocab
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")   # save for Promitee & Maha

    y_train = train_df["status"]
    y_test = test_df["status"]

    metrics = []   # collect each model's headline numbers for the results table

    # ---- LinearSVC --------------------------------------------------------------
    svm = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)  # linear support-vector classifier
    svm.fit(X_train, y_train)                                       # train on TF-IDF features
    joblib.dump(svm, "models/svm.joblib")
    metrics.append(evaluate("SVM (LinearSVC)", svm, X_test, y_test))

    # ---- RandomForest -----------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", max_depth=None, n_jobs=-1, random_state=SEED
    )                                                              # 200 trees, use all CPU cores
    rf.fit(X_train, y_train)
    joblib.dump(rf, "models/rf.joblib")
    metrics.append(evaluate("Random Forest", rf, X_test, y_test))

    # ---- XGBoost ----------------------------------------------------------------
    # XGBoost needs integer labels, so encode text labels -> ints first.
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)   # e.g. "Depression" -> 1
    y_test_enc = le.transform(y_test)
    joblib.dump(le, "models/label_encoder.joblib")
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        tree_method="hist", random_state=SEED, n_jobs=-1,
    )                                                              # gradient-boosted trees
    xgb.fit(X_train, y_train_enc)
    joblib.dump(xgb, "models/xgb.joblib")
    metrics.append(evaluate("XGBoost", xgb, X_test, y_test_enc, label_encoder=le))

    # ---- Save the results table -------------------------------------------------
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv("results/tables/full_test_metrics.csv", index=False)
    print("\nSaved results/tables/full_test_metrics.csv")

    return {
        "vectorizer": vectorizer, "svm": svm, "rf": rf, "xgb": xgb, "label_encoder": le,
        "train_df": train_df, "test_df": test_df, "metrics_df": metrics_df,
    }


def main():
    """CLI entry point -- optional, the notebook is the primary interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Cleaned CSV from clean_data.py")
    args = parser.parse_args()
    train_all_models(args.input)


if __name__ == "__main__":
    main()
