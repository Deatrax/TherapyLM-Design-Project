"""
shared_subset_compare.py  --  Step 4 of the pipeline (Promitee)

The paper compares the LLM against SVM and Random Forest on the SAME 500 posts,
so the ML-vs-LLM comparison is fair (its main table mixes evaluation sizes).

Written to be IMPORTED from notebooks/04_shared_subset_comparison.ipynb.
Standalone CLI use still works:
    python shared_subset_compare.py --llm-results results/tables/llm_zero_shot_results.csv
"""

import argparse
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

CATEGORIES = ["Normal", "Depression", "Anxiety", "Suicidal", "Stress", "Bipolar", "Personality disorder"]


def score(y_true, y_pred):
    """Return the two headline numbers as a dict."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=CATEGORIES, zero_division=0),
    }


def compare_on_shared_subset(llm_results_path: str, models_dir: str = "models") -> pd.DataFrame:
    """
    Load the 500 posts the LLM already scored, re-run the saved SVM and RF on
    those exact rows, and return a 3-row comparison table.
    """
    # The LLM results file already contains the exact 500 statements + true labels
    # + the LLM's predictions, so we read the same rows the LLM saw.
    llm = pd.read_csv(llm_results_path).dropna(subset=["predicted_label"])
    statements = llm["statement"].astype(str)
    y_true = llm["true_label"]

    # Load the artifacts Sadman saved in Step 2.
    vectorizer = joblib.load(f"{models_dir}/tfidf_vectorizer.joblib")
    svm = joblib.load(f"{models_dir}/svm.joblib")
    rf = joblib.load(f"{models_dir}/rf.joblib")

    # Turn the 500 posts into TF-IDF features using the ALREADY-FITTED vectoriser
    # (transform, not fit_transform -- we must use the vocabulary learned on train).
    X = vectorizer.transform(statements)

    rows = []
    rows.append({"model": "SVM (LinearSVC)", **score(y_true, svm.predict(X))})
    rows.append({"model": "Random Forest", **score(y_true, rf.predict(X))})
    llm_model_name = f"LLM ({llm.attrs.get('model', 'zero-shot')})" if hasattr(llm, "attrs") else "LLM (zero-shot)"
    rows.append({"model": llm_model_name, **score(y_true, llm["predicted_label"])})

    table = pd.DataFrame(rows)
    print("=== Shared 500-post subset comparison ===")
    print(table.to_string(index=False))

    import os
    os.makedirs("results/tables", exist_ok=True)
    table.to_csv("results/tables/shared_subset_metrics.csv", index=False)
    print("\nSaved results/tables/shared_subset_metrics.csv")

    return table


def rationale_analysis(llm_results_path: str):
    """Print mean LLM rationale length and the top confusion pairs (true -> predicted)."""
    llm = pd.read_csv(llm_results_path).dropna(subset=["predicted_label"])

    if "rationale" in llm.columns:
        mean_len = llm["rationale"].astype(str).str.len().mean()
        print(f"Mean LLM rationale length: {mean_len:.0f} characters")

    wrong = llm[llm["true_label"] != llm["predicted_label"]]
    if len(wrong):
        pair = (wrong["true_label"] + " -> " + wrong["predicted_label"]).value_counts().head(5)
        print("\nTop LLM confusion pairs (true -> predicted):")
        print(pair.to_string())
    return wrong


def main():
    """CLI entry point -- optional, the notebook is the primary interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-results", required=True, help="CSV produced by llm_zero_shot_eval.py")
    args = parser.parse_args()
    compare_on_shared_subset(args.llm_results)
    rationale_analysis(args.llm_results)


if __name__ == "__main__":
    main()
