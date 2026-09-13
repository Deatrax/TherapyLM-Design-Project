"""
explainability.py  --  Step 5 of the pipeline (Maha)

Reproduces the paper's three model-faithful explanation views:
  1. Top TF-IDF words per class from the SVM's linear coefficients  -> one bar chart per class
  2. Global SHAP feature importance for the XGBoost model           -> SHAP summary plot
  3. LIME local explanation for one individual post                 -> LIME HTML

Written to be IMPORTED from notebooks/05_explainability.ipynb, where figures
display inline automatically. They are also saved to results/figures/ either way.

Standalone CLI use still works:
    python explainability.py --test data/processed/test.csv
(if running headless with no display, add `import matplotlib; matplotlib.use("Agg")`
at the very top of your own script before importing this module)
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def svm_top_words(vectorizer, svm, out_dir="results/figures", top_n=15, show=True):
    """For each class, plot the words with the largest positive SVM coefficients."""
    os.makedirs(out_dir, exist_ok=True)
    feature_names = np.array(vectorizer.get_feature_names_out())  # the vocabulary (one name per feature)
    classes = svm.classes_                                        # the class label for each coefficient row
    figures = {}
    for i, cls in enumerate(classes):
        coefs = svm.coef_[i]                                     # this class's weight for every word
        top_idx = np.argsort(coefs)[-top_n:]                    # indices of the top_n largest weights
        words = feature_names[top_idx]
        weights = coefs[top_idx]
        fig = plt.figure(figsize=(6, 5))
        plt.barh(range(len(words)), weights)                    # horizontal bar chart
        plt.yticks(range(len(words)), words)                    # label each bar with its word
        plt.title(f"Top SVM features: {cls}")
        plt.xlabel("SVM coefficient")
        plt.tight_layout()
        safe = str(cls).replace(" ", "_")
        path = os.path.join(out_dir, f"svm_top_words_{safe}.png")
        plt.savefig(path, dpi=120)
        if show:
            plt.show()          # renders inline automatically inside a Jupyter notebook
        else:
            plt.close(fig)
        figures[cls] = path
    print(f"Saved per-class SVM word charts to {out_dir}/")
    return figures


def shap_summary(vectorizer, xgb, texts, out_dir="results/figures", sample=300, show=True):
    """Global SHAP importance for XGBoost on a sample of posts (full set is slow)."""
    import shap
    os.makedirs(out_dir, exist_ok=True)
    X = vectorizer.transform(texts[:sample])                    # features for a manageable sample
    explainer = shap.TreeExplainer(xgb)                         # fast exact SHAP for tree models
    shap_values = explainer.shap_values(X)                      # contribution of each feature per prediction
    fig = plt.figure()
    # summary_plot ranks features by mean absolute SHAP value across the sample.
    shap.summary_plot(
        shap_values, X, feature_names=vectorizer.get_feature_names_out(),
        show=False, max_display=20, plot_type="bar",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "shap_summary.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved SHAP summary to {path}")
    return path


def lime_example(vectorizer, xgb, label_encoder, texts, out_dir="results/figures", index=0, display_inline=True):
    """LIME local explanation for one post (uses XGBoost because it has predict_proba)."""
    from lime.lime_text import LimeTextExplainer

    os.makedirs(out_dir, exist_ok=True)
    class_names = list(label_encoder.classes_)                  # human-readable class names for LIME

    def predict_proba(list_of_texts):                           # LIME needs a text -> probability function
        X = vectorizer.transform(list_of_texts)
        return xgb.predict_proba(X)

    explainer = LimeTextExplainer(class_names=class_names)
    text = str(texts.iloc[index])
    exp = explainer.explain_instance(text, predict_proba, num_features=10, top_labels=1)
    out_path = os.path.join(out_dir, "lime_example.html")
    exp.save_to_file(out_path)                                  # interactive HTML you can screenshot
    print(f"Saved LIME explanation for one post to {out_path}")

    if display_inline:
        try:
            from IPython.display import display
            exp.show_in_notebook(text=True)   # renders the LIME widget directly in the notebook
        except ImportError:
            pass   # not running inside IPython/Jupyter -- the saved HTML file is still there
    return out_path


def load_artifacts(models_dir="models"):
    """Convenience loader: pulls everything Sadman saved in Step 2."""
    artifacts = {
        "vectorizer": joblib.load(f"{models_dir}/tfidf_vectorizer.joblib"),
        "svm": joblib.load(f"{models_dir}/svm.joblib"),
        "label_encoder": joblib.load(f"{models_dir}/label_encoder.joblib"),
    }
    try:
        artifacts["xgb"] = joblib.load(f"{models_dir}/xgb.joblib")
    except Exception as e:
        print(f"  NOTE: could not load xgb.joblib ({type(e).__name__}: {e}) -- skipping. "
              f"Not used anywhere in SVM/SHAP/LIME explainability, so this is safe to ignore.")
        artifacts["xgb"] = None
    return artifacts


def main():
    """CLI entry point -- optional, the notebook is the primary interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", required=True, help="test.csv from train_classical.py")
    args = parser.parse_args()

    artifacts = load_artifacts()
    test_df = pd.read_csv(args.test).dropna(subset=["statement"])
    texts = test_df["statement"].astype(str)

    svm_top_words(artifacts["vectorizer"], artifacts["svm"], show=False)
    shap_summary(artifacts["vectorizer"], artifacts["xgb"], texts, show=False)
    lime_example(artifacts["vectorizer"], artifacts["xgb"], artifacts["label_encoder"], texts, display_inline=False)


if __name__ == "__main__":
    main()
