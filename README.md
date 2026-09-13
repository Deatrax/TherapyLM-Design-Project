# TherapyLM: A Multi-Model, Multi-Family Comparison for Mental Health Text Classification

**CSE 4610 Design Project — Islamic University of Technology (IUT)**

## Overview

This project presents an empirical comparison of classical machine learning
and large language models (LLMs) for multi-class mental health text
classification from social media posts. Rather than evaluating a single
model in isolation, we benchmark **three classical ML classifiers** against
**six LLMs spanning three distinct model families** (OpenAI, Google Gemini,
and locally-hosted open-weight models), all evaluated on identical
held-out data for a fully fair, apples-to-apples comparison.

This multi-family design is directly informed by our own literature review
of recent work in this space, which found that the overwhelming majority of
existing studies evaluate only a single LLM, or restrict themselves to a
single model family (e.g., only BERT-based or only GPT-based systems). Our
empirical study addresses that gap directly by building an evaluation
pipeline general enough to run classical ML and LLMs from any of the three
families side by side, with identical data, identical metrics, and
identical explainability tooling.

## Task

7-class classification of social media posts into one of:
**Normal, Depression, Anxiety, Suicidal, Stress, Bipolar, Personality
disorder.**

## Dataset

**"Sentiment Analysis for Mental Health"** (Kaggle, suchintikasarkar) —
a real-world dataset of social media posts, cleaned to ~50,700 rows across
the seven classes above. Split 80/20 (stratified, fixed seed) into
train/test, with the test set held out entirely from classical ML training
and used as the fixed evaluation set for every LLM as well.

## Methodology

### 1. Classical ML baselines
TF-IDF vectorization (unigrams + bigrams) feeding three classifiers:
**LinearSVC (SVM)**, **Random Forest**, and **XGBoost**. Evaluated on the
full held-out test set, plus stratified 5-fold cross-validation on the
training partition for split-sensitivity robustness.

### 2. Multi-family LLM zero-shot evaluation
Six models, three families, all evaluated zero-shot (no fine-tuning) on the
same fixed, stratified sample of test posts:

| Family | Models |
|---|---|
| OpenAI | `gpt-5.6-sol`, `gpt-5.6-terra` |
| Google (Vertex AI) | `gemini-3.1-pro-preview`, `gemini-3.6-flash` |
| Local (LM Studio, open-weight) | `meta-llama-3.1-8b-instruct`, `google/gemma-4b-e4b` |

Every model classifies the identical set of posts (same random seed), so
results are directly comparable across models, families, and the classical
ML baselines above — regardless of which team member ran which model on
which machine.

### 3. Explainability
SVM coefficient inspection, SHAP, and LIME are used to interpret classical
ML decisions and surface which lexical features drive each class
prediction.

### 4. Unified comparison
Classical ML predictions are re-computed on the exact same fixed sample the
LLMs were evaluated on, so every model — classical or LLM, any family — is
compared on identical data with identical metrics (accuracy, macro-F1,
weighted-F1, and full per-class precision/recall/F1).

## Repository Structure

```
TherapyLM-Design-Project/
├── data/                          # gitignored — shared via zip
│   ├── raw/
│   └── processed/
├── models/                        # trained model artifacts (rf.joblib Drive-shared, too large for git)
├── results/
│   ├── tables/                    # all metrics, per-class breakdowns, combined results
│   └── figures/                   # comparison charts, confusion matrices, SHAP/LIME plots
├── src/
│   ├── clean_data.py              # dataset loading & cleaning
│   ├── train_classical.py         # TF-IDF + SVM/RF/XGBoost training & evaluation
│   ├── multi_llm_eval.py          # multi-family LLM evaluation engine
│   ├── explainability.py          # SHAP / LIME / SVM coefficient analysis
│   ├── llm_zero_shot_eval.py      # shared LLM eval helpers (categories, sampling, OpenAI client)
│   └── shared_subset_compare.py   # utilities for comparing models on the shared sample
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb
│   ├── 02_train_classical_models.ipynb
│   ├── 03a_llm_eval_openai.ipynb      # OpenAI family
│   ├── 03b_llm_eval_gemini.ipynb      # Google Gemini family
│   ├── 03c_llm_eval_local.ipynb       # Local open-weight family
│   ├── 04_combine_and_compare.ipynb   # unifies all results, produces final comparison
│   └── 05_explainability.ipynb
├── requirements.txt
└── env.example
```

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `env.example` to `.env` and fill in:
   - `OPENAI_API_KEY` — for the OpenAI family
   - `GOOGLE_VERTEX_API_KEY` — for the Gemini family (Vertex AI project)
3. For the local model family, install [LM Studio](https://lmstudio.ai),
   load the target models, and start its local server. Update
   `LM_STUDIO_BASE_URL` in `src/multi_llm_eval.py` to point at the machine
   actually running LM Studio.
4. Download the dataset manually from Kaggle and place it under
   `data/raw/` (not redistributed in this repo).

## Running the Pipeline

Run in order:

1. `01_data_cleaning_eda.ipynb` — clean the raw dataset
2. `02_train_classical_models.ipynb` — train and evaluate SVM/RF/XGBoost
3. `03a` / `03b` / `03c` — run independently (can be split across team
   members and machines); each evaluates its own model family on the same
   fixed sample and saves per-model result files
4. `04_combine_and_compare.ipynb` — merges every team member's results
   (once pushed/pulled), adds classical ML on the same fixed sample, and
   produces the final unified comparison table and chart
5. `05_explainability.ipynb` — SHAP/LIME/coefficient analysis for the
   classical models

## Team

| Member | Contribution |
|---|---|
| Sadman | Data cleaning, classical ML training, local (LM Studio) LLM family evaluation, repository coordination |
| Promitee | OpenAI family LLM evaluation |
| Maha | Google Gemini family LLM evaluation, explainability analysis, presentation |

## Results

See `results/tables/multi_llm_metrics_summary.csv` and
`results/tables/multi_llm_metrics_per_class.csv` for the full unified
comparison across all classical ML models and all six LLMs, and
`results/figures/full_model_comparison.png` for the summary chart.
