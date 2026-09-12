# TherapyLM Design Project — ML vs. LLM Mental Health Text Classification

**CSE 4610 Design Project.** A faithful, disclosed replication of:

> Xie, C., Zhu, D., Wang, Z., Zhang, H., & Wei, Z. *Explainable AI for Mental Health
> Detection from Social Media: A Comparative Study of Traditional Machine Learning and
> a Large Language Model.* SSRN preprint 6429778 (not peer reviewed).

We reproduce the paper's seven-class classification of mental-health social-media posts,
comparing three classical ML models (LinearSVC, Random Forest, XGBoost) against a
zero-shot large language model, plus SVM-coefficient / SHAP / LIME explainability.

> **Disclosure.** This is an educational replication, cleared by our supervisor. We
> substitute the paper's LLM (Kimi K2) with OpenAI `gpt-5.6-sol`, and we note every
> deviation from the original in our presentation.

## Team & pipeline

All work happens in the notebooks under `notebooks/`, which import their logic from `src/`.

| Step | Notebook | Owner |
|------|----------|-------|
| 1. Clean data + EDA | `notebooks/01_data_cleaning_eda.ipynb` | Sadman |
| 2. TF-IDF + train SVM/RF/XGBoost + eval | `notebooks/02_train_classical_models.ipynb` | Sadman |
| 3. LLM zero-shot (500 posts) | `notebooks/03_llm_zero_shot_eval.ipynb` | Promitee |
| 4. Shared-subset comparison | `notebooks/04_shared_subset_comparison.ipynb` | Promitee |
| 5. Explainability + results tables | `notebooks/05_explainability.ipynb` | Maha |

## Quickstart

```bash
pip install -r requirements.txt        # global Python is fine; venv optional
cp .env.example .env                   # then paste your OpenAI key into .env

# put the Kaggle CSV at data/raw/Combined_Data.csv, then:
jupyter lab
```

Open and run the notebooks in order (1 → 5). Full details, including how artifacts get
shared between team members (some folders are gitignored on purpose), are in
`docs/IMPLEMENTATION.md` — kept local by the team, not committed to this repo.

## Data

Kaggle: *Sentiment Analysis for Mental Health* (`suchintikasarkar/sentiment-analysis-for-mental-health`).
Download the CSV manually and place it at `data/raw/Combined_Data.csv`. **Raw posts are
not committed** to this repo (see `.gitignore`), and notebook outputs are cleared before
every commit for the same reason (see `docs/IMPLEMENTATION.md` §2.5).

## Ethics

Public secondary data; screening-support signals only, **not diagnosis**. We report
aggregate metrics and do not reproduce identifiable posts in slides, the repo, or committed
notebook outputs.
