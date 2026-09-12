"""
llm_zero_shot_eval.py  --  Step 3 of the pipeline (Promitee)

Zero-shot classification of mental-health social media posts using OpenAI's
gpt-5.6-sol, replicating the LLM baseline from the SSRN comparative ML-vs-LLM
paper.

Written to be IMPORTED from notebooks/03_llm_zero_shot_eval.ipynb.
Standalone CLI use still works:
    export OPENAI_API_KEY="sk-..."
    python llm_zero_shot_eval.py --input data/processed/test.csv --n 5     # smoke test
    python llm_zero_shot_eval.py --input data/processed/test.csv --n 500   # real run

OUTPUT
    results/tables/llm_zero_shot_results.csv -- statement, true_label, predicted_label, rationale
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv          # reads OPENAI_API_KEY from a local .env file
from openai import OpenAI
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

try:
    from tqdm.auto import tqdm          # nice progress bar in a notebook; falls back below if missing
except ImportError:
    tqdm = None

# NOTE: these must match the label spelling in the Kaggle file EXACTLY (note the
# lowercase 'd' in "Personality disorder"), or that class scores zero on a mismatch.
CATEGORIES = ["Normal", "Depression", "Anxiety", "Suicidal", "Stress", "Bipolar", "Personality disorder"]

SYSTEM_PROMPT = (
    "You are a careful annotator for a mental-health text classification research study. "
    "Classify the following social media post into exactly one of these seven categories: "
    + ", ".join(CATEGORIES) + ". "
    "This is for academic research on screening-support signals, not diagnosis. "
    "Respond only with the structured output requested."
)

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "name": "mental_health_classification",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": CATEGORIES},
            "rationale": {
                "type": "string",
                "description": "One short sentence explaining the classification.",
            },
        },
        "required": ["label", "rationale"],
        "additionalProperties": False,
    },
}

# Standard, short-context pricing per 1M tokens, current as of Sep 2026.
# Check https://developers.openai.com/api/docs/pricing before your real run in
# case prices have moved.
PRICES = {
    "gpt-5.6-sol": (4.00, 20.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-6-astra": (10.00, 50.00),
}


def get_client() -> OpenAI:
    """Load OPENAI_API_KEY from .env and build the API client. Call this once
    per notebook session."""
    load_dotenv()                                    # reads .env in the current working directory
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Copy .env.example to .env and paste your key in."
        )
    return OpenAI(api_key=api_key)


def classify_post(client, model, post_text, max_retries=4):
    """Call the model once for one post, with retry/backoff. Returns
    (label, rationale, input_tokens, output_tokens) or (None, None, None, None)
    if every retry fails."""
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    # Guard against unusually long posts inflating cost/latency
                    {"role": "user", "content": post_text[:4000]},
                ],
                text={"format": RESPONSE_SCHEMA},
                temperature=0.1,
            )
            data = json.loads(response.output_text)
            usage = getattr(response, "usage", None)
            in_tok = getattr(usage, "input_tokens", None) if usage else None
            out_tok = getattr(usage, "output_tokens", None) if usage else None
            return data["label"], data["rationale"], in_tok, out_tok
        except Exception as e:
            wait = 2 ** attempt
            print(f"    retry {attempt + 1}/{max_retries} after error: {e} (waiting {wait}s)", file=sys.stderr)
            time.sleep(wait)
    return None, None, None, None


def stratified_sample(df, n, seed):
    """Sample n rows from df, stratified by 'status', as close to
    proportional as the class sizes allow.

    NOTE: this deliberately avoids `df.groupby(...).apply(lambda g: g.sample(...))`.
    In pandas 3.x that pattern silently DROPS the grouping column ('status')
    from the result whenever the applied function returns a same-shaped
    subset of the group -- a real, version-specific pandas behaviour change
    that would otherwise break this function with no error, just a missing
    column later on. Building the sample explicitly with a list + concat
    sidesteps it and works identically on pandas 1.x, 2.x, and 3.x.
    """
    frac = min(1.0, n / len(df))
    parts = [group.sample(frac=frac, random_state=seed) for _, group in df.groupby("status")]
    sample = pd.concat(parts, ignore_index=False)
    if len(sample) > n:
        sample = sample.sample(n=n, random_state=seed)
    return sample.reset_index(drop=True)


def run_zero_shot_eval(
    input_path: str,
    n: int = 500,
    model: str = "gpt-5.6-sol",
    seed: int = 42,
    output_path: str = "results/tables/llm_zero_shot_results.csv",
    client: OpenAI = None,
    save_every: int = 25,
) -> pd.DataFrame:
    """
    Sample n posts, classify each with the LLM ONE AT A TIME (so you can watch
    cost accumulate live on platform.openai.com/usage as this runs), save +
    return the results DataFrame.

    Saves progress to output_path every `save_every` requests, not just at the
    end. This matters if you interrupt the cell partway through (e.g. because
    the live cost looks wrong) -- without this, everything classified so far
    would be lost even though you already paid for those API calls. With it,
    interrupting the kernel at any point still leaves you with a CSV
    containing at most `save_every - 1` un-saved rows, and the `finally`
    block below saves one last time regardless of how the loop exits
    (including on KeyboardInterrupt from an interrupt).

    Pass in a client from get_client() so a notebook only builds it once per
    session.
    """
    if client is None:
        client = get_client()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(input_path)
    df = df.dropna(subset=["statement", "status"])
    df = df[df["status"].isin(CATEGORIES)]
    if df.empty:
        raise ValueError("No rows left after filtering to the 7 known categories — check your 'status' column.")

    sample = stratified_sample(df, n, seed)
    print(f"Sampled {len(sample)} posts across classes:")
    print(sample["status"].value_counts(), "\n")

    results = []
    total_in, total_out = 0, 0
    iterator = sample.iterrows()
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(sample), desc=f"Classifying with {model}")

    def save_progress():
        """Write whatever's been collected so far. Safe to call repeatedly --
        each call overwrites output_path with the latest snapshot."""
        if results:
            pd.DataFrame(results).to_csv(output_path, index=False)

    try:
        for i, row in iterator:
            label, rationale, in_tok, out_tok = classify_post(client, model, row["statement"])
            results.append({
                "statement": row["statement"],
                "true_label": row["status"],
                "predicted_label": label,
                "rationale": rationale,
            })
            if in_tok:
                total_in += in_tok
            if out_tok:
                total_out += out_tok
            if len(results) % save_every == 0:
                save_progress()             # periodic checkpoint -- see docstring above
            if tqdm is None and ((i + 1) % 25 == 0 or (i + 1) == len(sample)):
                print(f"  {i + 1}/{len(sample)} done...")
    except KeyboardInterrupt:
        # Expected if you interrupt the cell after watching cost on the OpenAI
        # console -- print a clear message and fall through to return whatever
        # was collected, rather than raising and losing everything in the notebook.
        print(f"\n\nInterrupted after {len(results)}/{len(sample)} requests. "
              f"Returning partial results -- nothing paid-for is lost.")
    finally:
        # Runs no matter how the loop above ends -- normal completion, an
        # unhandled exception, OR an interrupted kernel (KeyboardInterrupt).
        # Guarantees you never lose more than `save_every - 1` already-paid-for
        # results to an interruption.
        save_progress()

    out_df = pd.DataFrame(results)
    print(f"\nSaved predictions to {output_path} ({len(out_df)} rows)")

    out_df.attrs["total_in_tokens"] = total_in     # stash token counts on the DataFrame
    out_df.attrs["total_out_tokens"] = total_out   # so report_metrics() can print cost later
    out_df.attrs["model"] = model
    return out_df


def report_metrics(out_df: pd.DataFrame):
    """Print accuracy / macro-F1 / weighted-F1, a per-class report, a confusion
    matrix, and an estimated $ cost, for a DataFrame produced by run_zero_shot_eval."""
    valid = out_df.dropna(subset=["predicted_label"])
    failed = len(out_df) - len(valid)
    if failed:
        print(f"WARNING: {failed} posts failed after all retries and were excluded from metrics.")

    acc = accuracy_score(valid["true_label"], valid["predicted_label"])
    macro_f1 = f1_score(valid["true_label"], valid["predicted_label"], average="macro", labels=CATEGORIES, zero_division=0)
    weighted_f1 = f1_score(valid["true_label"], valid["predicted_label"], average="weighted", labels=CATEGORIES, zero_division=0)

    model = out_df.attrs.get("model", "LLM")
    print(f"=== Zero-shot {model} results ===")
    print(f"Accuracy:     {acc:.3f}")
    print(f"Macro-F1:     {macro_f1:.3f}")
    print(f"Weighted-F1:  {weighted_f1:.3f}\n")

    print("Per-class report:")
    print(classification_report(valid["true_label"], valid["predicted_label"], labels=CATEGORIES, zero_division=0))

    print("Confusion matrix (rows = true, cols = predicted):")
    cm = confusion_matrix(valid["true_label"], valid["predicted_label"], labels=CATEGORIES)
    print(pd.DataFrame(cm, index=CATEGORIES, columns=CATEGORIES))

    total_in = out_df.attrs.get("total_in_tokens")
    total_out = out_df.attrs.get("total_out_tokens")
    if model in PRICES and total_in and total_out:
        in_price, out_price = PRICES[model]
        cost = (total_in / 1e6) * in_price + (total_out / 1e6) * out_price
        print(f"\nTokens used: {total_in} in / {total_out} out")
        print(f"Estimated cost: ${cost:.4f}")

    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}


def main():
    """CLI entry point -- optional, the notebook is the primary interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with 'statement' and 'status' columns")
    parser.add_argument("--n", type=int, default=500, help="Number of posts to sample (stratified)")
    parser.add_argument("--model", default="gpt-5.6-sol", help="OpenAI model name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/tables/llm_zero_shot_results.csv")
    args = parser.parse_args()

    out_df = run_zero_shot_eval(args.input, args.n, args.model, args.seed, args.output)
    report_metrics(out_df)


# ============================================================================
# OPTIONAL SCALE-UP: evaluate on (up to) the full test set, concurrently,
# with a stability check across chunks. Use this instead of the single
# 500-post run above if you have the OpenAI budget and want the LLM's
# headline number to be directly comparable to the classical models' FULL
# test-set numbers, rather than a smaller subset.
#
# NOTE ON TERMINOLOGY: this is NOT k-fold cross-validation, even though it
# reuses StratifiedKFold. K-fold retrains a model k times to see how much its
# performance depends on which data it trained on -- but a zero-shot LLM
# never trains on your data at all, so there's no "fold" for it to be
# trained on or held out from. Here, StratifiedKFold is used purely as a
# convenient way to cut the evaluation set into balanced, non-overlapping
# chunks, so we can report a mean +/- std across chunks (a stability check)
# alongside the pooled score over every post evaluated (the actual headline
# number). Nothing is "held out" from anything else.
# ============================================================================

def classify_posts_concurrent(client, model, df: pd.DataFrame, max_workers: int = 10):
    """
    Classify every row in df, firing up to max_workers requests at once.

    Returns a DataFrame with the same rows plus predicted_label/rationale
    columns, in the SAME ROW ORDER as the input (order is restored after
    the concurrent calls complete, since they don't finish in submission
    order) -- plus the total input/output token counts actually used.

    max_workers=10 is a conservative default that comfortably fits under
    the rate limits of a standard OpenAI account tier. If you hit 429 (rate
    limit) errors, lower it; if requests are comfortably fast with room to
    spare, you can raise it -- classify_post()'s existing retry/backoff
    absorbs occasional rate-limit errors either way, it'll just be slower.
    """
    results = [None] * len(df)          # pre-sized so we can drop each result back into its original position
    total_in, total_out = 0, 0
    rows = list(df.itertuples(index=False))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_pos = {
            pool.submit(classify_post, client, model, row.statement): pos
            for pos, row in enumerate(rows)
        }
        iterator = as_completed(future_to_pos)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(future_to_pos), desc=f"Classifying with {model} ({max_workers} workers)")

        for future in iterator:
            pos = future_to_pos[future]
            label, rationale, in_tok, out_tok = future.result()
            results[pos] = (label, rationale)
            if in_tok:
                total_in += in_tok
            if out_tok:
                total_out += out_tok

    out_df = df.copy().reset_index(drop=True)
    out_df["predicted_label"] = [r[0] for r in results]
    out_df["rationale"] = [r[1] for r in results]
    return out_df, total_in, total_out


def run_zero_shot_eval_full(
    input_path: str,
    n_chunks: int = 5,
    model: str = "gpt-5.6-sol",
    seed: int = 42,
    max_workers: int = 10,
    output_path: str = "results/tables/llm_zero_shot_results_full.csv",
    client: OpenAI = None,
) -> pd.DataFrame:
    """
    Evaluate the LLM on every row of input_path (typically test.csv, so this
    matches the classical models' full-test-set evaluation size), split into
    n_chunks non-overlapping stratified chunks purely for a mean +/- std
    stability check. See the module-level note above on why this isn't
    "k-fold" in the training sense.

    Cost/time at n_chunks=5 covering the ~10,461-row Kaggle test split and
    gpt-5.6-sol pricing: roughly $19-20 and ~15-20 minutes with the default
    10 concurrent workers (vs. 3-6 HOURS run sequentially -- don't run this
    without concurrency).
    """
    if client is None:
        client = get_client()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(input_path)
    df = df.dropna(subset=["statement", "status"])
    df = df[df["status"].isin(CATEGORIES)].reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows left after filtering to the 7 known categories — check your 'status' column.")
    df = df.rename(columns={"status": "true_label"})   # match run_zero_shot_eval()'s output schema
    # so this file can be dropped straight into shared_subset_compare.py in place of the 500-post version

    # Assign each row to one of n_chunks stratified, non-overlapping groups.
    # We only use StratifiedKFold's .split() for the partitioning itself --
    # nothing is trained here, so "train"/"test" indices just mean "the other
    # (n_chunks-1) groups" / "this one group" for chunk-assignment purposes.
    skf = StratifiedKFold(n_splits=n_chunks, shuffle=True, random_state=seed)
    chunk_id = pd.Series(0, index=df.index)
    for chunk_num, (_, chunk_idx) in enumerate(skf.split(df["statement"], df["true_label"])):
        chunk_id.iloc[chunk_idx] = chunk_num
    df["chunk"] = chunk_id

    print(f"Evaluating {len(df)} posts across {n_chunks} chunks with {model} "
          f"({max_workers} concurrent workers)...")
    print(df["true_label"].value_counts(), "\n")

    out_df, total_in, total_out = classify_posts_concurrent(client, model, df, max_workers=max_workers)
    out_df.to_csv(output_path, index=False)
    print(f"\nSaved predictions to {output_path}")

    out_df.attrs["total_in_tokens"] = total_in
    out_df.attrs["total_out_tokens"] = total_out
    out_df.attrs["model"] = model
    return out_df


def report_chunk_metrics(out_df: pd.DataFrame):
    """
    Print per-chunk accuracy/macro-F1 (the stability check), the mean +/- std
    across chunks, the POOLED score across every post evaluated (the actual
    headline number -- directly comparable to the classical models' full
    test-set numbers), and the estimated cost.
    """
    valid = out_df.dropna(subset=["predicted_label"])
    failed = len(out_df) - len(valid)
    if failed:
        print(f"WARNING: {failed} posts failed after all retries and were excluded from metrics.")

    print("Per-chunk metrics (the stability check):")
    chunk_rows = []
    for chunk_num, group in valid.groupby("chunk"):
        acc = accuracy_score(group["true_label"], group["predicted_label"])
        macro = f1_score(group["true_label"], group["predicted_label"], average="macro", labels=CATEGORIES, zero_division=0)
        chunk_rows.append({"chunk": chunk_num, "n": len(group), "accuracy": acc, "macro_f1": macro})
    chunk_df = pd.DataFrame(chunk_rows)
    print(chunk_df.to_string(index=False))

    print(f"\nMean +/- std across chunks: "
          f"accuracy {chunk_df['accuracy'].mean():.3f} +/- {chunk_df['accuracy'].std():.3f}  |  "
          f"macro-F1 {chunk_df['macro_f1'].mean():.3f} +/- {chunk_df['macro_f1'].std():.3f}")

    y_true = valid["true_label"]
    pooled_acc = accuracy_score(y_true, valid["predicted_label"])
    pooled_macro = f1_score(y_true, valid["predicted_label"], average="macro", labels=CATEGORIES, zero_division=0)
    pooled_weighted = f1_score(y_true, valid["predicted_label"], average="weighted", labels=CATEGORIES, zero_division=0)
    print(f"\n=== Pooled result across all {len(valid)} posts (the headline number) ===")
    print(f"Accuracy:     {pooled_acc:.3f}")
    print(f"Macro-F1:     {pooled_macro:.3f}")
    print(f"Weighted-F1:  {pooled_weighted:.3f}")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    cm = confusion_matrix(y_true, valid["predicted_label"], labels=CATEGORIES)
    print(pd.DataFrame(cm, index=CATEGORIES, columns=CATEGORIES))

    model = out_df.attrs.get("model", "LLM")
    total_in = out_df.attrs.get("total_in_tokens")
    total_out = out_df.attrs.get("total_out_tokens")
    if model in PRICES and total_in and total_out:
        in_price, out_price = PRICES[model]
        cost = (total_in / 1e6) * in_price + (total_out / 1e6) * out_price
        print(f"\nTokens used: {total_in} in / {total_out} out")
        print(f"Estimated cost: ${cost:.4f}")

    return {"accuracy": pooled_acc, "macro_f1": pooled_macro, "weighted_f1": pooled_weighted,
            "chunk_table": chunk_df}


if __name__ == "__main__":
    main()
