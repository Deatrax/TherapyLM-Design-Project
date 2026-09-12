"""
multi_llm_eval.py  --  Multi-model, multi-family LLM comparison

Evaluates the SAME stratified sample of posts across multiple LLMs spanning
three families (OpenAI, Google Gemini, and a locally-run open-weights model),
directly addressing the paper's own limitation of using only one LLM.

KEY ARCHITECTURAL FACT THIS RELIES ON: Gemini exposes an OpenAI-compatible
endpoint (generativelanguage.googleapis.com/v1beta/openai/), and Ollama
(the standard way to run local models) does too. So this file uses ONE
universal function against the `openai` Python client for all three
providers -- only `base_url` and `api_key` change per provider, not the
calling code. This is the entire reason this file is short.

Written to be IMPORTED from a notebook. Standalone CLI use also works:
    python multi_llm_eval.py --input data/processed/test.csv

SETUP
    1. OpenAI: OPENAI_API_KEY in .env (already set up if you did Step 3)
    2. Gemini: GEMINI_API_KEY in .env (get one free at aistudio.google.com)
    3. Local: install Ollama (ollama.com), then:
         ollama pull llama3.1:8b
       Ollama runs its own local server automatically after install; no key needed.

OUTPUT
    results/tables/multi_llm_<model>.csv   one file per model
    results/tables/multi_llm_all_results.csv   everything combined, long format
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

# Reuse the exact same categories and sampling logic already tested in Step 3,
# so this evaluates on a genuinely comparable set of posts.
from llm_zero_shot_eval import CATEGORIES, stratified_sample

# ============================================================================
# Provider configuration -- this table is the only per-provider "code" needed.
# Add/remove/rename models by editing this dict; nothing else needs to change.
# ============================================================================
PROVIDERS = {
    "gpt-6-astra": {
        "base_url": None,                 # None = OpenAI's default endpoint
        "api_key_env": "OPENAI_API_KEY",
        "use_json_mode": True,            # OpenAI reliably honors response_format
        "family": "OpenAI",
    },
    "gpt-5.6-terra": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "use_json_mode": True,
        "family": "OpenAI",
    },
    "gemini-3.1-pro": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "use_json_mode": True,
        "family": "Google",
    },
    "gemini-3.6-flash": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "use_json_mode": True,
        "family": "Google",
    },
    "llama3.1:8b": {
        "base_url": "http://localhost:11434/v1",
        "api_key_env": None,              # Ollama ignores the key; any string works
        "use_json_mode": False,           # safer default: not every Ollama/model combo
                                           # reliably honors response_format via the
                                           # compat layer -- rely on prompt + robust
                                           # parsing instead for the local model
        "family": "Local (Ollama)",
    },
}

SYSTEM_PROMPT = (
    "You are a careful annotator for a mental-health text classification research study. "
    "Classify the following social media post into exactly one of these seven categories: "
    + ", ".join(CATEGORIES) + ". "
    "This is for academic research on screening-support signals, not diagnosis. "
    "Respond with ONLY a single JSON object and nothing else -- no markdown code fences, "
    "no explanation outside the JSON -- in exactly this shape: "
    '{"label": "<one of the seven categories, spelled exactly as given>", '
    '"rationale": "<one short sentence>"}'
)


def get_client_for(model_key: str) -> OpenAI:
    """Build the right client for whichever provider this model belongs to.
    This is the ENTIRE multi-provider abstraction -- everything downstream
    just calls client.chat.completions.create() the same way regardless."""
    cfg = PROVIDERS[model_key]
    if cfg["api_key_env"]:
        api_key = os.environ.get(cfg["api_key_env"])
        if not api_key:
            raise RuntimeError(f"{cfg['api_key_env']} not set in .env -- needed for {model_key}")
    else:
        api_key = "not-needed-for-local-ollama"   # Ollama doesn't check this
    kwargs = {"api_key": api_key}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    return OpenAI(**kwargs)


def _extract_json(text: str):
    """
    Robust JSON extraction that doesn't assume the provider perfectly honored
    response_format. Handles: clean JSON, JSON wrapped in ```...``` fences
    (common local-model habit), and JSON with stray prose around it.
    Returns a dict, or None if nothing parseable was found.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def classify_post_universal(client, model, post_text, use_json_mode=True, max_retries=4):
    """
    Classify one post with one model, retrying with backoff. Works identically
    for OpenAI, Gemini (via its OpenAI-compat endpoint), and a local Ollama
    model -- the only thing that ever differs is which `client` was passed in.

    Returns (label, rationale, input_tokens, output_tokens), or all-None if
    every retry fails.
    """
    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": str(post_text)[:4000]},
                ],
                temperature=0.1,
            )
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            data = _extract_json(content)
            if not data or "label" not in data:
                raise ValueError(f"no parseable JSON label in response: {content[:150]!r}")

            label = data["label"]
            if label not in CATEGORIES:
                # local/smaller models sometimes vary case or add punctuation --
                # try a forgiving match before treating it as a real failure
                normalized = next((c for c in CATEGORIES if c.lower().strip(" .") == str(label).lower().strip(" .")), None)
                if normalized is None:
                    raise ValueError(f"label {label!r} isn't one of the 7 categories")
                label = normalized

            rationale = data.get("rationale", "")
            usage = getattr(response, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", None) if usage else None
            out_tok = getattr(usage, "completion_tokens", None) if usage else None
            return label, rationale, in_tok, out_tok
        except Exception as e:
            wait = 2 ** attempt
            print(f"    [{model}] retry {attempt + 1}/{max_retries} after error: {e} (waiting {wait}s)", file=sys.stderr)
            time.sleep(wait)
    return None, None, None, None


def run_multi_model_eval(
    input_path: str,
    models: list = None,
    n: int = 500,
    seed: int = 42,
    max_workers: int = 10,
    output_dir: str = "results/tables",
    save_every: int = 25,
) -> pd.DataFrame:
    """
    Evaluate every model in `models` (default: all of PROVIDERS) on the SAME
    stratified sample of n posts -- same seed, so every model sees identical
    posts, which is what makes the cross-model comparison fair.

    Saves one CSV per model plus a combined long-format CSV
    (results/tables/multi_llm_all_results.csv) with a 'model' and 'family'
    column, ready for a groupby comparison. Interrupt-safe per model: hitting
    stop partway through a model saves what that model has completed so far
    and moves on cleanly (nothing already paid for is lost).
    """
    load_dotenv()
    models = models or list(PROVIDERS.keys())
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)
    df = df.dropna(subset=["statement", "status"])
    df = df[df["status"].isin(CATEGORIES)]
    sample = stratified_sample(df, n, seed)
    print(f"Sampled {len(sample)} posts (the SAME set will be used for every model).")
    print(sample["status"].value_counts(), "\n")

    all_results = []

    for model_key in models:
        cfg = PROVIDERS[model_key]
        print(f"\n{'=' * 60}\n{model_key}  ({cfg['family']})\n{'=' * 60}")
        try:
            client = get_client_for(model_key)
        except RuntimeError as e:
            print(f"  SKIPPING {model_key}: {e}")
            continue

        safe_name = model_key.replace(":", "_").replace(".", "_")
        model_output_path = f"{output_dir}/multi_llm_{safe_name}.csv"

        results = []
        rows = list(sample.itertuples(index=False))

        def classify_one(row, _model=model_key, _client=client, _use_json=cfg["use_json_mode"]):
            label, rationale, in_tok, out_tok = classify_post_universal(
                _client, _model, row.statement, use_json_mode=_use_json
            )
            return {
                "model": _model, "family": cfg["family"],
                "statement": row.statement, "true_label": row.status,
                "predicted_label": label, "rationale": rationale,
                "in_tokens": in_tok, "out_tokens": out_tok,
            }

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(classify_one, row): i for i, row in enumerate(rows)}
                iterator = as_completed(futures)
                if tqdm is not None:
                    iterator = tqdm(iterator, total=len(futures), desc=model_key)
                for future in iterator:
                    results.append(future.result())
                    if len(results) % save_every == 0:
                        pd.DataFrame(results).to_csv(model_output_path, index=False)
        except KeyboardInterrupt:
            print(f"\n  Interrupted during {model_key} after {len(results)}/{len(rows)} -- "
                  f"saved what's done, moving on is your call (re-run to retry this model).")
        finally:
            if results:
                pd.DataFrame(results).to_csv(model_output_path, index=False)
                print(f"  Saved {len(results)} rows to {model_output_path}")

        all_results.extend(results)

    combined = pd.DataFrame(all_results)
    combined_path = f"{output_dir}/multi_llm_all_results.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nSaved combined results ({len(combined)} rows total) to {combined_path}")
    return combined


def report_multi_model_metrics(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Per-model accuracy/macro-F1/weighted-F1, sorted best-to-worst by macro-F1."""
    rows = []
    for model, group in combined_df.groupby("model"):
        valid = group.dropna(subset=["predicted_label"])
        if valid.empty:
            continue
        acc = accuracy_score(valid["true_label"], valid["predicted_label"])
        macro = f1_score(valid["true_label"], valid["predicted_label"], average="macro", labels=CATEGORIES, zero_division=0)
        weighted = f1_score(valid["true_label"], valid["predicted_label"], average="weighted", labels=CATEGORIES, zero_division=0)
        rows.append({
            "model": model, "family": group["family"].iloc[0], "n": len(valid),
            "accuracy": acc, "macro_f1": macro, "weighted_f1": weighted,
        })
    result_df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    print(result_df.to_string(index=False))
    return result_df


def combine_saved_results(models: list = None, output_dir: str = "results/tables") -> pd.DataFrame:
    """
    Combine already-saved per-model CSVs into one combined DataFrame WITHOUT
    re-running anything. Use this after team members have each independently
    run a different subset of models on their own machines (e.g. Promitee ran
    the OpenAI models, Maha ran Gemini, Sadman ran the local model on his
    dad's machine) and pushed their results/tables/multi_llm_*.csv files.

    This works correctly because every person calls run_multi_model_eval()
    with the SAME seed (42) and the SAME test.csv (already shared via git) --
    so stratified_sample() deterministically produces the IDENTICAL 500 posts
    on every machine, with no coordination needed beyond everyone using the
    defaults. Nobody needs to send the sampled subset to anyone else first.
    """
    models = models or list(PROVIDERS.keys())
    frames = []
    for model_key in models:
        safe_name = model_key.replace(":", "_").replace(".", "_")
        path = f"{output_dir}/multi_llm_{safe_name}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            frames.append(df)
            print(f"  Loaded {len(df)} rows for {model_key} from {path}")
        else:
            print(f"  MISSING: {path} -- {model_key} hasn't been run/pulled yet, skipping")
    if not frames:
        raise FileNotFoundError(
            "No per-model result files found in "
            f"{output_dir}/ -- has anyone pushed their results yet? "
            "Try `git pull` first."
        )
    combined = pd.concat(frames, ignore_index=True)
    combined_path = f"{output_dir}/multi_llm_all_results.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined {len(combined)} rows total -> {combined_path}")

    missing = [m for m in models if m not in combined["model"].unique()]
    if missing:
        print(f"\nStill waiting on: {missing}")
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--models", nargs="*", default=None, help="Subset of PROVIDERS keys; default = all")
    parser.add_argument("--max-workers", type=int, default=10)
    args = parser.parse_args()

    combined = run_multi_model_eval(args.input, models=args.models, n=args.n, max_workers=args.max_workers)
    report_multi_model_metrics(combined)


if __name__ == "__main__":
    main()
