"""
multi_llm_eval.py  --  Multi-model, multi-family LLM comparison

Evaluates the SAME stratified sample of posts across multiple LLMs spanning
three families (OpenAI, Google Gemini via Vertex AI trial credits, and
locally-run open-weights models via LM Studio), directly addressing the
paper's own limitation of using only one LLM.

TWO REQUEST PATHS, NOT ONE -- this is a change from an earlier version of
this file. Originally every provider went through one universal OpenAI-client
call (base_url swap only). That still works for OpenAI itself and for LM
Studio (both are genuinely OpenAI-compatible with simple/no auth). But Gemini
is different here: your Google Cloud trial credits only pay for calls made
through Vertex AI's NATIVE REST endpoint with a plain API key
(aiplatform.googleapis.com .../generateContent?key=...). Vertex's own
OpenAI-compatible layer exists too, but needs OAuth/Bearer-token auth rather
than a simple key -- more setup than what's already proven to work for you.
So Gemini gets its own request function (classify_post_vertex) using
`requests` directly, while OpenAI and LM Studio share the `openai` client
path (classify_post_openai_compat) via a `type` field in PROVIDERS.

Written to be IMPORTED from a notebook. Standalone CLI use also works:
    python multi_llm_eval.py --input data/processed/test.csv

SETUP
    1. OpenAI: OPENAI_API_KEY in .env (already set up if you did Step 3)
    2. Gemini/Vertex: GOOGLE_VERTEX_API_KEY in .env -- from a Google Cloud
       $300-trial project (console.cloud.google.com/freetrial), API key
       generated under APIs & Services -> Credentials, with the Vertex AI /
       Agent Platform API enabled and allowed on that key.
    3. Local (LM Studio): load your model(s) in LM Studio, start its local
       server, and make sure it's reachable at LM_STUDIO_BASE_URL below (a
       LAN IP:port if it's running on another machine, e.g. your dad's Mac --
       LM Studio's server must be bound to 0.0.0.0, not just localhost, for
       another machine to reach it). No API key needed.

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
import requests
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from llm_zero_shot_eval import CATEGORIES, stratified_sample

# ============================================================================
# Provider configuration
# ============================================================================
# LM Studio's server address -- update this if it changes. Point it at the
# machine actually running LM Studio (a LAN IP if that's a different machine
# from the one running this notebook).
LM_STUDIO_BASE_URL = "http://192.168.68.100:1234/v1"

PROVIDERS = {
    "gpt-5.6-sol": {
        "type": "openai_compat", "base_url": None, "api_key_env": "OPENAI_API_KEY",
        "use_json_mode": True, "supports_temperature": False, "family": "OpenAI",
    },
    "gpt-5.6-terra": {
        "type": "openai_compat", "base_url": None, "api_key_env": "OPENAI_API_KEY",
        "use_json_mode": True, "supports_temperature": False, "family": "OpenAI",
    },
    "gemini-3.1-pro-preview": {
        "type": "vertex", "api_key_env": "GOOGLE_VERTEX_API_KEY", "family": "Google",
    },
    "gemini-3.6-flash": {
        "type": "vertex", "api_key_env": "GOOGLE_VERTEX_API_KEY", "family": "Google",
    },
    "meta-llama-3.1-8b-instruct": {
        "type": "openai_compat", "base_url": LM_STUDIO_BASE_URL, "api_key_env": None,
        "use_json_mode": False,   # safer default via a local server's compat layer
        "family": "Local (LM Studio)",
    },
    "google/gemma-4b-e4b": {
        "type": "openai_compat", "base_url": LM_STUDIO_BASE_URL, "api_key_env": None,
        "use_json_mode": False,
        "family": "Local (LM Studio)",
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

# Some models (typically "reasoning"-style ones) reject a custom temperature
# entirely and only support their fixed default. Retrying the identical
# request against a model like that can never succeed -- it's a deterministic
# 400, not a transient failure -- so once we learn a model rejects it, we
# remember that for the rest of the run and stop sending it. This is a
# process-lifetime cache, not per-call: discovering it once (e.g. during a
# 5-post smoke test) means the full 500-post run never has to rediscover it.
_no_temperature_support = set()


def _extract_json(text: str):
    """Robust JSON extraction -- handles clean JSON, ```-fenced JSON, and JSON
    with stray prose around it. Returns a dict, or None if unparseable."""
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


def _normalize_label(data):
    """Pull a validated label + rationale out of a parsed JSON dict, or raise."""
    if not data or "label" not in data:
        raise ValueError(f"no parseable JSON label in: {data!r}")
    label = data["label"]
    if label not in CATEGORIES:
        normalized = next((c for c in CATEGORIES if c.lower().strip(" .") == str(label).lower().strip(" .")), None)
        if normalized is None:
            raise ValueError(f"label {label!r} isn't one of the 7 categories")
        label = normalized
    return label, data.get("rationale", "")


def get_openai_client_for(model_key: str) -> OpenAI:
    """Build an `openai` client for an 'openai_compat' provider (OpenAI itself
    or LM Studio) -- only base_url/api_key differ."""
    cfg = PROVIDERS[model_key]
    if cfg["api_key_env"]:
        api_key = os.environ.get(cfg["api_key_env"])
        if not api_key:
            raise RuntimeError(f"{cfg['api_key_env']} not set in .env -- needed for {model_key}")
    else:
        api_key = "not-needed-for-local-lm-studio"
    kwargs = {"api_key": api_key}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    return OpenAI(**kwargs)


def classify_post_openai_compat(client, model, post_text, use_json_mode=True, supports_temperature=True, max_retries=4):
    """Classify one post via the openai client -- used for OpenAI itself and
    for LM Studio (both are genuinely OpenAI-compatible)."""
    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": str(post_text)[:4000]},
                ],
            )
            if supports_temperature and model not in _no_temperature_support:
                kwargs["temperature"] = 0.1
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            label, rationale = _normalize_label(_extract_json(content))
            usage = getattr(response, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", None) if usage else None
            out_tok = getattr(usage, "completion_tokens", None) if usage else None
            return label, rationale, in_tok, out_tok
        except Exception as e:
            err_str = str(e).lower()
            if "temperature" in err_str and "does not support" in err_str and model not in _no_temperature_support:
                _no_temperature_support.add(model)
                print(f"    [{model}] this model doesn't support a custom temperature -- "
                      f"switching to its default for the rest of the run", file=sys.stderr)
                continue   # retry immediately with temperature dropped, no backoff needed
            wait = 2 ** attempt
            print(f"    [{model}] retry {attempt + 1}/{max_retries} after error: {e} (waiting {wait}s)", file=sys.stderr)
            time.sleep(wait)
    return None, None, None, None


def classify_post_vertex(model, post_text, max_retries=4):
    """
    Classify one post via Vertex AI's NATIVE generateContent REST endpoint --
    this is the path your $300 trial credit actually bills against. Uses
    Gemini's own generationConfig.responseMimeType for JSON mode (a native
    Gemini feature, unrelated to OpenAI's response_format).
    """
    api_key = os.environ.get("GOOGLE_VERTEX_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_VERTEX_API_KEY not set in .env")

    url = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": str(post_text)[:4000]}]}],
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            label, rationale = _normalize_label(_extract_json(text))
            usage = data.get("usageMetadata", {})
            in_tok = usage.get("promptTokenCount")
            out_tok = usage.get("candidatesTokenCount")
            return label, rationale, in_tok, out_tok
        except Exception as e:
            wait = 2 ** attempt
            print(f"    [{model}] retry {attempt + 1}/{max_retries} after error: {e} (waiting {wait}s)", file=sys.stderr)
            time.sleep(wait)
    return None, None, None, None


def classify_post_dispatch(model_key, post_text, openai_client=None):
    """Route to the right classify function based on PROVIDERS[model_key]['type']."""
    cfg = PROVIDERS[model_key]
    if cfg["type"] == "vertex":
        return classify_post_vertex(model_key, post_text)
    return classify_post_openai_compat(
        openai_client, model_key, post_text,
        use_json_mode=cfg["use_json_mode"],
        supports_temperature=cfg.get("supports_temperature", True),
    )


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
    posts, which is what makes the cross-model comparison fair, even when
    different team members run different models on different machines.

    Saves one CSV per model plus a combined long-format CSV. Interrupt-safe
    per model: stopping partway through a model saves what's done so far.
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

        openai_client = None
        if cfg["type"] == "openai_compat":
            try:
                openai_client = get_openai_client_for(model_key)
            except RuntimeError as e:
                print(f"  SKIPPING {model_key}: {e}")
                continue
        elif cfg["type"] == "vertex" and not os.environ.get(cfg["api_key_env"]):
            print(f"  SKIPPING {model_key}: {cfg['api_key_env']} not set in .env")
            continue

        safe_name = model_key.replace(":", "_").replace(".", "_").replace("/", "_")
        model_output_path = f"{output_dir}/multi_llm_{safe_name}.csv"

        results = []
        rows = list(sample.itertuples(index=False))

        def classify_one(row, _model=model_key, _client=openai_client):
            label, rationale, in_tok, out_tok = classify_post_dispatch(_model, row.statement, openai_client=_client)
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
                  f"saved what's done (re-run to retry this model).")
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


def combine_saved_results(models: list = None, output_dir: str = "results/tables") -> pd.DataFrame:
    """Combine already-saved per-model CSVs (produced by team members running
    independently on different machines) into one DataFrame, WITHOUT
    re-running anything. This works because everyone uses the same seed on
    the same test.csv, so the sample is identical everywhere."""
    models = models or list(PROVIDERS.keys())
    frames = []
    for model_key in models:
        safe_name = model_key.replace(":", "_").replace(".", "_").replace("/", "_")
        path = f"{output_dir}/multi_llm_{safe_name}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            frames.append(df)
            print(f"  Loaded {len(df)} rows for {model_key} from {path}")
        else:
            print(f"  MISSING: {path} -- {model_key} hasn't been run/pulled yet, skipping")
    if not frames:
        raise FileNotFoundError(f"No per-model result files found in {output_dir}/ -- try `git pull` first.")
    combined = pd.concat(frames, ignore_index=True)
    combined_path = f"{output_dir}/multi_llm_all_results.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined {len(combined)} rows total -> {combined_path}")
    missing = [m for m in models if m not in combined["model"].unique()]
    if missing:
        print(f"\nStill waiting on: {missing}")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--max-workers", type=int, default=10)
    args = parser.parse_args()

    combined = run_multi_model_eval(args.input, models=args.models, n=args.n, max_workers=args.max_workers)
    report_multi_model_metrics(combined)


if __name__ == "__main__":
    main()