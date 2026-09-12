"""
clean_data.py  --  Step 1 of the pipeline (Sadman)

Takes the raw Kaggle "Sentiment Analysis for Mental Health" CSV and produces a
cleaned CSV that every later step reads.

This module is written to be IMPORTED from a notebook
(notebooks/01_data_cleaning_eda.ipynb), but also runs standalone from the
command line if you ever want that:
    python clean_data.py --input data/raw/Combined_Data.csv --output data/processed/clean.csv
"""

import argparse                      # parse command-line flags (--input, --output)
import os                            # create output folders if they don't exist
import re                            # regular expressions, used to strip URLs/HTML
import pandas as pd                  # dataframes: load, filter, and save tabular data

# The seven labels the paper uses. We keep only rows whose status is one of these.
CATEGORIES = ["Normal", "Depression", "Anxiety", "Suicidal", "Stress", "Bipolar", "Personality disorder"]


def clean_text(text):
    """Reproduce the paper's preprocessing for a single post."""
    text = str(text)                                 # force to string (guards against NaN/float cells)
    text = re.sub(r"http\S+|www\.\S+", " ", text)    # delete URLs (http... or www...)
    text = re.sub(r"<.*?>", " ", text)               # delete HTML tags like <br> or <p>
    text = re.sub(r"[^A-Za-z0-9\s.,!?']", " ", text) # drop unsupported special chars, keep basic punctuation
    text = re.sub(r"\s+", " ", text)                 # collapse any run of whitespace into one space
    return text.strip()                              # remove leading/trailing spaces


def clean_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Do the full cleaning pass and save the result. Returns the cleaned
    DataFrame too, so a notebook can immediately inspect/plot it without
    re-reading the CSV from disk.
    """
    # The Kaggle file has an unnamed index column plus 'statement' and 'status'.
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} raw rows. Columns: {list(df.columns)}")

    # Normalise column names in case the file uses different capitalisation.
    df.columns = [c.strip().lower() for c in df.columns]
    # Keep only the two columns we care about.
    df = df[["statement", "status"]]

    # Drop rows with a missing post or missing label -- they can't be used.
    df = df.dropna(subset=["statement", "status"])

    # Apply the text cleaner to every post.
    df["statement"] = df["statement"].apply(clean_text)

    # Keep only rows whose label is one of the seven expected categories
    # (protects against stray/mis-spelled labels in the raw file).
    df = df[df["status"].isin(CATEGORIES)]

    # Remove posts shorter than 10 characters, exactly as the paper does.
    df = df[df["statement"].str.len() >= 10]

    # Drop exact duplicate posts so the same text can't land in both train and test.
    df = df.drop_duplicates(subset=["statement"]).reset_index(drop=True)

    # Save the cleaned data for the next step (create the folder first if needed).
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} cleaned rows to {output_path}")

    return df


def print_class_distribution(df: pd.DataFrame):
    """Print counts + percentages -- compare this to Table I of the paper."""
    print("Class distribution:")
    counts = df["status"].value_counts()
    for label, n in counts.items():
        print(f"  {label:22s} {n:6d}  ({n / len(df) * 100:.1f}%)")


def main():
    """CLI entry point -- optional, the notebook is the primary interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the raw Kaggle CSV")
    parser.add_argument("--output", required=True, help="Where to write the cleaned CSV")
    args = parser.parse_args()
    df = clean_dataset(args.input, args.output)
    print_class_distribution(df)


if __name__ == "__main__":     # only run main() when the file is executed directly
    main()
