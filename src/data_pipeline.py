import argparse
import re
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0

SIGNOFF_PATTERN = re.compile(r"[\^A-Z]{2,3}\s*$")
SIGNOFF_WHOLE_PATTERN = re.compile(r"^\s*[\^A-Z]{2,3}\s*$")
SIGNOFF_STRIP_PATTERN = re.compile(r"\s*[\^A-Z]{2,3}\s*$")


def is_signoff(text: str) -> bool:
    cleaned = text.strip()
    if len(cleaned) < 20:
        return True
    if SIGNOFF_WHOLE_PATTERN.match(cleaned):
        return True
    return cleaned.lower() in ("you're welcome", "thank you", "thanks", "you welcome", "ur welcome")


def strip_signoff(text: str) -> str:
    return SIGNOFF_STRIP_PATTERN.sub("", text).strip()


def detect_lang(text: str) -> str:
    try:
        return detect(text)
    except Exception:  # noqa: BLE001 - langdetect may fail on short/garbled text
        return "unknown"


def build_pipeline(csv_path: str, max_rows: int | None = None) -> pd.DataFrame:
    print(f"Reading CSV: {csv_path}" + (f" (max_rows={max_rows})" if max_rows else ""))
    read_kwargs = {"low_memory": False}
    if max_rows:
        read_kwargs["nrows"] = max_rows
    df = pd.read_csv(csv_path, **read_kwargs)
    print(f"  Total rows loaded: {len(df):,}")

    df["created_at"] = pd.to_datetime(df["created_at"], format="mixed", utc=True)

    agent_replies = df[
        (df["author_id"] == "AmazonHelp")
        & (~df["inbound"])
        & (df["in_response_to_tweet_id"].notna())
    ].copy()
    agent_replies["in_response_to_tweet_id"] = agent_replies["in_response_to_tweet_id"].astype(int)
    print(f"  Agent replies (AmazonHelp, inbound=False, has response_to): {len(agent_replies):,}")

    customer_tweets = df[
        (df["inbound"]) & (df["tweet_id"].isin(agent_replies["in_response_to_tweet_id"]))
    ].copy()
    print(f"  Matching customer tweets found: {len(customer_tweets):,}")

    agent_replies = agent_replies.rename(
        columns={"tweet_id": "agent_tweet_id", "text": "agent_text_raw"}
    )
    customer_tweets = customer_tweets.rename(columns={"author_id": "customer_id"})
    merged = agent_replies.merge(
        customer_tweets[["tweet_id", "text", "customer_id"]],
        left_on="in_response_to_tweet_id",
        right_on="tweet_id",
        how="inner",
    )
    print(f"  Merged pairs: {len(merged):,}")

    merged["agent_text_clean"] = merged["agent_text_raw"].apply(strip_signoff)
    merged = merged[~merged["agent_text_clean"].apply(is_signoff)].copy()
    print(f"  After removing sign-offs: {len(merged):,}")

    merged = merged.sort_values("created_at")
    merged = merged.drop_duplicates(subset="in_response_to_tweet_id", keep="first")
    print(f"  First-turn only (per customer tweet): {len(merged):,}")

    result = pd.DataFrame(
        {
            "customer_text": merged["text"].values,
            "agent_text": merged["agent_text_clean"].values,
            "agent_text_raw": merged["agent_text_raw"].values,
            "customer_id": merged["customer_id"].values,
            "tweet_id": merged["in_response_to_tweet_id"].values,
        }
    )
    result = result.dropna(subset=["customer_text", "agent_text"])
    result = result[result["customer_text"].str.strip().str.len() > 0]
    result = result[result["agent_text"].str.strip().str.len() > 0]
    print(f"  Final pairs: {len(result):,}")

    print("\nLanguage distribution (customer_text):")
    langs = result["customer_text"].apply(detect_lang)
    lang_counts = langs.value_counts()
    for lang, count in lang_counts.items():
        pct = count / len(result) * 100
        print(f"  {lang}: {count:,} ({pct:.1f}%)")
    result["customer_lang"] = langs

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract AmazonHelp customer-agent pairs from twcs.csv"
    )
    parser.add_argument("--input", default="data/raw/twcs.csv", help="Path to twcs.csv")
    parser.add_argument(
        "--output",
        default="data/processed/amazon_resolved_pairs.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None, help="Max rows to read from CSV (for testing)"
    )
    args = parser.parse_args()

    result = build_pipeline(args.input, max_rows=args.max_rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"\nSaved {len(result):,} pairs to {output_path}")


if __name__ == "__main__":
    main()
