"""
Standalone Principal-Level Smart Sampler & Ingestion Utility.
100% Non-destructive: Reads amazon_resolved_pairs.parquet, performs language-stratified
and text-diversity sampling to select 5,000 optimal pairs, and ingests them into ChromaDB.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vector_store import KnowledgeBase

PARQUET_PATH = ROOT / "data" / "processed" / "amazon_resolved_pairs.parquet"
TARGET_TOTAL = 5000


def create_smart_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform stratified sampling across languages and text length buckets.
    """
    print(f"Loaded raw dataset with {len(df):,} pairs.")

    # 1. Deduplicate identical customer inquiries
    df = df.drop_duplicates(subset=["customer_text"]).reset_index(drop=True)
    print(f"After deduplicating customer text: {len(df):,} pairs.")

    # 2. Extract multi-lingual pairs (JA, ES, FR, PT, HI)
    target_langs = {"ja", "es", "fr", "pt", "hi"}
    multilingual_df = df[df["customer_lang"].isin(target_langs)]
    print(f"Found {len(multilingual_df):,} target multi-lingual pairs.")

    # Cap multi-lingual at ~1,500 rows stratified by language
    multilingual_samples = []
    for lang, group in multilingual_df.groupby("customer_lang"):
        cap = min(len(group), 300)
        multilingual_samples.append(group.sample(n=cap, random_state=42))

    sampled_multilingual = pd.concat(multilingual_samples) if multilingual_samples else pd.DataFrame()

    # 3. Sample English pairs stratified by inquiry text length (short, medium, detailed complaints)
    en_df = df[df["customer_lang"] == "en"].copy()
    en_df["char_len"] = en_df["customer_text"].str.len()

    # Create 3 length quantiles for diversity
    en_df["len_bucket"] = pd.qcut(en_df["char_len"], q=3, labels=["short", "medium", "detailed"])

    needed_en = TARGET_TOTAL - len(sampled_multilingual)
    en_samples = []
    per_bucket = needed_en // 3

    for bucket_name, group in en_df.groupby("len_bucket"):
        n_sample = min(len(group), per_bucket)
        en_samples.append(group.sample(n=n_sample, random_state=42))

    sampled_en = pd.concat(en_samples) if en_samples else pd.DataFrame()

    # Combine into final target dataset
    final_df = pd.concat([sampled_multilingual, sampled_en]).reset_index(drop=True)
    # Clean up temporary columns if present
    cols_to_drop = [c for c in ["char_len", "len_bucket"] if c in final_df.columns]
    if cols_to_drop:
        final_df = final_df.drop(columns=cols_to_drop)

    print(f"Created final smart sample: {len(final_df):,} pairs across {final_df['customer_lang'].nunique()} languages.")
    return final_df


def main():
    if not PARQUET_PATH.exists():
        print(f"Error: Parquet file not found at {PARQUET_PATH}")
        sys.exit(1)

    df = pd.read_parquet(PARQUET_PATH)
    smart_df = create_smart_sample(df)

    print("\nIngesting smart sample into ChromaDB vector store...")
    kb = KnowledgeBase()
    total_ingested = kb.ingest(smart_df, force=True)

    print(f"\nSUCCESS! Ingested {total_ingested:,} stratified resolution pairs.")
    print(f"ChromaDB collection now contains {kb.count():,} documents.")


if __name__ == "__main__":
    main()
