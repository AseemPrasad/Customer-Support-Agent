"""
Generator for a 500-Sample Stratified Golden Benchmark Evaluation Dataset.
Creates data/golden_set/golden_eval_500.gold.jsonl with non-null ground-truth labels.
"""

import json
import random
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
PARQUET_PATH = ROOT / "data" / "processed" / "amazon_resolved_pairs.parquet"
OUTPUT_JSONL = ROOT / "data" / "golden_set" / "golden_eval_500.gold.jsonl"

ORDER_ID_PATTERN = re.compile(r"\b\d{3}-\d{7}-\d{7}\b")
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]?){12,15}\d\b")
PHONE_PATTERN = re.compile(r"\+?\d[\d\s-]{7,}\d")

SEVERE_KEYWORDS = {
    "lawyer", "sue", "lawsuit", "attorney", "police", "chargeback",
    "consumer court", "fraud", "unauthorized charge", "account hacked",
    "hacked", "data breach", "legal action", "solicitor"
}

INTENT_MAP = {
    "order_tracking_delay": "order_status",
    "damaged_defective_item": "product_quality_complaint",
    "billing_unauthorized_charge": "payments_billing",
    "legal_regulatory_churn": "legal_regulatory_churn",
    "account_access_security": "account_access_security",
    "refund_related": "refund_related",
    "order_status": "order_status",
    "general_inquiry": "general_inquiry",
}


def main():
    if not PARQUET_PATH.exists():
        print(f"Error: {PARQUET_PATH} not found.")
        return

    df = pd.read_parquet(PARQUET_PATH)
    df = df.drop_duplicates(subset=["customer_text"]).reset_index(drop=True)

    # 350 English, 150 Multi-lingual
    en_df = df[df["customer_lang"] == "en"].sample(n=350, random_state=42)
    multi_df = df[df["customer_lang"].isin({"ja", "es", "fr", "pt", "hi"})].sample(n=150, random_state=42)

    sample_df = pd.concat([en_df, multi_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    records = []
    for idx, row in sample_df.iterrows():
        tweet = str(row["customer_text"])
        lang = str(row["customer_lang"])
        source_intent = str(row.get("source_intent", "general_inquiry"))

        gold_intent = INTENT_MAP.get(source_intent, "general_inquiry")

        has_pii = bool(
            ORDER_ID_PATTERN.search(tweet)
            or EMAIL_PATTERN.search(tweet)
            or CREDIT_CARD_PATTERN.search(tweet)
            or PHONE_PATTERN.search(tweet)
        )
        has_severe = any(kw in tweet.lower() for kw in SEVERE_KEYWORDS)

        if has_pii or has_severe or gold_intent in ("legal_regulatory_churn", "account_access_security"):
            gold_decision = "ESCALATE"
            gold_reason = "PII_OR_SEVERE_SECURITY"
        else:
            gold_decision = "AUTO_REPLY"
            gold_reason = "NONE"

        rec = {
            "test_id": idx + 1,
            "input_tweet": tweet,
            "expected_language": lang,
            "gold_intent": gold_intent,
            "gold_decision": gold_decision,
            "gold_reason": gold_reason,
            "source_intent": source_intent
        }
        records.append(rec)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    auto_c = sum(1 for r in records if r["gold_decision"] == "AUTO_REPLY")
    esc_c = sum(1 for r in records if r["gold_decision"] == "ESCALATE")

    print(f"Successfully generated 500-sample benchmark dataset: {OUTPUT_JSONL.name}")
    print(f"Total Records: {len(records)}")
    print(f"  - Gold AUTO_REPLY: {auto_c}")
    print(f"  - Gold ESCALATE: {esc_c}")


if __name__ == "__main__":
    main()
