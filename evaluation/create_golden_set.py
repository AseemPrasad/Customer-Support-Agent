#!/usr/bin/env python
"""Generate a stratified 200-example golden set for manual labeling.

Samples from data/processed/amazon_resolved_pairs.parquet:
  - 120 English examples stratified by intent:
      30 order_tracking_delay, 30 damaged_defective_item,
      30 billing_unauthorized_charge, 15 legal_regulatory_churn, 15 PII leaks
  - 80 non-English: 20 ja, 15 es, 15 fr, 15 pt, 15 hi

Outputs:
  - data/golden_set/golden_eval_200.jsonl (one JSON record per line)
  - data/golden_set/golden_set_annotations_template.csv (for annotators)

Run:  python -m evaluation.create_golden_set
"""

import json
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = ROOT / "data" / "processed" / "amazon_resolved_pairs.parquet"
OUTPUT_DIR = ROOT / "data" / "golden_set"
JSONL_PATH = OUTPUT_DIR / "golden_eval_200.jsonl"
CSV_PATH = OUTPUT_DIR / "golden_set_annotations_template.csv"

SEED = 42

ORDER_ID_PATTERN = re.compile(r"\b\d{3}-\d{7}-\d{7}\b")

# Intent keyword heuristics for the stratified English sample.
LEGAL_KEYWORDS = [
    "lawyer",
    "sue",
    "lawsuit",
    "attorney",
    "police",
    "chargeback",
    "consumer court",
    "fraud",
    "unauthorized charge",
    "account hacked",
    "data breach",
    "legal action",
    "solicitor",
]

TRACKING_KEYWORDS = [
    "tracking",
    "delay",
    "delayed",
    "late delivery",
    "where is my order",
    "not received",
    "never arrived",
    "lost package",
    "stuck in transit",
    "delivery date",
    "shipping",
    "package hasn't",
    "hasn't arrived",
    "hasnt arrived",
    "haven't received",
    "havent received",
]

DAMAGE_KEYWORDS = [
    "damaged",
    "defective",
    "broken",
    "cracked",
    "not working",
    "doesn't work",
    "doesnt work",
    "quality",
    "stains",
    "refurbished",
    "received used",
    "wrong item",
    "missing parts",
]

BILLING_KEYWORDS = [
    "charged twice",
    "double charge",
    "double charged",
    "billing",
    "bank statement",
    "credit card",
    "wrong amount",
    "overcharged",
    "over-charged",
    "unexpected charge",
    "extra charge",
    "refund",
    "charge on my",
    "deducted",
]

NON_ENGLISH_QUOTA = {"ja": 20, "es": 15, "fr": 15, "pt": 15, "hi": 15}
ENGLISH_QUOTA = {
    "order_tracking_delay": 30,
    "damaged_defective_item": 30,
    "billing_unauthorized_charge": 30,
    "legal_regulatory_churn": 15,
    "pii_leak": 15,
}


def classify_en_intent(text: str) -> str:
    """Heuristically classify an English tweet into a golden-set intent bucket."""
    lower = text.lower()
    if ORDER_ID_PATTERN.search(text):
        return "pii_leak"
    if any(k in lower for k in LEGAL_KEYWORDS):
        return "legal_regulatory_churn"
    if any(k in lower for k in TRACKING_KEYWORDS):
        return "order_tracking_delay"
    if any(k in lower for k in DAMAGE_KEYWORDS):
        return "damaged_defective_item"
    if any(k in lower for k in BILLING_KEYWORDS):
        return "billing_unauthorized_charge"
    return "general_inquiry"


def make_entry(test_id: int, tweet: str, lang: str, source_intent: str) -> dict:
    return {
        "test_id": test_id,
        "input_tweet": tweet,
        "expected_language": lang,
        "gold_intent": None,
        "gold_decision": None,
        "gold_reason": None,
        "required_routing": None,
        "forbidden_elements": [],
        "source_intent": source_intent,  # sampling bucket; not part of the gold label
    }


def build_dataset(df: pd.DataFrame) -> list[dict]:
    random.seed(SEED)

    en = df[df["customer_lang"] == "en"].copy()
    en["intent_bucket"] = en["customer_text"].apply(classify_en_intent)
    jp = df[df["customer_lang"] == "ja"]
    es = df[df["customer_lang"] == "es"]
    fr = df[df["customer_lang"] == "fr"]
    pt = df[df["customer_lang"] == "pt"]
    hi = df[df["customer_lang"] == "hi"]

    entries: list[dict] = []
    test_id = 1

    def take(pool: pd.DataFrame, intent: str, n: int, lang: str) -> None:
        nonlocal test_id
        subset = pool[pool["intent_bucket"] == intent] if intent else pool
        n = min(n, len(subset))
        for tweet in subset["customer_text"].sample(n=n, random_state=SEED).tolist():
            entries.append(make_entry(test_id, tweet, lang, intent))
            test_id += 1
        if n < (ENGLISH_QUOTA[intent] if intent else n):
            print(f"  WARNING: only {n} samples available for en/{intent}")

    print("Sampling English strata...")
    for intent, quota in ENGLISH_QUOTA.items():
        take(en, intent, quota, "en")

    print("Sampling non-English...")
    for lang, quota in NON_ENGLISH_QUOTA.items():
        pool = {"ja": jp, "es": es, "fr": fr, "pt": pt, "hi": hi}[lang]
        n = min(quota, len(pool))
        for tweet in pool["customer_text"].sample(n=n, random_state=SEED).tolist():
            entries.append(make_entry(test_id, tweet, lang, lang))
            test_id += 1
        if n < quota:
            print(f"  WARNING: only {n} samples available for {lang}")

    return entries


def write_jsonl(entries: list[dict]) -> None:
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries)
    print(f"Wrote {len(entries)} records -> {JSONL_PATH}")


def write_csv_template(entries: list[dict]) -> None:
    header = (
        "# INSTRUCTIONS\n"
        "# ------------\n"
        "# Fill columns below for each test_id:\n"
        "#   gold_intent: order_tracking_delay | damaged_defective_item | "
        "billing_unauthorized_charge |\n"
        "#                legal_regulatory_churn | account_access_security | general_inquiry\n"
        "#   gold_decision: AUTO_REPLY or ESCALATE\n"
        "#   gold_reason: leave NONE for AUTO_REPLY, else a short reason "
        "(e.g. 'PII order id', 'legal threat')\n"
        "#   required_routing: team to route to if ESCALATE, else NA\n"
        "#   forbidden_elements: comma-separated list of things the reply MUST NOT contain\n"
        "#                       (e.g. 'order id PII', 'email', 'legal admission', "
        "'refund promise')\n"
        "#   agent_notes: free text for ambiguity/tricky cases\n"
        "# Do not modify test_id, input_tweet or expected_language.\n"
        "# The source_intent column is only the sampling bucket, NOT the gold label.\n"
    )
    rows = []
    for e in entries:
        tweet = e["input_tweet"].replace("\r", " ").replace("\n", " ").replace('"', "'").strip()
        rows.append(
            {
                "test_id": e["test_id"],
                "input_tweet": tweet,
                "expected_language": e["expected_language"],
                "gold_intent": "",
                "gold_decision": "",
                "gold_reason": "",
                "required_routing": "",
                "forbidden_elements": "",
                "agent_notes": "",
                "source_intent": e["source_intent"],
            }
        )
    df = pd.DataFrame(rows)
    csv_text = header + df.to_csv(index=False, lineterminator="\n")
    with open(CSV_PATH, "w", encoding="utf-8-sig") as f:
        f.write(csv_text)
    print(f"Wrote CSV template -> {CSV_PATH}")


def main() -> None:
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Missing processed pairs: {PARQUET_PATH}")

    print(f"Loading {PARQUET_PATH} ...")
    df = pd.read_parquet(PARQUET_PATH, columns=["customer_text", "customer_lang"])

    entries = build_dataset(df)
    if len(entries) != 200:
        raise RuntimeError(f"Expected 200 entries, got {len(entries)}")

    lang_counts: dict[str, int] = {}
    intent_counts: dict[str, int] = {}
    for e in entries:
        lang_counts[e["expected_language"]] = lang_counts.get(e["expected_language"], 0) + 1
        intent_counts[e["source_intent"]] = intent_counts.get(e["source_intent"], 0) + 1
    print("\nFinal composition:")
    print("  languages:", dict(sorted(lang_counts.items())))
    print("  intents:  ", dict(sorted(intent_counts.items())))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(entries)
    write_csv_template(entries)


if __name__ == "__main__":
    main()
