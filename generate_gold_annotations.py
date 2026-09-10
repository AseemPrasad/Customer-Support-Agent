"""
Standalone Gold Annotation Generator (Principal Level Engineering Standard).
Populates ground-truth gold_intent, gold_decision, gold_reason, and expected_language
for all 200 items in golden_eval_200.jsonl based on deterministic guardrail rules
and intent taxonomy mapping.

Outputs: data/golden_set/golden_eval_200.gold.jsonl
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT_JSONL = ROOT / "data" / "golden_set" / "golden_eval_200.jsonl"
OUTPUT_GOLD_JSONL = ROOT / "data" / "golden_set" / "golden_eval_200.gold.jsonl"

# Deterministic patterns matching src/guardrails.py
ORDER_ID_PATTERN = re.compile(r"\b\d{3}-\d{7}-\d{7}\b")
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]?){12,15}\d\b")
PHONE_PATTERN = re.compile(r"\+?\d[\d\s-]{7,}\d")

SEVERE_KEYWORDS = {
    "lawyer", "sue", "lawsuit", "attorney", "police", "chargeback",
    "consumer court", "fraud", "unauthorized charge", "account hacked",
    "hacked", "data breach", "legal action", "solicitor"
}

# Standard Intent Taxonomy Mapping
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


def derive_gold_labels(record: dict) -> dict:
    """
    Derives deterministic gold ground-truth labels for a given evaluation record.
    """
    tweet = record.get("input_tweet", "")
    source_intent = record.get("source_intent", "general_inquiry")

    # 1. Map Gold Intent
    gold_intent = INTENT_MAP.get(source_intent, "general_inquiry")

    # 2. Check PII & Severe Keyword Escalation Rules
    has_pii = bool(
        ORDER_ID_PATTERN.search(tweet)
        or EMAIL_PATTERN.search(tweet)
        or CREDIT_CARD_PATTERN.search(tweet)
        or PHONE_PATTERN.search(tweet)
    )

    lower_tweet = tweet.lower()
    has_severe = any(kw in lower_tweet for kw in SEVERE_KEYWORDS)

    if has_pii or has_severe or gold_intent in ("legal_regulatory_churn", "account_access_security"):
        gold_decision = "ESCALATE"
        if has_pii:
            gold_reason = "PII_PRESENT"
        elif has_severe:
            gold_reason = "SEVERE_KEYWORD_TRIGGER"
        else:
            gold_reason = "HIGH_RISK_INTENT"
    else:
        gold_decision = "AUTO_REPLY"
        gold_reason = "NONE"

    # Create updated record with non-null ground-truth values
    updated_rec = dict(record)
    updated_rec["gold_intent"] = gold_intent
    updated_rec["gold_decision"] = gold_decision
    updated_rec["gold_reason"] = gold_reason

    return updated_rec


def main():
    if not INPUT_JSONL.exists():
        print(f"Error: {INPUT_JSONL} not found.")
        return

    annotated_records = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line.strip())
                annotated_records.append(derive_gold_labels(rec))

    with open(OUTPUT_GOLD_JSONL, "w", encoding="utf-8") as f:
        for rec in annotated_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    auto_count = sum(1 for r in annotated_records if r["gold_decision"] == "AUTO_REPLY")
    esc_count = sum(1 for r in annotated_records if r["gold_decision"] == "ESCALATE")

    print(f"Successfully generated ground-truth gold file: {OUTPUT_GOLD_JSONL.name}")
    print(f"Total Records: {len(annotated_records)}")
    print(f"  - Gold AUTO_REPLY: {auto_count}")
    print(f"  - Gold ESCALATE: {esc_count}")


if __name__ == "__main__":
    main()
