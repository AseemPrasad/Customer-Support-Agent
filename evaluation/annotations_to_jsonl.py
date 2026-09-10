#!/usr/bin/env python
"""Convert an annotated golden-set CSV back into a gold-labeled JSONL.

Reads a filled-in annotation template (data/golden_set/golden_set_annotations_template.csv)
and writes a JSONL with gold labels merged into scoring-ready records.

The CSV is produced by `python -m evaluation.create_golden_set`. Fill in the
gold_intent / gold_decision / gold_reason / required_routing / forbidden_elements
columns, then run:

    python -m evaluation.annotations_to_jsonl \
        --csv data/golden_set/golden_set_annotations_template.csv \
        --output data/golden_set/golden_eval_200.gold.jsonl

Annotators can optionally batch work in multiple CSV files; pass
`--output_dir` to instead write one JSONL per CSV with a ".gold.jsonl" suffix.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

BOOL_MAP = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}


def parse_bool(value: object) -> bool | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        return value
    return BOOL_MAP.get(str(value).strip().lower())


def parse_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    if "," in text:
        return [p.strip() for p in text.split(",") if p.strip()]
    return [text] if text else []


def convert(csv_path: Path, output_path: Path) -> int:
    """Convert one annotated CSV into a gold-labeled JSONL.

    Returns the number of records written.
    """
    text = csv_path.read_text(encoding="utf-8-sig")
    body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    df = pd.read_csv(pd.io.common.StringIO(body))

    rows = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            record = {
                "test_id": int(r["test_id"]),
                "input_tweet": str(r["input_tweet"]),
                "expected_language": str(r["expected_language"]),
                "gold_intent": (
                    str(r["gold_intent"]).strip()
                    if isinstance(r["gold_intent"], str) and r["gold_intent"].strip()
                    else None
                ),
                "gold_decision": (
                    str(r["gold_decision"]).strip().upper()
                    if isinstance(r["gold_decision"], str) and r["gold_decision"].strip()
                    else None
                ),
                "gold_reason": (
                    str(r["gold_reason"]).strip()
                    if isinstance(r["gold_reason"], str) and r["gold_reason"].strip()
                    else None
                ),
                "required_routing": parse_list(r["required_routing"]),
                "forbidden_elements": parse_list(r["forbidden_elements"]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert annotated golden-set CSV(s) to JSONL")
    parser.add_argument("--csv", type=Path, help="Path to a single annotated CSV")
    parser.add_argument("--input-dir", type=Path, help="Directory of annotated CSVs (batch mode)")
    parser.add_argument("--output", type=Path, help="Output JSONL path (single-file mode)")
    parser.add_argument("--output-dir", type=Path, help="Output directory (batch mode)")
    args = parser.parse_args()

    csv_paths: list[Path] = []
    if args.csv:
        csv_paths = [args.csv]
    elif args.input_dir:
        csv_paths = sorted(args.input_dir.glob("*.csv"))
    else:
        raise SystemExit("Provide --csv or --input-dir.")

    for csv_path in csv_paths:
        if args.output_dir:
            output = args.output_dir / (csv_path.stem + ".gold.jsonl")
        else:
            output = args.output
        if not output:
            raise SystemExit("Provide --output or --output-dir.")
        n = convert(csv_path, output)
        print(f"Wrote {n} records -> {output}")


if __name__ == "__main__":
    main()
