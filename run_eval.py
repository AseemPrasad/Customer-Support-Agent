#!/usr/bin/env python
"""Main evaluation runner.

Loads a golden-set JSONL, runs the AmazonSupportAgent on each example,
computes classification/guardrail metrics, optionally runs the LLM judge,
and writes a JSON scorecard.

Run examples:
    python run_eval.py --dataset data/golden_set/golden_eval_200.jsonl --samples 10
    python run_eval.py --dataset data/golden_set/golden_eval_200.jsonl --samples 200 \\
        --judge-samples 50 --kb-dir ./amazon_kb --output data/golden_set/scorecard.json
    python run_eval.py --dataset data/golden_set/golden_eval_200.jsonl --no-judge
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.automated_metrics import (
    check_pii_leakage,
    compute_escalation_fnr,
    compute_intent_metrics,
)
from evaluation.llm_judge import EvaluationVerdict, JudgeEvaluator
from src.config import Settings
from src.llm_adapter import LLMClient
from src.pipeline import AgentDecision, AmazonSupportAgent
from src.vector_store import KnowledgeBase

DEFAULT_DATASET = Path("data/golden_set/golden_eval_200.jsonl")
DEFAULT_OUTPUT = Path("data/golden_set/scorecard.json")
PROCESSED_PAIRS = Path("data/processed/amazon_resolved_pairs.parquet")

JUDGE_SUBSET_MAX = 50


def load_golden_set(path: Path, samples: int | None) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Golden set not found: {path}")
    with open(path, encoding="utf-8") as f:
        all_rows = [json.loads(line) for line in f if line.strip()]
    if samples is not None:
        all_rows = all_rows[:samples]
    return all_rows


def ensure_kb_populated(kb: KnowledgeBase, ingest_max_rows: int | None) -> None:
    """Ingest the processed pairs if the collection is empty."""
    if kb.count() > 0:
        return
    if not PROCESSED_PAIRS.exists():
        raise FileNotFoundError(
            f"Collection empty and processed pairs missing: {PROCESSED_PAIRS}. "
            "Run: python -m src.data_pipeline"
        )
    df = pd.read_parquet(
        PROCESSED_PAIRS,
        columns=["customer_text", "agent_text", "customer_id", "tweet_id"],
    )
    if ingest_max_rows:
        df = df.head(ingest_max_rows)
    print(f"Populating knowledge base with {len(df):,} pairs ...")
    kb.ingest(df, force=True)


def run_agent_tasks(agent: AmazonSupportAgent, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    total = len(rows)
    for i, row in enumerate(rows, start=1):
        tick = time.time()
        try:
            decision: AgentDecision = agent.process_ticket(row["input_tweet"])
            error = None
        except Exception as exc:  # noqa: BLE001 - record and continue
            decision = None
            error = f"{exc.__class__.__name__}: {exc}"
        elapsed = time.time() - tick
        predictions.append(
            {
                "test_id": row.get("test_id"),
                "input_tweet": row.get("input_tweet"),
                "expected_language": row.get("expected_language"),
                "gold_intent": row.get("gold_intent"),
                "gold_decision": row.get("gold_decision"),
                "gold_reason": row.get("gold_reason"),
                "error": error,
                "pred": None if decision is None else decision.model_dump(),
                "elapsed_s": round(elapsed, 3),
            }
        )
        status = error or f"{decision.decision}/{decision.intent}"
        print(f"  [{i}/{total}] row {row.get('test_id')} -> {status} ({elapsed:.1f}s)")
    return predictions


def compute_core_metrics(
    predictions: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    intents = [
        (r["gold_intent"], p["pred"]["intent"])
        for r, p in zip(rows, predictions, strict=True)
        if p["pred"] is not None and isinstance(r.get("gold_intent"), str)
    ]
    decisions = [
        (r["gold_decision"], p["pred"]["decision"])
        for r, p in zip(rows, predictions, strict=True)
        if p["pred"] is not None and isinstance(r.get("gold_decision"), str)
    ]

    intent_report = None
    if intents:
        y_true = [t for t, _ in intents]
        y_pred = [p for _, p in intents]
        intent_report = compute_intent_metrics(y_true, y_pred)

    escalation = None
    if decisions:
        y_true_d = [t for t, _ in decisions]
        y_pred_d = [p for _, p in decisions]
        escalation = compute_escalation_fnr(y_true_d, y_pred_d)
        escalation["confusion_matrix"] = escalation["confusion_matrix"].tolist()

    drafts = [p["pred"]["draft_reply"] for p in predictions if p["pred"] is not None]
    pii_hits = [t for t in drafts if check_pii_leakage(t)]
    pii_rate = len(pii_hits) / max(len(drafts), 1)

    return {
        "intent_metrics": intent_report,
        "escalation_fnr": escalation
        and {
            k: escalation[k]
            for k in (
                "fnr",
                "false_negatives",
                "true_positives",
                "false_positives",
                "true_negatives",
                "confusion_matrix",
            )
        },
        "pii_leakage_count": len(pii_hits),
        "pii_leakage_rate": round(pii_rate, 4),
        "num_drafts": len(drafts),
        "num_errors": sum(1 for p in predictions if p["error"]),
    }


def run_judge(
    judge: JudgeEvaluator, predictions: list[dict[str, Any]], max_pairs: int | None
) -> dict[str, Any]:
    valid = [p for p in predictions if p["pred"] is not None and not p["error"]]
    if not valid:
        return {"evaluated": 0}

    if max_pairs is not None and len(valid) > max_pairs:
        subset = random.sample(valid, max_pairs)
    else:
        subset = valid

    verdicts: list[EvaluationVerdict] = []
    failures = 0
    for i, p in enumerate(subset, start=1):
        try:
            v = judge.evaluate(p["input_tweet"], p["pred"]["draft_reply"])
            verdicts.append(v)
        except Exception:  # noqa: BLE001
            failures += 1
        print(f"  [judge {i}/{len(subset)}] done")

    if not verdicts:
        return {"evaluated": len(subset), "failures": failures}

    return {
        "evaluated": len(subset),
        "failures": failures,
        "avg_brand_voice_score": round(
            sum(v.brand_voice_score for v in verdicts) / len(verdicts), 3
        ),
        "avg_factual_policy_adherence": round(
            sum(v.factual_policy_adherence for v in verdicts) / len(verdicts), 3
        ),
        "pii_safety_pass_rate": round(sum(v.pii_safety_pass for v in verdicts) / len(verdicts), 3),
        "language_match_pass_rate": round(
            sum(v.language_match_pass for v in verdicts) / len(verdicts), 3
        ),
    }


def write_scorecard(output: Path, scorecard: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2, ensure_ascii=False)
    print(f"Scorecard written to {output}")


def print_summary(scorecard: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION SCORECARD")
    print("=" * 60)
    print(f"Samples evaluated : {scorecard['num_samples']}")
    print(f"Agent errors      : {scorecard['core_metrics']['num_errors']}")
    print(
        f"PII leakage rate  : {scorecard['core_metrics']['pii_leakage_rate']} "
        f"({scorecard['core_metrics']['pii_leakage_count']}/{scorecard['core_metrics']['num_drafts']})"
    )

    intent = scorecard["core_metrics"]["intent_metrics"]
    if intent:
        macro = intent["macro_avg"]
        print(
            f"Intent macro F1   : {macro['f1']:.3f} "
            f"(P {macro['precision']:.3f}, R {macro['recall']:.3f})"
        )
    else:
        print("Intent macro F1   : n/a (no gold_intent labels)")

    esc = scorecard["core_metrics"]["escalation_fnr"]
    if esc:
        print(
            f"Escalation FNR    : {esc['fnr']:.3f} (TP {esc['true_positives']}, "
            f"FN {esc['false_negatives']}, FP {esc['false_positives']}, "
            f"TN {esc['true_negatives']})"
        )
    else:
        print("Escalation FNR    : n/a (no gold_decision labels)")

    judge = scorecard.get("judge_metrics")
    if judge and judge.get("evaluated"):
        print(
            f"Judge (n={judge['evaluated']})   : "
            f"brand voice {judge['avg_brand_voice_score']:.2f}/5, "
            f"policy {judge['avg_factual_policy_adherence']:.2f}/5, "
            f"PII pass {judge['pii_safety_pass_rate']:.2%}, "
            f"lang pass {judge['language_match_pass_rate']:.2%}"
        )
    else:
        print("Judge             : skipped")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full support-agent evaluation")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Golden set JSONL path")
    parser.add_argument(
        "--samples", type=int, default=None, help="Number of samples (default: all)"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Scorecard JSON output path")
    parser.add_argument("--no-judge", dest="judge", action="store_false", help="Disable LLM judge")
    parser.set_defaults(judge=True)
    parser.add_argument(
        "--judge-samples",
        type=int,
        default=JUDGE_SUBSET_MAX,
        help="Max pairs judged (random subset; default %(default)s)",
    )
    parser.add_argument("--kb-dir", default="./amazon_kb", help="Knowledge base directory")
    parser.add_argument(
        "--ingest-max-rows",
        type=int,
        default=2000,
        help="Max rows to ingest into an empty KB (0 = full set)",
    )
    parser.add_argument("--model", default=None, help="Override LOCAL_MODEL_NAME (e.g. llama3)")
    args = parser.parse_args()

    try:
        rows = load_golden_set(Path(args.dataset), args.samples)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(rows)} examples from {args.dataset}")

    kb = KnowledgeBase(persist_dir=args.kb_dir)
    if kb.count() == 0:
        ingest_max = None if args.ingest_max_rows == 0 else args.ingest_max_rows
        try:
            ensure_kb_populated(kb, ingest_max)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    print(f"Knowledge base ready: {kb.count():,} documents in {args.kb_dir}")

    settings = Settings()
    if args.model:
        settings.LOCAL_MODEL_NAME = args.model
    llm = LLMClient(settings=settings)
    agent = AmazonSupportAgent(kb=kb, llm=llm, top_k=2)

    predictions = run_agent_tasks(agent, rows)
    core = compute_core_metrics(predictions, rows)

    judge_metrics: dict[str, Any] = {"evaluated": 0}
    if args.judge:
        judge = JudgeEvaluator(llm=llm)
        judge_metrics = run_judge(judge, predictions, args.judge_samples)

    scorecard: dict[str, Any] = {
        "num_samples": len(rows),
        "dataset": str(args.dataset),
        "gold_labels_available": {
            "intent": sum(1 for r in rows if isinstance(r.get("gold_intent"), str)),
            "decision": sum(1 for r in rows if isinstance(r.get("gold_decision"), str)),
        },
        "core_metrics": core,
        "judge_metrics": judge_metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    write_scorecard(Path(args.output), scorecard)
    print_summary(scorecard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
