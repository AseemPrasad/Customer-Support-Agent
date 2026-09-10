"""Judge calibration: agreement between LLM judge scores and human scores.

Loads a calibration file (JSONL) with fields:
    inquiry, response,
    human_brand_voice_score, human_policy_adherence,
    human_pii_safety_pass, human_language_match_pass

Runs the LLM judge on every pair, then reports:
    - Cohen's quadratic weighted kappa for brand_voice_score and policy adherence
    - Binary classification accuracy for PII safety and language match

Run:  python -m evaluation.judge_calibration [--input PATH] [--max-pairs N] [--model NAME]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, cohen_kappa_score

from evaluation.llm_judge import EvaluationVerdict, JudgeEvaluator
from src.config import Settings
from src.llm_adapter import LLMClient

DEFAULT_INPUT = (
    Path(__file__).resolve().parent.parent / "data" / "golden_set" / "judge_calibration_50.jsonl"
)

REQUIRED_FIELDS = {
    "inquiry",
    "response",
    "human_brand_voice_score",
    "human_policy_adherence",
    "human_pii_safety_pass",
    "human_language_match_pass",
}


def load_calibration(path: Path, max_pairs: int | None = None) -> list[dict[str, Any]]:
    """
    Load calibration JSONL entries, validating required fields.

    Args:
        path: Path to the JSONL file.
        max_pairs: Optional cap on number of entries (for testing).

    Returns:
        List of entry dicts.

    Raises:
        FileNotFoundError with setup instructions if path is missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {path}\n"
            "\nTo create it, prepare a JSONL file with one JSON object per line, e.g.:\n"
            '{"inquiry": "...", "response": "...", "human_brand_voice_score": 4, '
            '"human_policy_adherence": 4, "human_pii_safety_pass": true, '
            '"human_language_match_pass": true}\n\n'
            "Pro tip: sample ~50 inquiry/response pairs from\n"
            "data/golden_set/golden_eval_200.jsonl (use the input_tweet + a generated\n"
            "reply), have a human score them, and store as "
            "data/golden_set/judge_calibration_50.jsonl"
        )

    entries: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            missing = REQUIRED_FIELDS - set(entry)
            if missing:
                sample = entry.get("inquiry", "?")[:40]
                raise ValueError(f"Entry {sample!r} missing fields: {sorted(missing)}")
            entries.append(entry)
            if max_pairs and len(entries) >= max_pairs:
                break
    return entries


def score_pair(
    judge: JudgeEvaluator, entry: dict[str, Any], index: int, total: int
) -> tuple[int, EvaluationVerdict | None]:
    """
    Judge one calibration pair, reporting progress and judge failures.

    Returns (index, verdict) with verdict None if judging failed.
    """
    print(f"  [{index + 1}/{total}] evaluating...")
    try:
        verdict = judge.evaluate(entry["inquiry"], entry["response"])
    except Exception as exc:  # noqa: BLE001 - surface any judge failure
        print(f"    judge error: {exc.__class__.__name__}: {exc}")
        return index, None
    return index, verdict


def compute_agreement(
    entries: list[dict[str, Any]], verdicts: list[EvaluationVerdict]
) -> dict[str, Any]:
    """
    Compute judge-human agreement metrics.

    Args:
        entries: Calibration entries (human scores).
        verdicts: Judge verdicts (aligned with entries).

    Returns:
        Dict with kappa and accuracy values plus row counts.
    """
    human_bv = [int(e["human_brand_voice_score"]) for e in entries]
    human_pa = [int(e["human_policy_adherence"]) for e in entries]
    human_pii = [bool(e["human_pii_safety_pass"]) for e in entries]
    human_lang = [bool(e["human_language_match_pass"]) for e in entries]

    llm_bv = [int(v.brand_voice_score) for v in verdicts]
    llm_pa = [int(v.factual_policy_adherence) for v in verdicts]
    llm_pii = [bool(v.pii_safety_pass) for v in verdicts]
    llm_lang = [bool(v.language_match_pass) for v in verdicts]

    return {
        "brand_voice_kappa": cohen_kappa_score(human_bv, llm_bv, weights="quadratic"),
        "policy_kappa": cohen_kappa_score(human_pa, llm_pa, weights="quadratic"),
        "pii_accuracy": accuracy_score(human_pii, llm_pii),
        "lang_accuracy": accuracy_score(human_lang, llm_lang),
        "n": len(entries),
    }


def print_report(results: dict[str, Any]) -> None:
    """Print a formatted calibration table."""
    n = results["n"]
    sep = "+" + "-" * 46 + "+" + "-" * 14 + "+"
    print()
    print(f"Judge <-> Human agreement (n = {n})")
    print(sep)
    print(f"| {'Metric':<44} | {'Score':>12} |")
    print(sep)
    rows = [
        ("Brand voice - Cohen quadratic kappa", results["brand_voice_kappa"]),
        ("Policy adherence - Cohen quadratic kappa", results["policy_kappa"]),
        ("PII safety - accuracy", results["pii_accuracy"]),
        ("Language match - accuracy", results["lang_accuracy"]),
    ]
    for label, value in rows:
        print(f"| {label:<44} | {value:>11.3f} |")
    print(sep)

    kappas = [k for k in (results["brand_voice_kappa"], results["policy_kappa"]) if k is not None]
    avg_k = sum(kappas) / max(len(kappas), 1)
    print("\nInterpretation: kappa >= 0.8 = strong, 0.6-0.8 = moderate.")
    print(
        f"Average weighted kappa: {avg_k:.3f} | "
        f"Average binary accuracy: {(results['pii_accuracy'] + results['lang_accuracy']) / 2:.3f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Judge-human calibration agreement")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to calibration JSONL")
    parser.add_argument("--max-pairs", type=int, default=None, help="Evaluate only first N pairs")
    parser.add_argument("--model", default=None, help="Override LOCAL_MODEL_NAME (e.g. llama3)")
    args = parser.parse_args(argv)

    try:
        entries = load_calibration(Path(args.input), max_pairs=args.max_pairs)
    except (FileNotFoundError, ValueError) as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(entries)} calibration pairs from {args.input}")

    settings = Settings()
    if args.model:
        settings.LOCAL_MODEL_NAME = args.model
    judge = JudgeEvaluator(llm=LLMClient(settings=settings))

    scored: list[tuple[int, EvaluationVerdict | None]] = [
        score_pair(judge, entry, i, len(entries)) for i, entry in enumerate(entries)
    ]

    valid = [(i, v) for i, v in scored if v is not None]
    if not valid:
        print("No judgments succeeded. Check the LLM endpoint/model and retry.", file=sys.stderr)
        return 1

    failed = len(entries) - len(valid)
    if failed:
        print(f"WARNING: {failed} of {len(entries)} pairs failed judging and were skipped.")

    aligned_entries = [entries[i] for i, _ in valid]
    aligned_verdicts = [v for _, v in valid]

    results = compute_agreement(aligned_entries, aligned_verdicts)
    print_report(results)

    report_path = Path(args.input).with_suffix(".calibration_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"n={results['n']}\n")
        f.write(f"brand_voice_kappa={results['brand_voice_kappa']:.4f}\n")
        f.write(f"policy_kappa={results['policy_kappa']:.4f}\n")
        f.write(f"pii_accuracy={results['pii_accuracy']:.4f}\n")
        f.write(f"lang_accuracy={results['lang_accuracy']:.4f}\n")
    print(f"Calibration report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
