"""Automated evaluation metrics for the Amazon support agent pipeline.

Provides intent classification metrics (precision/recall/F1), escalation
false-negative rate with confusion matrix, and a PII-leakage check that
reuses the deterministic guardrail patterns.
"""

from typing import Any

from sklearn.metrics import classification_report, confusion_matrix

from src.guardrails import PATTERNS


def compute_intent_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    """
    Compute per-intent precision, recall, F1, support and macro averages.

    Args:
        y_true: Gold intent labels.
        y_pred: Predicted intent labels.

    Returns:
        {
            "per_class": {intent: {precision, recall, f1, support}},
            "macro_avg": {precision, recall, f1},
            "classification_report": dict/str,
        }
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")

    labels = sorted(set(y_true) | set(y_pred))

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    per_class: dict[str, dict[str, float]] = {}
    macro = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for label in labels:
        per_class[label] = {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        macro["precision"] += per_class[label]["precision"]
        macro["recall"] += per_class[label]["recall"]
        macro["f1"] += per_class[label]["f1"]

    n = max(len(labels), 1)
    macro = {k: v / n for k, v in macro.items()}

    return {
        "per_class": per_class,
        "macro_avg": macro,
        "classification_report_text": classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        ),
    }


def compute_escalation_fnr(
    y_true_decision: list[str], y_pred_decision: list[str]
) -> dict[str, Any]:
    """
    Compute escalation false-negative rate and confusion matrix.

    Positive class = "ESCALATE". FNR = FN / (FN + TP), i.e. how many
    tickets that should have been escalated were marked AUTO_REPLY.

    Args:
        y_true_decision: Gold decisions ("ESCALATE" or "AUTO_REPLY").
        y_pred_decision: Predicted decisions.

    Returns:
        {"fnr": float, "false_negatives": int, "true_positives": int,
         "confusion_matrix": np.ndarray, "tn": int, "fp": int, "fn": int, "tp": int}
    """
    if len(y_true_decision) != len(y_pred_decision):
        raise ValueError("y_true_decision and y_pred_decision must have equal length")

    y_true = ["ESCALATE" if d == "ESCALATE" else "AUTO_REPLY" for d in y_true_decision]
    y_pred = ["ESCALATE" if d == "ESCALATE" else "AUTO_REPLY" for d in y_pred_decision]

    labels = ["AUTO_REPLY", "ESCALATE"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "fnr": float(fnr),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "total_true_escalations": int(fn + tp),
        "confusion_matrix": cm,
        "confusion_matrix_labels": labels,
    }


def check_pii_leakage(response_text: str) -> bool:
    """
    Detect PII (order ID, email, phone, credit card) in a model response.

    Reuses the deterministic guardrail patterns; keyword triggers are not
    considered PII leakage.

    Args:
        response_text: The agent's draft reply text.

    Returns:
        True if any PII pattern is found in the response.
    """
    if not response_text:
        return False
    return any(pattern.search(response_text) for pattern, _ in PATTERNS)
