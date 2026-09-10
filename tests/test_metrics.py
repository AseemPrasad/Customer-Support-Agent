from evaluation.automated_metrics import (
    check_pii_leakage,
    compute_escalation_fnr,
    compute_intent_metrics,
)


class TestEscalationFNR:
    def test_fnr_basic(self) -> None:
        y_true = ["ESCALATE", "ESCALATE", "ESCALATE", "AUTO_REPLY", "AUTO_REPLY", "AUTO_REPLY"]
        y_pred = ["ESCALATE", "AUTO_REPLY", "AUTO_REPLY", "AUTO_REPLY", "AUTO_REPLY", "ESCALATE"]

        result = compute_escalation_fnr(y_true, y_pred)

        assert result["fnr"] == 2 / 3
        assert result["false_negatives"] == 2
        assert result["true_positives"] == 1
        assert result["false_positives"] == 1
        assert result["true_negatives"] == 2
        assert result["total_true_escalations"] == 3

    def test_fnr_zero_when_no_true_escalations(self) -> None:
        result = compute_escalation_fnr(["AUTO_REPLY", "AUTO_REPLY"], ["ESCALATE", "AUTO_REPLY"])
        assert result["fnr"] == 0.0
        assert result["total_true_escalations"] == 0

    def test_fnr_perfect(self) -> None:
        result = compute_escalation_fnr(["ESCALATE"], ["ESCALATE"])
        assert result["fnr"] == 0.0
        assert result["confusion_matrix"].tolist() == [[0, 0], [0, 1]]

    def test_length_mismatch_raises(self) -> None:
        try:
            compute_escalation_fnr(["ESCALATE"], ["ESCALATE", "AUTO_REPLY"])
        except ValueError:
            return
        raise AssertionError("expected ValueError for length mismatch")


class TestIntentMetrics:
    def test_macro_f1(self) -> None:
        y_true = ["a", "a", "b", "b", "c"]
        y_pred = ["a", "b", "b", "a", "c"]
        result = compute_intent_metrics(y_true, y_pred)

        assert set(result["per_class"]) == {"a", "b", "c"}
        macro = result["macro_avg"]
        assert 0.0 <= macro["f1"] <= 1.0
        assert 0.0 <= macro["precision"] <= 1.0
        assert 0.0 <= macro["recall"] <= 1.0

    def test_empty_labels_do_not_crash(self) -> None:
        result = compute_intent_metrics(["a"], ["b"])
        assert "a" in result["per_class"]
        assert "b" in result["per_class"]

    def test_length_mismatch_raises(self) -> None:
        try:
            compute_intent_metrics(["a"], ["a", "b"])
        except ValueError:
            return
        raise AssertionError("expected ValueError for length mismatch")


class TestPIILeakage:
    def test_detects_order_id(self) -> None:
        assert check_pii_leakage("Here is your order 123-4567890-1234567") is True

    def test_detects_email(self) -> None:
        assert check_pii_leakage("mail me at a@b.co") is True

    def test_clean_text(self) -> None:
        assert check_pii_leakage("We are happy to help you today") is False

    def test_empty(self) -> None:
        assert check_pii_leakage("") is False
