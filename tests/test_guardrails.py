import pytest

from src.guardrails import check_deterministic_escalation


class TestPIIOrderID:
    def test_detects_order_id(self) -> None:
        flag, reason = check_deterministic_escalation(
            "My order 123-4567890-1234567 has not arrived."
        )
        assert flag is True
        assert "Order ID" in reason

    def test_no_order_id(self) -> None:
        flag, reason = check_deterministic_escalation("Where is my package?")
        assert flag is False
        assert reason == ""


class TestPIIEmail:
    def test_detects_email(self) -> None:
        flag, reason = check_deterministic_escalation(
            "Please email me at john.doe+tag@example.co.uk"
        )
        assert flag is True
        assert "email" in reason.lower()


class TestPIIPhone:
    def test_detects_phone(self) -> None:
        flag, _ = check_deterministic_escalation("Call +1 202-555-0134")
        assert flag is True


class TestPIICreditCard:
    def test_detects_credit_card(self) -> None:
        flag, reason = check_deterministic_escalation("Card 4532 1234 5678 9012 was charged")
        assert flag is True
        assert "credit" in reason.lower()


class TestLegalKeywords:
    @pytest.mark.parametrize(
        "text",
        [
            "I will sue Amazon",
            "retaining a lawyer",
            "my attorney is ready",
            "we will file a lawsuit",
            "contact the police",
            "start a chargeback",
            "consumer court complaint",
            "this is fraud",
            "an unauthorized charge appeared",
            "my account was hacked",
            "a data breach happened",
            "I am taking legal action",
            "my solicitor will call",
        ],
    )
    def test_detects_legal_keywords(self, text: str) -> None:
        flag, reason = check_deterministic_escalation(text)
        assert flag is True
        assert reason


class TestCleanText:
    @pytest.mark.parametrize(
        "text",
        [
            "Where is my refund? It has been 5 days.",
            "When will my package ship?",
            "",
            None,
            "  ",
        ],
    )
    def test_no_escalation(self, text: str) -> None:
        flag, reason = check_deterministic_escalation(text)
        assert flag is False
        assert reason == ""
