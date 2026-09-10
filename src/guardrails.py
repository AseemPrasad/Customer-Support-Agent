import re

SEVERE_KEYWORDS = {
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
    "hacked",
    "data breach",
    "legal action",
    "solicitor",
}

ORDER_ID_PATTERN = re.compile(r"\b\d{3}-\d{7}-\d{7}\b")
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_PATTERN = re.compile(r"\+?\d[\d\s-]{7,}\d")
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]?){12,15}\d\b")

PATTERNS = [
    (ORDER_ID_PATTERN, "Contains Amazon Order ID (PII protection rule)"),
    (EMAIL_PATTERN, "Contains email address (PII protection rule)"),
    (CREDIT_CARD_PATTERN, "Contains credit card number (PII protection rule)"),
    (PHONE_PATTERN, "Contains phone number (PII protection rule)"),
]

KEYWORD_DESC = {
    "lawyer": "Legal threat detected",
    "sue": "Legal threat detected",
    "lawsuit": "Legal threat detected",
    "attorney": "Legal threat detected",
    "police": "Law enforcement involvement detected",
    "chargeback": "Chargeback threat detected",
    "consumer court": "Legal threat detected",
    "fraud": "Fraud allegation detected",
    "unauthorized charge": "Unauthorized charge reported",
    "account hacked": "Account compromise reported",
    "hacked": "Account compromise reported",
    "data breach": "Security incident reported",
    "legal action": "Legal threat detected",
    "solicitor": "Legal threat detected",
}


def check_deterministic_escalation(text: str) -> tuple[bool, str]:
    """
    Detect PII and severe keywords that warrant immediate escalation.

    Args:
        text: The message to inspect (e.g. customer_text).

    Returns:
        (True, reason) if any PII pattern or severe keyword is matched,
        otherwise (False, "").
    """
    if not text:
        return False, ""

    for pattern, reason in PATTERNS:
        if pattern.search(text):
            return True, reason

    lower = text.lower()
    for keyword in SEVERE_KEYWORDS:
        if keyword in lower:
            return True, KEYWORD_DESC[keyword]

    return False, ""
