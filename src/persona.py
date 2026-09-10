from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0

SUPPORTED_LANGS = ("en", "ja", "pt", "es", "fr", "hi")

SIGNATURE_MAP = {
    "en": ["^TN", "^SM", "^AB"],
    "ja": ["ET", "KA", "YS"],
    "pt": ["^RC", "^JM"],
    "es": ["^MM", "^JL"],
    "fr": ["^LB", "^CD"],
    "hi": ["^AK", "^SK"],
}

LANGUAGE_NAMES = {
    "en": "English",
    "ja": "Japanese",
    "pt": "Portuguese",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "other": "Other",
}

SYSTEM_PROMPT_TEMPLATE = """You are an AmazonHelp customer service agent responding to a customer on Twitter.

Brand voice guidelines:
- Be concise and empathetic. Acknowledge the customer's issue with warmth in 1-2 sentences.
- Be action-oriented: clearly state the next step the customer should take.
- Never make legal admissions (no "we admit", "it's our fault", "you are entitled to compensation").
- Redirect sensitive info safely: instruct the customer to send account/order details via Direct Message or the official secure link, never ask for them in a public reply.
- Do not fabricate order status, refund amounts, or policies you cannot verify.
- Only answer resolving next steps; if the issue is beyond scope, respond that you will escalate to the appropriate team.
- End every reply with an appropriate agent signature from your language: {signature}.
- Reply in the same language as the customer's message ({language_name}).

Historical example pairs are provided in context to guide tone; do not copy personal details from them.
"""


def detect_language(text: str) -> str:
    """
    Detect the language of text and map it to a supported language code.

    Returns one of en, ja, pt, es, fr, hi, or "other" if unsupported/undetectable.

    Args:
        text: The input text.

    Returns:
        Supported language code or "other".
    """
    if not text or not text.strip():
        return "other"
    try:
        detected = detect(text[:500])
    except Exception:  # noqa: BLE001 - langdetect may fail on short/garbled text
        detected = ""
    iso2 = detected.split("-")[0]
    if iso2 in ("en", "ja", "pt", "es", "fr", "hi"):
        return iso2
    return "other"


def get_signature(language: str) -> str:
    """
    Return the first signature for a language, or empty string if unsupported.

    Args:
        language: Language code (en, ja, pt, es, fr, hi, ...).

    Returns:
        A signature string or "".
    """
    sigs = SIGNATURE_MAP.get(language, [])
    return sigs[0] if sigs else ""


def translate_language_name(language: str) -> str:
    """
    Pretty-print a language code.

    Args:
        language: Language code.

    Returns:
        Human-readable language name.
    """
    return LANGUAGE_NAMES.get(language, language)
