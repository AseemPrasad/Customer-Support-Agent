import random
import re
from typing import Literal

from pydantic import BaseModel, Field

from src.guardrails import check_deterministic_escalation
from src.llm_adapter import LLMClient
from src.persona import (
    SIGNATURE_MAP,
    SYSTEM_PROMPT_TEMPLATE,
    detect_language,
    get_signature,
    translate_language_name,
)
from src.vector_store import KnowledgeBase

MAX_REPLY_CHARS = 240

Intent = Literal[
    "order_status",
    "refund_related",
    "shipping_delivery",
    "account_access_security",
    "payments_billing",
    "product_quality_complaint",
    "legal_regulatory_churn",
    "general_inquiry",
]

Language = Literal["en", "ja", "pt", "es", "fr", "hi", "other"]
Decision = Literal["AUTO_REPLY", "ESCALATE"]


class AgentDecision(BaseModel):
    """Structured output produced by the support agent for one ticket."""

    detected_language: Language
    intent: Intent
    decision: Decision = "AUTO_REPLY"
    escalation_reason: str = "NONE"
    draft_reply: str = Field(default="", max_length=500)
    rag_snippets: list[str] = Field(default_factory=list)

    def get_reason(self) -> str:
        return self.escalation_reason


EMERGENCY_CANNED_RESPONSES = {
    "en": "Thank you for reaching out. To keep your account and details safe, please send us a Direct Message or use the official Amazon Help page so we can assist you privately.",
    "ja": "ご連絡ありがとうございます。お客様の情報を安全に守るため、ダイレクトメッセージまたはAmazonヘルプページよりご連絡ください。",
    "pt": "Obrigado pelo contato. Para proteger suas informações, envie-nos uma Mensagem Direta ou use a página oficial de Ajuda da Amazon.",
    "es": "Gracias por escribirnos. Para mantener tus datos seguros, envíanos un Mensaje Directo o usa la página oficial de Ayuda de Amazon.",
    "fr": "Merci de nous avoir contactés. Pour protéger vos informations, envoyez-nous un message privé ou utilisez la page d'Aide officielle d'Amazon.",
    "hi": "संपर्क के लिए धन्यवाद। आपकी जानकारी को सुरक्षित रखने के लिए कृपया हमें डायरेक्ट मैसेज भेजें या Amazon हेल्प पेज का उपयोग करें।",
}

LEGAL_INTENTS = {
    "lawyer",
    "sue",
    "lawsuit",
    "attorney",
    "police",
    "legal action",
    "solicitor",
    "consumer court",
}


class AmazonSupportAgent:
    """Orchestrates guardrails, retrieval, and LLM generation for a support ticket."""

    def __init__(
        self,
        kb: KnowledgeBase,
        llm: LLMClient | None = None,
        top_k: int = 2,
    ) -> None:
        """
        Args:
            kb: Vector store containing historical resolved pairs.
            llm: LLMClient instance; created from settings if not provided.
            top_k: Number of similar historical pairs to retrieve.
        """
        self.kb = kb
        self.llm = llm or LLMClient()
        self.top_k = top_k

    def process_ticket(self, user_tweet: str) -> AgentDecision:
        """
        Process a single customer tweet into an AgentDecision.

        Deterministic guardrails (PII / severe keywords) short-circuit to
        ESCALATE with a safe canned reply. Otherwise a retriever-augmented
        prompt is sent to the LLM for structured decisioning.

        Args:
            user_tweet: The customer's incoming tweet text.

        Returns:
            AgentDecision with intent, decision, and draft reply.
        """
        flag, reason = check_deterministic_escalation(user_tweet)

        if flag:
            return self._build_escalation(user_tweet, reason)

        lang = detect_language(user_tweet)
        retrieved = self.kb.search_similar(user_tweet, top_k=self.top_k)

        context_parts = []
        for i, hit in enumerate(retrieved, start=1):
            meta = hit.get("metadata") or {}
            context_parts.append(
                f"Example {i}:\nCustomer: {hit.get('document', '')}\n"
                f"Agent resolved: {meta.get('agent_text', '')}"
            )
        context = (
            "\n\n".join(context_parts) if context_parts else "No historical examples available."
        )

        signature = get_signature(lang) or get_signature("en")
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            signature=signature,
            language_name=translate_language_name(lang),
        )
        system_prompt += (
            "\n\nOperational guidelines:\n"
            "- Respond only in JSON matching the AgentDecision schema below.\n"
            "- detected_language must be one of en, ja, pt, es, fr, hi, other.\n"
            "- intent must be one of: order_status, refund_related, shipping_delivery, "
            "account_access_security, payments_billing, product_quality_complaint, "
            "legal_regulatory_churn, general_inquiry.\n"
            "- decision must be AUTO_REPLY unless the issue clearly requires a human, "
            "then ESCALATE with a brief escalation_reason.\n"
            "- draft_reply must be under 240 characters, contain no PII, be in the "
            "customer's language, and end with the required signature."
        )

        user_prompt = (
            "Historical resolved examples:\n"
            f"{context}\n\n"
            "Incoming customer tweet:\n"
            f"{user_tweet}\n\n"
            "Return your decision as a single JSON object."
        )

        decision = self.llm.generate_structured(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=AgentDecision,
            temperature=0.1,
        )

        decision.draft_reply = self._enforce_reply_constraints(
            decision.draft_reply, decision.detected_language
        )
        decision.rag_snippets = [
            f"Customer: {hit.get('document', '')} ➔ Agent: {(hit.get('metadata') or {}).get('agent_text', '')}"
            for hit in retrieved
        ]
        return decision

    def _build_escalation(self, user_tweet: str, reason: str) -> AgentDecision:
        """Construct an ESCALATE decision from deterministic guardrail flag."""
        lang = detect_language(user_tweet)
        intent: Intent = (
            "legal_regulatory_churn"
            if self._is_legal_trigger(user_tweet)
            else "account_access_security"
        )

        canned = EMERGENCY_CANNED_RESPONSES.get(lang) or EMERGENCY_CANNED_RESPONSES["en"]
        canned = self._enforce_reply_constraints(canned, lang)
        return AgentDecision(
            detected_language=lang,
            intent=intent,
            decision="ESCALATE",
            escalation_reason=reason,
            draft_reply=canned,
        )

    @staticmethod
    def _is_legal_trigger(text: str) -> bool:
        lower = text.lower()
        return any(k in lower for k in LEGAL_INTENTS)

    @staticmethod
    def _enforce_reply_constraints(text: str, language: str) -> str:
        """Truncate to max chars and append a signature if missing."""
        text = (text or "").strip()
        text = re.sub(r"\s*\^[A-Z]{2,3}\s*$", "", text).strip()
        signature_source = language if language != "other" else "en"
        sigs = SIGNATURE_MAP.get(signature_source, SIGNATURE_MAP["en"])
        sig = random.choice(sigs)
        has_sig = bool(re.search(r"[\^A-Z]{2,3}\s*$", text))
        reserved = 0 if has_sig else len(sig) + 1  # leading space + signature
        marker_len = len(" ...")
        if len(text) > MAX_REPLY_CHARS:
            limit = MAX_REPLY_CHARS - reserved - marker_len
            text = text[: max(limit, 0)].rstrip() + " ..."
        text = text.strip()
        if not re.search(r"[\^A-Z]{2,3}\s*$", text):
            text = f"{text} {sig}"
        return text.strip()
