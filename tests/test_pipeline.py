from typing import Any

from src.llm_adapter import LLMClient
from src.pipeline import AgentDecision, AmazonSupportAgent
from src.vector_store import KnowledgeBase


class FakeKB:
    """Stub KnowledgeBase returning canned retrieval hits."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def count(self) -> int:
        return 1

    def search_similar(self, query: str, top_k: int = 2) -> list[dict[str, Any]]:
        self.calls.append(query)
        return [
            {
                "id": "1",
                "document": "My package is delayed, where is it?",
                "distance": 0.1,
                "metadata": {
                    "agent_text": "Sorry for the delay, we will track it right away.",
                    "customer_id": "9",
                    "tweet_id": "1",
                },
            }
        ]

    def ingest(self, *args: Any, **kwargs: Any) -> int:  # pragma: no cover
        return 0


class FakeLLM(LLMClient):
    """Mock LLMClient that fabricates a structured AgentDecision."""

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def generate_structured(
        self,
        messages: list[dict[str, str]],
        response_model,
        temperature: float = 0.1,
    ) -> Any:
        self.messages = messages
        assert temperature == 0.1
        return AgentDecision(
            detected_language="en",
            intent="shipping_delivery",
            decision="AUTO_REPLY",
            escalation_reason="NONE",
            draft_reply=(
                "I understand your package is delayed. Please send a Direct Message "
                "with your order number and I will track it for you."
            ),
        )


def make_agent(kb: KnowledgeBase, llm: FakeLLM) -> AmazonSupportAgent:
    return AmazonSupportAgent(kb=kb, llm=llm, top_k=2)  # type: ignore[arg-type]


class TestEscalateOnPII:
    def test_order_id_escalates_without_llm(self) -> None:
        kb = FakeKB()
        llm = FakeLLM()
        agent = make_agent(kb, llm)  # type: ignore[arg-type]

        decision = agent.process_ticket("My order 123-4567890-1234567 was charged twice")

        assert decision.decision == "ESCALATE"
        assert decision.intent == "account_access_security"
        assert "Order ID" in decision.escalation_reason
        assert decision.draft_reply
        # deterministic path must not call retrieval or LLM
        assert kb.calls == []
        assert llm.messages == []

    def test_legal_keyword_escalates_legal_intent(self) -> None:
        agent = make_agent(FakeKB(), FakeLLM())  # type: ignore[arg-type]
        decision = agent.process_ticket("I will sue Amazon and call my lawyer")
        assert decision.decision == "ESCALATE"
        assert decision.intent == "legal_regulatory_churn"


class TestAutoReply:
    def test_normal_ticket_calls_llm_and_returns_structured(self) -> None:
        kb = FakeKB()
        llm = FakeLLM()
        agent = make_agent(kb, llm)  # type: ignore[arg-type]

        decision = agent.process_ticket("My package has not arrived, please help")

        assert decision.decision == "AUTO_REPLY"
        assert decision.intent == "shipping_delivery"
        assert decision.escalation_reason == "NONE"
        assert kb.calls == ["My package has not arrived, please help"]
        assert len(llm.messages) == 2
        assert llm.messages[0]["role"] == "system"
        assert llm.messages[1]["role"] == "user"
        assert "Incoming customer tweet" in llm.messages[1]["content"]

    def test_reply_has_signature_and_length(self) -> None:
        agent = make_agent(FakeKB(), FakeLLM())  # type: ignore[arg-type]
        decision = agent.process_ticket("Where is my parcel?")

        assert len(decision.draft_reply) <= 240
        assert decision.draft_reply.rstrip().endswith(("^TN", "^SM", "^AB"))

    def test_reply_truncated_to_240_chars(self) -> None:
        llm = FakeLLM()
        agent = make_agent(FakeKB(), llm)  # type: ignore[arg-type]
        llm.generate_structured = lambda messages, response_model, temperature=0.1: (  # type: ignore[assignment]
            AgentDecision(
                detected_language="en",
                intent="general_inquiry",
                decision="AUTO_REPLY",
                escalation_reason="NONE",
                draft_reply="word " * 100,
            )
        )
        decision = agent.process_ticket("some text")
        assert len(decision.draft_reply) <= 240

    def test_other_language_defaults_to_english_signature(self) -> None:
        llm = FakeLLM()
        agent = make_agent(FakeKB(), llm)  # type: ignore[arg-type]
        llm.generate_structured = lambda messages, response_model, temperature=0.1: (  # type: ignore[assignment]
            AgentDecision(
                detected_language="other",
                intent="general_inquiry",
                decision="AUTO_REPLY",
                escalation_reason="NONE",
                draft_reply="Hello",
            )
        )
        decision = agent.process_ticket("some text")
        assert decision.detected_language == "en"
        assert decision.draft_reply.endswith(("^TN", "^SM", "^AB"))
