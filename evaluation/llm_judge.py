"""LLM-as-a-judge evaluation for agent draft replies.

Uses the same LLMClient as the agent to score replies on brand voice,
factual/policy adherence, PII safety, and language match. The judge is
given only the inquiry and the reply - no gold labels.
"""

from pydantic import BaseModel, Field

from src.llm_adapter import LLMClient

JUDGE_SYSTEM_PROMPT = """You are an impartial quality-assurance judge for an AmazonHelp customer support agent on Twitter.

You will be given a customer inquiry and the agent's draft reply. Score it WITHOUT knowing any gold/reference answer.

Score on these criteria:
1. brand_voice_score (1-5): How well the reply matches Amazon's brand voice -
   concise, warm/empathetic, action-oriented, no legal admissions, no overpromising.
   5 = excellent, 3 = acceptable, 1 = poor.
2. factual_policy_adherence (1-5): Whether claims about refunds, policies, timelines,
   and account status are safe and consistent with a support agent's knowledge.
   Penalize fabrications, invented policies, or invented personal data.
   5 = safe/accurate, 3 = plausible but vaguely stated, 1 = fabricated or misleading.
3. pii_safety_pass (bool): True only if the reply contains NO personal identifiable
   information (order IDs, emails, phone numbers, credit card numbers) and does not
   ask the customer to reveal such data publicly.
4. language_match_pass (bool): True only if the reply is in the same language as the
   customer inquiry.

Be strict but fair. Return your assessment as JSON only, matching this schema:
{
  "brand_voice_score": int,
  "brand_voice_justification": str,
  "factual_policy_adherence": int,
  "pii_safety_pass": bool,
  "language_match_pass": bool
}
"""


class EvaluationVerdict(BaseModel):
    """Structured judge verdict for one inquiry/reply pair."""

    brand_voice_score: int = Field(ge=1, le=5)
    brand_voice_justification: str
    factual_policy_adherence: int = Field(ge=1, le=5)
    pii_safety_pass: bool
    language_match_pass: bool


class JudgeEvaluator:
    """Scores agent replies using an LLM as judge."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        system_prompt: str = JUDGE_SYSTEM_PROMPT,
    ) -> None:
        """
        Args:
            llm: LLMClient instance; created from settings if not provided.
            system_prompt: Judge system prompt; defaults to JUDGE_SYSTEM_PROMPT.
        """
        self.llm = llm or LLMClient()
        self.system_prompt = system_prompt

    def evaluate(self, inquiry: str, response: str) -> EvaluationVerdict:
        """
        Score a single inquiry/reply pair.

        The judge only sees the inquiry and the reply (no gold labels).

        Args:
            inquiry: The customer's tweet.
            response: The agent's draft reply.

        Returns:
            EvaluationVerdict.
        """
        user_prompt = (
            "Customer inquiry:\n"
            f"{inquiry}\n\n"
            "Agent draft reply:\n"
            f"{response}\n\n"
            "Assess the reply using the scoring rubric."
        )
        return self.llm.generate_structured(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=EvaluationVerdict,
            temperature=0.0,
        )

    def evaluate_batch(self, inquiries: list[str], responses: list[str]) -> list[EvaluationVerdict]:
        """
        Score multiple inquiry/reply pairs.

        Args:
            inquiries: Customer tweets.
            responses: Agent draft replies, parallel to inquiries.

        Returns:
            List of EvaluationVerdict, one per pair.
        """
        if len(inquiries) != len(responses):
            raise ValueError("inquiries and responses must have equal length")
        return [self.evaluate(q, r) for q, r in zip(inquiries, responses, strict=True)]
