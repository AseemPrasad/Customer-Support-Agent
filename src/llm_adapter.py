import json
import re
from typing import Any, TypeVar

from openai import OpenAI

from src.config import Settings
from src.config import settings as default_settings

T = TypeVar("T")


class LLMClient:
    """Unified OpenAI-compatible client for local (ollama) and remote (OpenAI) models."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: OpenAI | None = None,
    ) -> None:
        """
        Create a client from settings.

        Args:
            settings: Settings instance; defaults to the global settings.
            client: Pre-built OpenAI client (for testing/mocking).
        """
        self.settings = settings or default_settings

        if self.settings.LLM_PROVIDER == "local":
            self.base_url = self.settings.LOCAL_BASE_URL
            self.api_key = "ollama"
            self.model = self.settings.LOCAL_MODEL_NAME
        elif self.settings.LLM_PROVIDER == "remote":
            self.base_url = self.settings.REMOTE_BASE_URL
            self.api_key = self.settings.REMOTE_API_KEY
            self.model = self.settings.REMOTE_MODEL_NAME
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.settings.LLM_PROVIDER}")

        if client is not None:
            self.client = client
        else:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def generate_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        temperature: float = 0.1,
    ) -> T:
        """
        Request a structured response matching response_model.

        Tries client.beta.chat.completions.parse when available; falls back
        to chat.completions.create with JSON mode plus manual parsing.

        Args:
            messages: OpenAI-style message list [{"role": ..., "content": ...}].
            response_model: Pydantic model to validate/return.
            temperature: Sampling temperature.

        Returns:
            An instance of response_model.

        Raises:
            ValueError if the response cannot be parsed into response_model.
        """
        schema_json = json.dumps(response_model.model_json_schema())

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_model,
                temperature=temperature,
            )
            content = completion.choices[0].message.content
            if content and completion.choices[0].message.parsed is not None:
                return completion.choices[0].message.parsed
            if content:
                return self._parse_into_model(content, response_model)
            if completion.choices[0].message.refusal:
                raise ValueError(f"Model refused: {completion.choices[0].message.refusal}")
            raise ValueError("Empty completion content")
        except (AttributeError, TypeError) as exc:
            # beta.parse unavailable (older server/client) -> JSON mode fallback
            if isinstance(exc, ValueError):
                raise
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            content = completion.choices[0].message.content or ""
            return self._parse_into_model(content, response_model, schema_json=schema_json)

    def _parse_into_model(
        self,
        content: str,
        response_model: type[T],
        schema_json: str | None = None,
    ) -> T:
        """
        Parse raw model text into response_model, attempting several recovery paths.

        Args:
            content: Raw text from the model.
            response_model: Target pydantic model.
            schema_json: Optional schema string for error context.

        Returns:
            An instance of response_model.

        Raises:
            ValueError if all parsing attempts fail.
        """
        attempts: list[Any] = []

        attempts.append(content)

        fenced = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        if fenced:
            attempts.append(fenced.group(1).strip())

        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end > start:
            attempts.append(content[start : end + 1])

        for attempt in attempts:
            try:
                data = json.loads(attempt)
                return response_model.model_validate(data)
            except (json.JSONDecodeError, ValueError):
                continue

        hint = f"Expected schema: {schema_json}" if schema_json else ""
        msg = f"Could not parse model output into {response_model.__name__}. {hint}"
        raise ValueError(f"{msg}\nRaw: {content[:500]}")
