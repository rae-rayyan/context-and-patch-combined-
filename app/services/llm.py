from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


class PatchLLM(Protocol):
    def generate_patch(self, prompt: str) -> dict[str, Any]:
        """Return a JSON-compatible patch plan."""


@dataclass(slots=True)
class OpenAIPatchLLM:
    """Small adapter around the OpenAI Responses API.

    The Patch Agent itself stays provider-agnostic. This adapter is optional
    and imported lazily so the rest of the project does not require openai.
    """

    model: str | None = None
    api_key: str | None = None
    _client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.model = self.model or os.getenv("OPENAI_MODEL", "gpt-5.3-codex")
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The OpenAI adapter requires the 'openai' package."
            ) from exc

        self._client = OpenAI(api_key=self.api_key)

    def generate_patch(self, prompt: str) -> dict[str, Any]:
        response = self._client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior software repair engineer. "
                        "Return ONLY valid JSON matching the requested schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        text = getattr(response, "output_text", "")
        return _parse_json_object(text)


def _parse_json_object(text: str) -> dict[str, Any]:
    """Parse JSON even when a model wraps it in a code fence."""
    candidate = text.strip()

    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("LLM response did not contain a JSON object.")

    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse LLM patch JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("LLM patch response must be a JSON object.")
    return parsed
