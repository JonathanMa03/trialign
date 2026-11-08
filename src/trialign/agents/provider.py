"""Structured provider adapter, adapted from Modern AutoCrit's JSON boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel


class ProviderUnavailable(RuntimeError):
    pass


def local_configuration(root: Path) -> dict:
    from dotenv import dotenv_values

    configured = os.environ.get("TRIALIGN_ENV_FILE")
    paths = [Path(configured)] if configured else [root / ".env"]
    values = {}
    origin = "environment"
    for path in paths:
        if path.is_file():
            values = {k: v for k, v in dotenv_values(path).items() if v}
            origin = "explicit environment file" if configured else "TriAlign .env"
            break
    values.update({k: v for k, v in os.environ.items() if k.startswith(("OPENAI_", "TRIALIGN_"))})
    return {
        "api_key": values.get("OPENAI_API_KEY", ""),
        "model": values.get("TRIALIGN_JUDGE_MODEL", values.get("OPENAI_MODEL", "gpt-4o-mini")),
        "critic_model": values.get("TRIALIGN_CRITIC_MODEL")
        or values.get("TRIALIGN_JUDGE_MODEL", values.get("OPENAI_MODEL", "gpt-4o-mini")),
        "base_url": values.get("OPENAI_BASE_URL"),
        "origin": origin,
    }


class OpenAIProvider:
    name = "openai"
    version = "responses-structured-1"

    def __init__(self, configuration: dict):
        from openai import OpenAI

        if not configuration.get("api_key"):
            raise ProviderUnavailable("No OpenAI credential is configured.")
        self.model = configuration["model"]
        self.critic_model = configuration.get("critic_model") or self.model
        self.client = OpenAI(
            api_key=configuration["api_key"],
            base_url=configuration.get("base_url"),
            timeout=35,
            max_retries=0,
        )
        self.calls = 0
        self.resolved_models = []
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    def generate(
        self,
        role: str,
        instructions: str,
        payload: dict,
        schema: type[BaseModel],
        *,
        timeout_seconds: float = 35,
    ) -> BaseModel:
        # Call and revision budgets are enforced by the orchestrator, not SDK retries.
        self.calls += 1
        response = self.client.responses.create(
            model=self.critic_model
            if role in {"criterion_critic_assessment", "criterion_critique"}
            else self.model,
            instructions=instructions,
            input=json.dumps(payload, ensure_ascii=False),
            text={
                "format": {
                    "type": "json_schema",
                    "name": role,
                    "schema": schema.model_json_schema(),
                    "strict": True,
                }
            },
            max_output_tokens=1800,
            store=False,
            timeout=max(1, timeout_seconds),
        )
        if response.usage:
            self.usage["input_tokens"] += response.usage.input_tokens
            self.usage["output_tokens"] += response.usage.output_tokens
        if response.model not in self.resolved_models:
            self.resolved_models.append(response.model)
        if response.status != "completed" or not response.output_text:
            raise ProviderUnavailable("Provider returned an incomplete or refused response.")
        return schema.model_validate_json(response.output_text)
