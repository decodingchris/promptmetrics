import os
import logging
import json
import time
from pathlib import Path
import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Type, TypeVar

load_dotenv()
logger = logging.getLogger("promptmetrics.llm_providers.openrouter")

T = TypeVar("T", bound=BaseModel)


def get_model_details():
    """Fetches model details from OpenRouter, with local caching."""
    cache_dir = Path.home() / ".cache" / "promptmetrics"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "openrouter_models.json"

    if cache_file.exists():
        try:
            cache_data = json.loads(cache_file.read_text())
            if time.time() - cache_data.get("timestamp", 0) < 86400:
                return cache_data.get("models", {})
        except json.JSONDecodeError:
            logger.warning("Could not decode model cache. Refetching.")

    try:
        response = httpx.get("https://openrouter.ai/api/v1/models")
        response.raise_for_status()
        models_data = response.json().get("data", [])
        models_map = {model["id"]: model for model in models_data}

        cache_payload = {"timestamp": time.time(), "models": models_map}
        cache_file.write_text(json.dumps(cache_payload, indent=2))
        return models_map
    except (httpx.RequestError, json.JSONDecodeError) as e:
        logger.error(f"Failed to fetch/cache model details from OpenRouter: {e}")
        return {}


class OpenRouterLLM:
    """An asynchronous client for all OpenRouter API interactions."""

    MODELS_CACHE = get_model_details()

    def __init__(self, model_name: str):
        self.model_name = model_name
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment or .env file."
            )

        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=300.0,
            max_retries=3,
        )
        self.headers = {
            "HTTP-Referer": os.getenv(
                "YOUR_SITE_URL", "https://github.com/decodingchris/promptmetrics"
            ),
            "X-Title": os.getenv("YOUR_APP_NAME", "PromptMetrics"),
        }

        model_info = self.MODELS_CACHE.get(model_name)
        self.supports_vision = False
        self.supports_reasoning = False
        if model_info:
            input_modalities = model_info.get("architecture", {}).get(
                "input_modalities", []
            )
            if "image" in input_modalities:
                self.supports_vision = True

            supported_params = model_info.get("supported_parameters", [])
            if "reasoning" in supported_params:
                self.supports_reasoning = True

    async def generate(
        self, messages: list, temperature: float = 0.0, max_tokens: int = 8192
    ) -> dict:
        log_payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        logger.info(
            json.dumps(
                {"event": "generate_request", "payload": log_payload}, default=str
            )
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_headers=self.headers,
            )
            message = response.choices[0].message
            response_content = message.content.strip() if message.content else ""
            reasoning = getattr(message, "reasoning", None)

            log_response = {"content": response_content, "reasoning": reasoning}
            logger.info(
                json.dumps(
                    {"event": "generate_response_success", "response": log_response}
                )
            )

            return {"content": response_content, "reasoning": reasoning}
        except Exception as e:
            logger.error(
                json.dumps({"event": "generate_response_error", "error": str(e)})
            )
            print(f"\n--- API Error (generate) for model {self.model_name}: {e} ---")
            return {"content": f"API_ERROR: {e}", "reasoning": None}

    async def generate_structured(
        self, prompt: str, response_model: Type[T], max_tokens: int = 4096
    ) -> T:
        log_payload = {
            "model": self.model_name,
            "prompt": prompt,
            "response_model": response_model.__name__,
            "max_tokens": max_tokens,
        }
        logger.info(
            json.dumps({"event": "generate_structured_request", "payload": log_payload})
        )

        try:
            completion = await self.client.chat.completions.parse(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format=response_model,
                temperature=0.0,
                extra_headers=self.headers,
                max_tokens=max_tokens,
            )

            message = completion.choices[0].message

            if message.refusal:
                reason = f"Model refused to respond: {message.refusal}"
                logger.warning(
                    json.dumps(
                        {"event": "generate_structured_refusal", "reason": reason}
                    )
                )
                return response_model(reasoning=reason)

            parsed_response = message.parsed
            if not parsed_response:
                unparsed_content = message.content or ""
                reason = (
                    f"Failed to parse model response. This usually means the evaluator model ('{self.model_name}') "
                    "does not support structured outputs or returned malformed JSON. "
                    f"Raw response: '{unparsed_content[:100]}...'"
                )
                logger.error(
                    json.dumps(
                        {
                            "event": "generate_structured_parsing_failed",
                            "reason": reason,
                        }
                    )
                )
                return response_model(reasoning=reason)

            logger.info(
                json.dumps(
                    {
                        "event": "generate_structured_success",
                        "response": parsed_response.model_dump(),
                    }
                )
            )
            return parsed_response

        except Exception as e:
            error_details = {
                "event": "generate_structured_error",
                "error": str(e),
                "error_type": type(e).__name__,
            }
            logger.error(json.dumps(error_details))

            helpful_reason = (
                f"Evaluator API call failed with a '{type(e).__name__}'. This can happen if the evaluator model "
                f"('{self.model_name}') does not support structured outputs. Try a different --evaluator_model, "
                f"like 'mistralai/mistral-small-3.2-24b-instruct'. Original error: {e}"
            )
            print(
                f"\n--- Evaluator API Error (generate_structured) for model {self.model_name}: {helpful_reason} ---"
            )
            return response_model(reasoning=helpful_reason)