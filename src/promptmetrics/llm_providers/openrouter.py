import os
import logging
import json
import time
from pathlib import Path
import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from typing import Type, TypeVar, Any, cast
import inspect

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

    def _build_json_schema_response_format(self, response_model: Type[T]) -> dict:
        """
        Builds a 'json_schema' response_format payload from a Pydantic v2 model.
        Compatible with OpenAI-style JSON schema constrained outputs that many
        OpenRouter-routed models honor.
        """
        try:
            schema = response_model.model_json_schema()
        except Exception:
            # Extremely defensive; for BaseModel subclasses this should exist.
            schema = {"type": "object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": schema,
                "strict": True,
            },
        }

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

            # Primary path: native Pydantic parsing support in SDK/model
            message = completion.choices[0].message

            if message.refusal:
                reason = f"Model refused to respond: {message.refusal}"
                logger.warning(
                    json.dumps(
                        {"event": "generate_structured_refusal", "reason": reason}
                    )
                )
                return response_model(reasoning=reason)

            parsed_response = getattr(message, "parsed", None)
            # Preserve current behavior for portability with tests:
            # If parsing returned None (not an exception), return the standard error.
            # We only attempt fallbacks when the parse call throws.
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
            # Fallback 1: JSON Schema constrained outputs via response_format=json_schema
            try:
                rf = self._build_json_schema_response_format(response_model)
                # Support both awaitable and direct returns (for tests/mocks)
                res_schema = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=cast(Any, [{"role": "user", "content": prompt}]),
                    response_format=cast(Any, rf),
                    temperature=0.0,
                    extra_headers=self.headers,
                    max_tokens=max_tokens,
                )
                if inspect.isawaitable(res_schema):
                    completion_schema = await cast(Any, res_schema)
                else:
                    completion_schema = res_schema  # type: ignore[assignment]

                msg = completion_schema.choices[0].message
                content = (msg.content or "").strip()
                try:
                    parsed = response_model.model_validate_json(content)
                    logger.info(
                        json.dumps(
                            {
                                "event": "generate_structured_fallback_json_schema_success",
                                "response": parsed.model_dump(),
                            }
                        )
                    )
                    return parsed
                except ValidationError:
                    try:
                        parsed = response_model.model_validate(json.loads(content))
                        logger.info(
                            json.dumps(
                                {
                                    "event": "generate_structured_fallback_json_schema_success_object",
                                    "response": parsed.model_dump(),
                                }
                            )
                        )
                        return parsed
                    except Exception:
                        pass
            except Exception as e2:
                logger.warning(
                    json.dumps(
                        {
                            "event": "generate_structured_fallback_json_schema_error",
                            "error": str(e2),
                        }
                    )
                )

            # Fallback 2: JSON mode with a short system hint
            try:
                rf_json_mode = {"type": "json_object"}
                # Keep hint short to minimize prompt drift.
                hint = "Return only a JSON object that matches the requested structure. Do not include any extra text."
                res_json = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=cast(
                        Any,
                        [
                            {"role": "system", "content": hint},
                            {"role": "user", "content": prompt},
                        ],
                    ),
                    response_format=cast(Any, rf_json_mode),
                    temperature=0.0,
                    extra_headers=self.headers,
                    max_tokens=max_tokens,
                )
                if inspect.isawaitable(res_json):
                    completion_json = await cast(Any, res_json)
                else:
                    completion_json = res_json  # type: ignore[assignment]
                msg = completion_json.choices[0].message
                content = (msg.content or "").strip()
                try:
                    parsed = response_model.model_validate_json(content)
                    logger.info(
                        json.dumps(
                            {
                                "event": "generate_structured_fallback_json_mode_success",
                                "response": parsed.model_dump(),
                            }
                        )
                    )
                    return parsed
                except ValidationError:
                    try:
                        parsed = response_model.model_validate(json.loads(content))
                        logger.info(
                            json.dumps(
                                {
                                    "event": "generate_structured_fallback_json_mode_success_object",
                                    "response": parsed.model_dump(),
                                }
                            )
                        )
                        return parsed
                    except Exception:
                        pass
            except Exception as e3:
                logger.warning(
                    json.dumps(
                        {
                            "event": "generate_structured_fallback_json_mode_error",
                            "error": str(e3),
                        }
                    )
                )

            # All strategies failed: preserve the helpful message you already return today.
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
