import os
import logging
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Type, Union

load_dotenv()
logger = logging.getLogger("promptmetrics.llm_providers.openrouter")

T = Type[BaseModel]
StructuredResponse = Union[BaseModel, dict]

class OpenRouterLLM:
    """An asynchronous client for all OpenRouter API interactions."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment or .env file.")
        
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=300.0,
            max_retries=3,
        )
        self.headers = {
            "HTTP-Referer": os.getenv("YOUR_SITE_URL", "https://github.com/decodingchris/promptmetrics"),
            "X-Title": os.getenv("YOUR_APP_NAME", "PromptMetrics"),
        }

    async def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 8192) -> dict:
        log_payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        logger.info(json.dumps({"event": "generate_request", "payload": log_payload}))
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                extra_headers=self.headers,
            )
            message = response.choices[0].message
            response_content = message.content.strip() if message.content else ""
            reasoning = getattr(message, 'reasoning', None)

            log_response = {"content": response_content, "reasoning": reasoning}
            logger.info(json.dumps({"event": "generate_response_success", "response": log_response}))
            
            return {
                "content": response_content,
                "reasoning": reasoning
            }
        except Exception as e:
            logger.error(json.dumps({"event": "generate_response_error", "error": str(e)}))
            print(f"\n--- API Error (generate) for model {self.model_name}: {e} ---")
            return {"content": f"API_ERROR: {e}", "reasoning": None}

    async def generate_structured(self, prompt: str, response_model: T) -> StructuredResponse:
        log_payload = {
            "model": self.model_name,
            "prompt": prompt,
            "response_model": response_model.__name__
        }
        logger.info(json.dumps({"event": "generate_structured_request", "payload": log_payload}))
        
        try:
            completion = await self.client.chat.completions.parse(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format=response_model,
                temperature=0.0,
                extra_headers=self.headers,
                max_tokens=1024,
            )

            message = completion.choices[0].message

            if message.refusal:
                reason = f"Model refused to respond: {message.refusal}"
                logger.warning(json.dumps({"event": "generate_structured_refusal", "reason": reason}))
                return {"is_correct": None, "reasoning": reason, "extracted_answer": None}

            parsed_response = message.parsed
            if not parsed_response:
                unparsed_content = message.content or ""
                reason = (
                    f"Failed to parse model response. This usually means the judge model ('{self.model_name}') "
                    "does not support structured outputs or returned malformed JSON. "
                    f"Raw response: '{unparsed_content[:100]}...'"
                )
                logger.error(json.dumps({"event": "generate_structured_parsing_failed", "reason": reason}))
                return {"is_correct": None, "reasoning": reason, "extracted_answer": None}

            logger.info(json.dumps({"event": "generate_structured_success", "response": parsed_response.model_dump()}))
            return parsed_response
        
        except Exception as e:
            error_details = {
                "event": "generate_structured_error",
                "error": str(e),
                "error_type": type(e).__name__,
            }
            logger.error(json.dumps(error_details))

            helpful_reason = (
                f"Judge API call failed with a '{type(e).__name__}'. This can happen if the judge model "
                f"('{self.model_name}') does not support structured outputs. Try a different --judge_model, "
                f"like 'mistralai/mistral-small-3.2-24b-instruct'. Original error: {e}"
            )
            print(f"\n--- Judge API Error (generate_structured) for model {self.model_name}: {helpful_reason} ---")
            return {"is_correct": None, "reasoning": helpful_reason, "extracted_answer": None}