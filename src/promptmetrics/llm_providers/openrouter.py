import os
import logging
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from typing import Type

load_dotenv()
logger = logging.getLogger("promptmetrics.llm_providers.openrouter")

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

    async def generate_structured(self, prompt: str, response_model: Type[BaseModel]) -> dict:
        log_payload = {
            "model": self.model_name,
            "prompt": prompt,
            "response_model": response_model.__name__
        }
        logger.info(json.dumps({"event": "generate_structured_request", "payload": log_payload}))
        
        raw_content = "Response object not created"
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
                extra_headers=self.headers,
                max_tokens=1024,
            )
            raw_content = response.choices[0].message.content
            parsed_response = response_model.model_validate_json(raw_content)
            response_dict = parsed_response.model_dump()
            logger.info(json.dumps({"event": "generate_structured_success", "response": response_dict}))
            return response_dict
        except (ValidationError, json.JSONDecodeError) as e:
            error_details = {
                "event": "generate_structured_error",
                "error": str(e),
                "raw_model_response": raw_content,
            }
            logger.error(json.dumps(error_details))
            print(f"\n--- Judge API Error (generate_structured) for model {self.model_name}: {e} ---")
            return {"is_correct": None, "reasoning": f"Judge API call failed: {e}", "extracted_answer": None}
        except Exception as e:
            error_details = {
                "event": "generate_structured_error",
                "error": str(e),
                "raw_model_response": "N/A - API call failed before response",
            }
            logger.error(json.dumps(error_details))
            print(f"\n--- Judge API Error (generate_structured) for model {self.model_name}: {e} ---")
            return {"is_correct": None, "reasoning": f"Judge API call failed: {e}", "extracted_answer": None}