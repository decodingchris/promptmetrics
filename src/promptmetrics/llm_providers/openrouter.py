import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Type

load_dotenv()

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

    async def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 8192) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                extra_headers=self.headers,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"\n--- API Error (generate) for model {self.model_name}: {e} ---")
            return f"API_ERROR: {e}"

    async def generate_structured(self, prompt: str, response_model: Type[BaseModel]) -> dict:
        try:
            parsed_response = await self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format=response_model,
                temperature=0.0,
                extra_headers=self.headers,
                max_tokens=1024,
            )
            return parsed_response.choices[0].message.parsed.model_dump()
        except Exception as e:
            print(f"\n--- Judge API Error (generate_structured) for model {self.model_name}: {e} ---")
            return {"is_correct": None, "reasoning": f"Judge API call failed: {e}", "extracted_answer": None}