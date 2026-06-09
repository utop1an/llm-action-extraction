import time
from typing import Dict, Any
from .base import BaseLLMClient


class OpenAIClient(BaseLLMClient):

    def __init__(self, config: Dict[str, Any]):
        try:
            from openai import OpenAI, AsyncOpenAI
        except ImportError:
            raise ImportError("OpenAI library not found. Please install it using 'pip install openai'.")
        
        if not config.get("api_key"):
            raise ValueError("OpenAI API key is required")
        super().__init__(config)
        client_kwargs = {
            "api_key": config.get("api_key"),
            "base_url": config.get("base_url") or "https://api.openai.com/v1",
        }
        if config.get("timeout") is not None:
            client_kwargs["timeout"] = config["timeout"]
        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)

        

    async def generate_async(self, prompt: str, temperature: float = 0) -> Dict[str, Any]:
        start_time = time.time()
        try:
            params = self._build_chat_params(prompt, temperature)
            response = await self.async_client.chat.completions.create(**params)
            
            end_time = time.time()
            
            return {
                "content": response.choices[0].message.content,
                "response_time": end_time - start_time,
                "model": self.config["model_name"],
                "usage": response.usage.model_dump() if response.usage else None,
            }
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")

    def generate(self, prompt: str, temperature: float = 0) -> Dict[str, Any]:
        start_time = time.time()
        try:
            params = self._build_chat_params(prompt, temperature)
            response = self.client.chat.completions.create(**params)
            
            end_time = time.time()
            
            return {
                "content": response.choices[0].message.content,
                "response_time": end_time - start_time,
                "model": self.config["model_name"],
                "usage": response.usage.model_dump() if response.usage else None,
            }
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")

    def _build_chat_params(self, prompt: str, temperature: float) -> Dict[str, Any]:
        params = {
            "model": self.config["model_name"],
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.config.get("supports_custom_sampling", True):
            params.update(
                {
                    "temperature": temperature,
                    "top_p": 1,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                }
            )
        if self.config.get("max_tokens") is not None:
            params["max_tokens"] = self.config["max_tokens"]
        return params
