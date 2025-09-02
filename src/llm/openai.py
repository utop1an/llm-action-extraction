import time
from typing import Dict, Any
from .base import BaseLLMClient


class OpenAIClient(BaseLLMClient):

    def __init__(self, config: Dict[str, Any]):
        try:
            from openai import OpenAI, AsyncOpenAI
        except ImportError:
            print("OpenAI library not found. Please install it using 'pip install openai'.")
        
        if "api_key" not in config:
            raise ValueError("OpenAI API key is required")
        super().__init__(config)
        self.client = OpenAI(api_key=config.get("api_key"), base_url=config.get("base_url", "https://api.openai.com/v1"))
        self.async_client = AsyncOpenAI(api_key=self.config["api_key"], base_url=self.config.get("base_url", "https://api.openai.com/v1"))

        

    async def generate_async(self, prompt: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            response = await self.async_client.chat.completions.create(
                model=self.config["model_name"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.get("max_tokens", 2048),
                temperature=self.config.get("temperature", 0.7)
            )
            
            end_time = time.time()
            
            return {
                "content": response.choices[0].message.content,
                "response_time": end_time - start_time,
                "model": self.config["model_name"]
            }
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")

    def generate(self, prompt: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.config["model_name"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.get("max_tokens", 2048),
                temperature=self.config.get("temperature", 0.7)
            )
            
            end_time = time.time()
            
            return {
                "content": response.choices[0].message.content,
                "response_time": end_time - start_time,
                "model": self.config["model_name"]
            }
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")