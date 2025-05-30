from abc import ABC, abstractmethod
import time
from typing import Dict, Any
import openai
from openai import AsyncOpenAI
import asyncio

class BaseLLMClient(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_client()

    @abstractmethod
    def setup_client(self):
        pass

    @abstractmethod
    async def generate(self, prompt: str) -> Dict[str, Any]:
        pass

class OpenAIClient(BaseLLMClient):
    def setup_client(self):
        self.client = AsyncOpenAI(api_key=self.config["api_key"])

    async def generate(self, prompt: str) -> Dict[str, Any]:
        start_time = time.time()
        
        response = await self.client.chat.completions.create(
            model=self.config["model_name"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config["max_tokens"]
        )
        
        end_time = time.time()
        
        return {
            "text": response.choices[0].message.content,
            "response_time": end_time - start_time,
            "token_count": response.usage.total_tokens,
            "model": self.config["model_name"]
        }


def get_llm_client(model_config: Dict[str, Any]) -> BaseLLMClient:
    clients = {
        "openai": OpenAIClient,
    }
    
    client_class = clients.get(model_config["provider"])
    if not client_class:
        raise ValueError(f"Unsupported provider: {model_config['provider']}")
    
    return client_class(model_config) 