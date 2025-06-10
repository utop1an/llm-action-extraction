from abc import ABC, abstractmethod
import time
from typing import Dict, Any
from openai import OpenAI, AsyncOpenAI
from ollama import Client, AsyncClient
import re

def split_think_content(text):
    match = re.search(r"<think>(.*?)</think>\s*(.*)", text, re.DOTALL)
    if match:
        think_part = match.group(1).strip()
        content_part = match.group(2).strip()
        return think_part, content_part
    else:
        return None, text.strip()

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


class OllamaClient(BaseLLMClient):
    def setup_client(self):
        self.client = Client(host='http://localhost:11434')
        self.async_client = AsyncClient(host='http://localhost:11434')

    async def generate_async(self, prompt: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            response = await self.async_client.chat(
                model=self.config["model_name"],
                messages=[{
                    "role": "user", 
                    "content": prompt
                }],
                options= {
                    "temperature": self.config.get("temperature", 0.7)
                }
            )

            think_part, content_part = split_think_content(response.message.content)
            return {
                "content": content_part,
                "think": think_part,
                "response_time": end_time - start_time,
                "model": self.config["model_name"]
            }
        except Exception as e:
            raise RuntimeError(f"Ollama API call failed: {str(e)}")

    def generate(self, prompt: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            response = self.client.chat(
                model=self.config["model_name"],
                messages=[{
                    "role": "user", 
                    "content": prompt
                }],
                options= {
                    "temperature": self.config.get("temperature", 0.7)
                }
            )

            end_time = time.time()
            think_part, content_part = split_think_content(response.message.content)
            
            return {
                "content": content_part,
                "think": think_part,
                "response_time": end_time - start_time,
                "model": self.config["model_name"]
            }
        except Exception as e:
            raise RuntimeError(f"Ollama API call failed: {str(e)}")

class OpenAIClient(BaseLLMClient):
    def setup_client(self):
        if "api_key" not in self.config:
            raise ValueError("OpenAI API key is required")
        self.client = OpenAI(api_key=self.config["api_key"])
        self.async_client = AsyncOpenAI(api_key=self.config["api_key"])

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


def get_llm_client(model_config: Dict[str, Any]) -> BaseLLMClient:
    if "provider" not in model_config:
        raise ValueError("Provider must be specified in model_config")
    if "model_name" not in model_config:
        raise ValueError("model_name must be specified in model_config")
    
    if model_config["provider"] == "openai":
        return OpenAIClient(model_config)
    elif model_config["provider"] == "ollama":
        return OllamaClient(model_config)
    else:
        raise ValueError(f"Unsupported provider: {model_config['provider']}")  



if __name__ == "__main__":
    client = get_llm_client({"provider": "ollama", "model_name": "deepseek-r1:8b"})
    response = client.generate("What is learning action models?")
    print(response)