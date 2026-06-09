import time
from typing import Dict, Any
from .base import BaseLLMClient
import re

def split_think_content(text):
    match = re.search(r"<think>(.*?)</think>\s*(.*)", text, re.DOTALL)
    if match:
        think_part = match.group(1).strip()
        content_part = match.group(2).strip()
        return think_part, content_part
    else:
        return None, text.strip()


class OllamaClient(BaseLLMClient):
    def __init__(self, config: Dict[str, Any]):
        try:
            from ollama import Client, AsyncClient
        except ImportError:
            raise ImportError("Ollama library not found. Please install it using 'pip install ollama'.")
            
        super().__init__(config)
        host = config.get("host", "http://localhost:11434")
        self.client = Client(host=host)
        self.async_client = AsyncClient(host=host)
        

    async def generate_async(self, prompt: str, temperature: float = 0) -> Dict[str, Any]:
        start_time = time.time()
        try:
            response = await self.async_client.chat(
                model=self.config["model_name"],
                messages=[{
                    "role": "user", 
                    "content": prompt
                }],
                options= {
                    "temperature": temperature
                }
            )

            think_part, content_part = split_think_content(response.message.content)
            end_time = time.time()
            return {
                "content": content_part,
                "think": think_part,
                "response_time": end_time - start_time,
                "model": self.config["model_name"],
                "usage": {
                    "prompt_eval_count": getattr(response, "prompt_eval_count", None),
                    "eval_count": getattr(response, "eval_count", None),
                },
            }
        except Exception as e:
            raise RuntimeError(f"Ollama API call failed: {str(e)}")

    def generate(self, prompt: str, temperature: float = 0) -> Dict[str, Any]:
        start_time = time.time()
        try:
            response = self.client.chat(
                model=self.config["model_name"],
                messages=[{
                    "role": "user", 
                    "content": prompt
                }],
                options= {
                    "temperature": temperature
                }
            )

            end_time = time.time()
            think_part, content_part = split_think_content(response.message.content)
            
            return {
                "content": content_part,
                "think": think_part,
                "response_time": end_time - start_time,
                "model": self.config["model_name"],
                "usage": {
                    "prompt_eval_count": getattr(response, "prompt_eval_count", None),
                    "eval_count": getattr(response, "eval_count", None),
                },
            }
        except Exception as e:
            raise RuntimeError(f"Ollama API call failed: {str(e)}")
