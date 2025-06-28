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
            print("Ollama library not found. Please install it using 'pip install ollama'.")
            
        super().__init__(config)
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
            end_time = time.time()
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
