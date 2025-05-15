import copy
from typing import Dict, Any, Optional

from openai import AsyncOpenAI


class LLMProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        provider_name: str,
        model_name: str,
        system_prompt: str = "",
        rag_prompt:str=""
    ):
        self.provider_name = provider_name
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.rag_prompt = rag_prompt
        self.api = AsyncOpenAI(base_url=base_url, api_key=api_key)
