# -*- coding: utf-8 -*-
"""
@Desc    : MetaGPT LLM provider.
"""
from openai.types import CompletionUsage

from lamas.configs.llm_config import LLMType
from lamas.provider import OpenAILLM
from lamas.provider.llm_provider_registry import register_provider


@register_provider(LLMType.METAGPT)
class MetaGPTLLM(OpenAILLM):
    def _calc_usage(self, messages: list[dict], rsp: str) -> CompletionUsage:
        # The current billing is based on usage frequency. If there is a future billing logic based on the
        # number of tokens, please refine the logic here accordingly.
        return CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
