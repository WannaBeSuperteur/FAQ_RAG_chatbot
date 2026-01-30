
from openai import OpenAI
from typing import List, Dict
import sys


# OpenAI Streaming Response

def openai_stream_answer(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    messages: List[Dict[str, str]],
) -> str:

    """
        Arguments:
            - client        (OpenAI Client) : OpenAI Client
            - model_name    (str)           : OpenAI Model Name (example: gpt-4o-mini)
            - system_prompt (str)           : system prompt for OpenAI Client
            - messages      (dict(str))     : message for client,
                                              {"role": "user", "content": (user prompt with RAG result)}

        Returns:
            - (str) : streamed response message
    """

    raise NotImplementedError
