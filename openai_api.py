
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

    input_items = [{"role": "system", "content": system_prompt}] + messages

    stream = client.responses.create(
        model=model_name,
        input=input_items,
        stream=True
    )
    streamed_response_list: List[str] = []

    for event in stream:
        etype = getattr(event, "type", None)

        if etype is None and isinstance(event, dict):
            etype = event.get("type")

        if etype == "response.output_text.delta":  # run streaming
            delta = getattr(event, "delta", None)
            if delta is None and isinstance(event, dict):
                delta = event.get("delta")
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
                streamed_response_list.append(delta)

        if etype == "error":  # error
            sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(streamed_response_list)
