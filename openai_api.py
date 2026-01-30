
from openai import OpenAI
from typing import List, Dict
import sys

from logging_utils import add_log


# OpenAI Streaming Response

def openai_stream_answer(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
) -> str:

    """
        Arguments:
            - client     (OpenAI Client) : OpenAI Client
            - model_name (str)           : OpenAI Model Name (example: gpt-4o-mini)
            - messages   (dict(str))     : message for client,
                                           {"role": "user", "content": (user prompt with RAG result)}

        Returns:
            - (str) : streamed response message
    """

    for message in messages:
        print(message)

    stream = client.responses.create(
        model=model_name,
        input=messages,
        stream=True
    )
    streamed_response_list: List[str] = []

    for event in stream:
        etype = getattr(event, "type", None)
        add_log(tag='info', case_id=22, content=f'OpenAI streaming event type: {etype}')

        if etype is None and isinstance(event, dict):
            etype = event.get("type")
            add_log(tag='info', case_id=23, content=f'OpenAI streaming event type: {etype}')

        if etype == "response.output_text.delta":  # run streaming
            delta = getattr(event, "delta", None)
            add_log(tag='info', case_id=24, content=f'OpenAI streaming event type: {etype}, delta: {delta}')

            if delta is None and isinstance(event, dict):
                delta = event.get("delta")
                add_log(tag='info', case_id=25, content=f'OpenAI streaming delta: {delta}')
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
                streamed_response_list.append(delta)
                add_log(tag='info', case_id=26, content=f'OpenAI streaming delta: {delta}')

        if etype == "error":  # error
            sys.stdout.flush()
            add_log(tag='error', case_id=27, content='OpenAI streaming error')

    sys.stdout.write("\n")
    sys.stdout.flush()

    add_log(tag='info', case_id=28, content='OpenAI streaming finished')
    return "".join(streamed_response_list)
