
from openai import OpenAI
from typing import List
import pandas as pd
import time

from utils import load_faq_as_dataframe, set_openai_key
from embedding import HFMeanPoolingEmbedder, EMBEDDING_MODEL_NAME
from chroma_db import build_or_load_chroma
from chatbot_utils import build_instructions, build_prompt_with_rag_result
from rag import retrieve_top_k
from openai_api import openai_stream_answer


# remove warning messages
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)


FAQ_PKL_PATH = 'final_result.pkl'
DISTANCE_THRESHOLD = 0.3
OPENAI_MODEL = 'gpt-4o-mini'
QUESTION_LIST_PATH = 'evaluation_question_list.txt'
TOP_K = 2


def run_evaluation(question_list:List[str]):
    set_openai_key()

    # load FAQ as Pandas DataFrame
    faq_df = load_faq_as_dataframe(FAQ_PKL_PATH)

    # load Chroma DB (with Q&A Embedded)
    embedder = HFMeanPoolingEmbedder(model_name=EMBEDDING_MODEL_NAME)
    collection = build_or_load_chroma(faq_df, embedder)

    # OpenAI Client and system prompt
    client = OpenAI()
    instructions = build_instructions()

    evaluation_result = {
        'elapsed_time': [],
        'retrieve_top_k_time': [],
        'openai_api_time': [],
        'test_query': [],
        'shortest_distance': [],  # shortest distance of RAG retrieval
        'assistant_answer': []
    }

    for test_query in question_list:
        start_at = time.time()

        # run RAG for only 'question' part
        print(f'test > {test_query}')
        rag_retrieved_faqs, shortest_distance = retrieve_top_k(collection, test_query, TOP_K)
        retrieve_top_k_time = time.time() - start_at

        # handle user questions not related to 'Naver SmartStore'
        openai_api_answer = True
        if shortest_distance > DISTANCE_THRESHOLD:
            print('chatbot > [REFUSAL]')
            openai_api_answer = False

        if openai_api_answer:
            # OpenAI final input prompt = (user question + RAG-retrieved FAQs)
            # without chat history for test
            prompt_with_rag_result = build_prompt_with_rag_result(test_query, rag_retrieved_faqs)
            final_content = prompt_with_rag_result + '\n' + instructions
            trimmed_message = [{"role": "user", "content": final_content}]

            print("chatbot > ")
            openai_api_start_at = time.time()
            assistant_text = openai_stream_answer(
                client=client,
                model_name=OPENAI_MODEL,
                messages=trimmed_message
            )
            openai_api_time = time.time() - openai_api_start_at
        else:
            assistant_text = '[REFUSAL]'
            openai_api_time = None

        # append to evaluation result
        elapsed_time = time.time() - start_at
        evaluation_result['elapsed_time'].append(elapsed_time)
        evaluation_result['retrieve_top_k_time'].append(retrieve_top_k_time)
        evaluation_result['openai_api_time'].append(openai_api_time)
        evaluation_result['test_query'].append(test_query)
        evaluation_result['shortest_distance'].append(shortest_distance)  # shortest distance of RAG retrieval
        evaluation_result['assistant_answer'].append(assistant_text)

        # save as Pandas DataFrame
        evaluation_result_df = pd.DataFrame(evaluation_result)
        evaluation_result_df.to_csv('evaluation_result_details.csv')


if __name__ == "__main__":
    with open(QUESTION_LIST_PATH, 'r', encoding="utf-8") as f:
        question_list = f.readlines()
        question_list = [question.replace('\n', '') for question in question_list]
        question_list = list(filter(lambda x: len(x) >= 1, question_list))
        f.close()

    assert len(question_list) == 100
    run_evaluation(question_list)
