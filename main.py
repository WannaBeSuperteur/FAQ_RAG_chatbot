
import os
from openai import OpenAI

from logging_utils import add_log
from utils import load_faq_as_dataframe
from embedding import HFMeanPoolingEmbedder, EMBEDDING_MODEL_NAME
from chroma_db import build_or_load_chroma
from chatbot_utils import build_system_prompt, append_to_history, load_recent_history, build_prompt_with_rag_result
from rag import retrieve_top_k
from openai_api import openai_stream_answer


FAQ_PKL_PATH = 'final_result.pkl'
DISTANCE_THRESHOLD = 0.3
OPENAI_MODEL = 'gpt-4o-mini'
TOP_K = 5


def set_openai_key():
    with open('chatgpt_key.txt', 'r') as f:
        openai_key = f.readlines()[0]
        os.environ['OPENAI_API_KEY'] = openai_key
    add_log(tag='info', case_id=1, content=f'OpenAI Key set: {openai_key[:16]}****')


def main():
    add_log(tag='info', case_id=0, content='start')
    set_openai_key()

    # load FAQ as Pandas DataFrame
    faq_df = load_faq_as_dataframe(FAQ_PKL_PATH)

    # load Chroma DB (with Q&A Embedded)
    embedder = HFMeanPoolingEmbedder(model_name=EMBEDDING_MODEL_NAME)
    collection = build_or_load_chroma(faq_df, embedder)

    # OpenAI Client and system prompt
    client = OpenAI()
    add_log(tag='info', case_id=12, content='OpenAI client generated successful!')
    system_prompt = build_system_prompt()
    add_log(tag='info', case_id=13, content='System prompt build successful!')

    while True:
        user_query = input("input user question > ").strip()
        add_log(tag='info', case_id=14, content=f'user query: {user_query}')
        if not user_query:
            add_log(tag='warning', case_id=15, content='user query is empty')
            continue

        # save history
        append_to_history("user", user_query)

        # run RAG for only 'question' part
        rag_retrieved_faqs, shortest_distance = retrieve_top_k(collection, user_query, TOP_K)

        # handle user questions not related to 'Naver SmartStore'
        if shortest_distance > DISTANCE_THRESHOLD:
            refusal = (
                "죄송하지만, 해당 질문은 네이버 스마트스토어 FAQ 범위를 벗어납니다.\n"
                "스마트스토어 가입/운영/관리 등 네이버 스마트스토어와 관련된 질문으로 다시 말씀해 주실 수 있을까요?\n"
                "추가로 궁금한 점이 있으신가요?"
            )
            print(f"chatbot > {refusal}\n")
            add_log(tag='info', case_id=19, content=f'chatbot response (refusal): {refusal}, shortest_distance: {shortest_distance}')
            continue

        # OpenAI final input prompt = (user question + RAG-retrieved FAQs)
        recent_history = load_recent_history(max_messages=10)
        prompt_with_rag_result = build_prompt_with_rag_result(user_query, rag_retrieved_faqs)

        # recent_history include last user question (prompt) -> add RAG-result for last user question
        trimmed = []
        for m in recent_history[:-1]:
            trimmed.append(m)
        trimmed.append({"role": "user", "content": prompt_with_rag_result})

        print("chatbot > ")
        assistant_text = openai_stream_answer(
            client=client,
            model_name=OPENAI_MODEL,
            system_prompt=system_prompt,
            messages=trimmed
        )
        add_log(tag='info', case_id=32, content=f'chatbot response: {assistant_text}')

        # save history
        append_to_history("assistant", assistant_text)


if __name__ == "__main__":
    main()
