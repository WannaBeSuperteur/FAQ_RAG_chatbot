
import datetime
import json
from typing import List, Dict, Any

from logging_utils import add_log

CHAT_HISTORY_PATH = "chat_history.txt"


def build_system_prompt() -> str:
    return (
        "너는 '네이버 스마트스토어' 질의응답용 한국어 FAQ 챗봇이다.\n"
        "사용자 질문에 답할 때, 제공된 'FAQ 컨텍스트'를 최우선 근거로 한다.\n"
        "컨텍스트에서 근거를 찾기 어렵거나 스마트스토어와 무관한 질문이면, 억지로 그럴듯한 답변을 하지 말고 솔직하게 답변이 불가능하다고 사과한다.\n"
        "사용자 질문에 대한 응답은 항상 한국어로 한다.\n"
        "답변의 끝부분에는 추가 질문에 대한 제안 문장을 추가한다.\n"
    )


def append_to_history(role:str, content:str, chat_history_path:str=CHAT_HISTORY_PATH) -> None:
    item = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "role": role,
        "content": content,
    }

    try:
        with open(chat_history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        add_log(tag='info', case_id=16, content='append to chat history successful!')
    except Exception as e:
        add_log(tag='error', case_id=17, content=f'append to chat history failed. error: {e}')


def load_recent_history(max_messages:int, chat_history_path:str=CHAT_HISTORY_PATH) -> List[Dict[str, str]]:
    """
        Arguments:
            - max_messages      (int) : count of messages to load, for recent history
            - chat_history_path (str) : file path for chat history

        Returns:
            - (list(dict)) : loaded messages
    """

    # get last N (= the value of max_messages) lines
    with open(chat_history_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = list(filter(lambda x: len(x) >= 1, lines))
    lines = lines[-max_messages:]

    # return loaded messages
    loaded_messages: List[Dict[str, str]] = []
    for line in lines:
        try:
            line_as_json = json.loads(line)
            if line_as_json.get("role") in ("user", "assistant"):
                loaded_messages.append({"role": line_as_json["role"], "content": line_as_json.get("content", "")})
            add_log(tag='info', case_id=20, content='chat history load successful!')
        except Exception as e:
            add_log(tag='error', case_id=21, content=f'chat history load failed. error: {e}')

    return loaded_messages


def build_prompt_with_rag_result(user_query:str, rag_retrieved_faqs:List[Dict[str, Any]]) -> str:
    """
        Arguments:
            - user_query (str) : original user question (= user prompt)
            - rag_retrieved_faqs (list(dict(str))) : retrieved FAQs (using RAG)

        Returns:
            - (str) : prompt with RAG result
    """

    rag_retrieved_faq_lines = []
    for i, c in enumerate(rag_retrieved_faqs, 1):
        rag_retrieved_faq_lines.append(
            f"[질문/답변 {i}]\n"
            f"- 질문: {c['matched_question']}\n"
            f"- 답변: {c['answer']}\n"
        )
    no_faq_message = "(FAQ 추출 결과가 없습니다. 사용자의 질문이 스마트스토어와 관련성이 낮다고 판단한다고 답변해야 합니다.)"
    faq_context = "\n".join(rag_retrieved_faq_lines) if rag_retrieved_faq_lines else no_faq_message
    add_log(tag='info', case_id=33, content=f'retrieved FAQ count: {len(rag_retrieved_faq_lines)}')
    add_log(tag='info', case_id=30, content=f'faq_context: {faq_context}')

    prompt_with_rag_result = (
        "아래는 사용자의 질문에 따라 검색된 FAQ 컨텍스트입니다.\n"
        "이 컨텍스트를 근거로 사용자의 질문에 답변합니다.\n\n"
        f"=== FAQ 컨텍스트 ===\n{faq_context}\n\n"
        f"=== 사용자 질문 ===\n{user_query}\n"
    )

    add_log(tag='info', case_id=31, content=f'prompt_with_rag_result: {prompt_with_rag_result}')
    return prompt_with_rag_result
