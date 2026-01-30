
import datetime
import json
from typing import List, Dict, Any

from logging_utils import add_log

CHAT_HISTORY_PATH = "chat_history.txt"


def build_instructions() -> str:
    return (
        "너는 '네이버 스마트스토어' 질의응답용 한국어 FAQ 챗봇이다. 다음의 지시를 반드시 따른다.\n"
        " - 1. 사용자 질문에 답할 때, 제공된 'FAQ 컨텍스트'를 최우선 근거로 한다.\n"
        " - 2. 컨텍스트에서 근거를 찾기 어렵거나 스마트스토어와 무관한 질문이면, 억지로 그럴듯한 답변을 하지 말고 솔직하게 답변이 불가능하다고 사과한다.\n"
        " - 3. 사용자 질문에 대한 응답은 항상 한국어로 한다.\n"
        " - 4. 답변의 끝부분에는 질의응답 맥락 측면에서 사용자가 궁금해 할 만한 내용에 대한 추가 사항 안내에 대한 제안을 포함한다.\n"
        "   - 이때 반드시 '~을 안내해 드릴까요? 라는 질문 형식으로 한다.\n"
        "   - 예시: '미성년자도 판매 회원 등록이 가능한가요?'가 사용자 질문이었다면, 답변 끝에 다음을 추가한다.\n"
        "   - 추가 사항 안내 제안 예시 1: 등록에 필요한 서류 안내해드릴까요?\n"
        "   - 추가 사항 안내 제안 예시 2: 등록 절차에 소요되는 시간은 얼마나 걸리는지 안내해 드릴까요?\n"
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
