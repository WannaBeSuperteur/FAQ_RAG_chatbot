from logging_utils import add_log
from utils import extract_last_question_text
from typing import Tuple, List, Dict, Any
from chromadb.api.models.Collection import Collection


TOP_K = 5


# retrieve FAQ documents using RAQ
def retrieve_top_k(collection: Collection, user_query:str, k:int=TOP_K) -> Tuple[List[Dict[str, Any]], float]:
    """
        Arguments:
            - collection (collection) : Chroma DB collection
            - user_query (str)        : original user question (= user prompt)
            - k          (int)        : count of FAQ documents to retrieve

        Returns:
            - rag_retrieved_faqs (list(dict(str))) : retrieved FAQs (using RAG)
            - shortest_distance  (float)           : shortest distance (from user question)
    """

    q = extract_last_question_text(user_query)
    res = collection.query(
        query_texts=[q],
        n_results=k,
    )

    metas = (res.get("metadatas") or [[]])[0]
    distances = (res.get("distances") or [[]])[0]

    rag_retrieved_faqs: List[Dict[str, Any]] = []
    for meta, distance in zip(metas, distances):
        faq_item = {
            "distance": float(distance),
            "matched_question": (meta or {}).get("question", ""),
            "answer": (meta or {}).get("answer", ""),
        }
        rag_retrieved_faqs.append(faq_item)
        add_log(tag='info', case_id=29, content=f'FAQ item: {faq_item}')

    shortest_distance = float(distances[0]) if distances else 1.0

    add_log(tag='info', case_id=18, content=f'TOP-K retrieved. shortest_distance: {shortest_distance}')
    return rag_retrieved_faqs, shortest_distance
