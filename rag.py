

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
        rag_retrieved_faqs.append(
            {
                "matched_question": (meta or {}).get("question", ""),
                "answer": (meta or {}).get("answer", ""),
                "distance": float(distance),
            }
        )

    shortest_distance = float(distances[0]) if distances else 1.0
    return rag_retrieved_faqs, shortest_distance
