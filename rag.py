

from typing import Tuple, List, Dict, Any, Collection


TOP_K = 5


# for example, FAQ question is "[A] [B] C" -> return only "C" part
def extract_last_question_text(s:str) -> str:
    return s.split(']')[-1]


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

    raise NotImplementedError
