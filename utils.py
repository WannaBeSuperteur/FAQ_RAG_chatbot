
import pickle
import pandas as pd


# for example, FAQ question is "[A] [B] C" -> return only "C" part
def extract_last_question_text(s:str) -> str:
    return s.split(']')[-1]


def load_faq_as_dataframe(faq_path:str):

    """
    Arguments:
        - faq_path (str) : Pickle FAQ 파일 경로

    Returns:
        - (Pandas DataFrame) : Pandas DataFrame 형태로 load 된 Pickle FAQ
    """

    with open(faq_path, "rb") as f:
        faq_pickle = pickle.load(f)

    faq_dict = {'question': [], 'answer': []}
    for q, a in faq_pickle.items():
        faq_dict['question'].append(q)
        faq_dict['answer'].append(a)

    faq_df = pd.DataFrame(faq_dict)
    faq_df["q_for_rag"] = faq_df["question"].apply(extract_last_question_text)

    faq_df = faq_df[faq_df["q_for_rag"].str.strip().astype(bool)]
    faq_df = faq_df[faq_df["answer"].str.strip().astype(bool)]
    faq_df = faq_df.reset_index(drop=True)

    return faq_df
