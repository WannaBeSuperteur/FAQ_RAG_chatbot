
import pandas as pd
from embedding import HFMeanPoolingEmbedder
import chromadb
from chromadb.api.models.Collection import Collection
from tqdm import tqdm


CHROMA_DIR = "chroma_db"                     # Chroma DB save path
CHROMA_COLLECTION = "smartstore_faq_q_only"  # Chroma DB collection name
CHROMA_DB_BATCH_SIZE = 256


def build_or_load_chroma(faq_df:pd.DataFrame, embedder:HFMeanPoolingEmbedder) -> Collection:
    """
        Arguments:
            - faq_df   (Pandas DataFrame) : original FAQ DataFrame
            - embedder (Embedder)         : Embedding Function (default: Mean Pooling)

        Returns:
            - (collection) : Chroma DB collection
    """

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )

    # if collection get successful -> then return this collection
    if collection.count() > 0:
        return collection

    ids = [str(i) for i in range(len(faq_df))]
    documents = faq_df["q_for_rag"].tolist()
    metadatas = [
        {"question": faq_df.loc[i, "question"], "answer": faq_df.loc[i, "answer"]}
        for i in range(len(faq_df))
    ]

    print(f"[Index] ChromaDB index construction start : N={len(faq_df)}")
    for i in tqdm(range(0, len(faq_df), CHROMA_DB_BATCH_SIZE)):
        collection.add(
            ids=ids[i:i + CHROMA_DB_BATCH_SIZE],
            documents=documents[i:i + CHROMA_DB_BATCH_SIZE],
            metadatas=metadatas[i:i + CHROMA_DB_BATCH_SIZE],
        )
    print("[Index] ChromaDB index construction finished!!")

    return collection
