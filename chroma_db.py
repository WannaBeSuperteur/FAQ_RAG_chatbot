
import pandas as pd
from embedding import HFMeanPoolingEmbedder
from typing import Collection


def build_or_load_chroma(df:pd.DataFrame, embedder:HFMeanPoolingEmbedder) -> Collection:
    """
        Arguments:
            - df       (Pandas DataFrame) : original FAQ DataFrame
            - embedder (Embedder)         : Embedding Function (default: Mean Pooling)

        Returns:
            - (collection) : Chroma DB collection
    """

    raise NotImplementedError
