
import torch
from typing import List


EMBEDDING_MODEL_NAME = "telepix/PIXIE-Rune-Preview"


class HFMeanPoolingEmbedder:
    def __init__(self, model_name:str=EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 로딩
        raise NotImplementedError

    @torch.no_grad()
    def encode(self, texts:List[str], batch_size:int=32) -> List[List[float]]:

        """
        Arguments:
            - texts      (list(str)) : list of text to encode
            - batch_size (int)       : batch size

        Returns:
            - list(list(float)) : list of Embedding vectors
        """

        raise NotImplementedError

    def __call__(self, input:List[str]) -> List[List[float]]:
        return self.encode(input)

