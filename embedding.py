
import torch
from typing import List
from transformers import AutoTokenizer, AutoModel
import numpy as np


EMBEDDING_MODEL_NAME = "telepix/PIXIE-Rune-Preview"


class HFMeanPoolingEmbedder:
    def __init__(self, model_name:str=EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts:List[str], batch_size:int=32) -> List[List[float]]:

        """
        Arguments:
            - texts      (list(str)) : list of text to encode
            - batch_size (int)       : batch size

        Returns:
            - list(list(float)) : list of Embedding vectors
        """

        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            # shape notations:
            # - B: batch size
            # - T: max token count (max value of token counts for all texts)
            # - H: hidden = embedding dimension
            out = self.model(**enc)
            last_hidden = out.last_hidden_state         # shape: (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1)  # shape: (B, T, 1)
            masked = last_hidden * mask
            summed = masked.sum(dim=1)                  # shape: (B, H)
            counts = mask.sum(dim=1).clamp(min=1)       # shape: (B, 1)
            mean_pooled = summed / counts

            mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            all_vecs.append(mean_pooled.cpu().numpy())

        vecs = np.vstack(all_vecs).astype(np.float32)
        return vecs.tolist()

    def __call__(self, input:List[str]) -> List[List[float]]:
        return self.encode(input)

