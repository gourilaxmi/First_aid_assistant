"""
Embedding generation utilities for RAG system
"""
import numpy as np
from typing import List

from utils.logger_config import setup_logger

logger = setup_logger(__name__)


class EmbeddingGenerator:
    """Generate BioBERT embeddings for medical text"""

    def __init__(self, model_name: str = "dmis-lab/biobert-v1.1"):
        logger.info(f"Loading BioBERT model: {model_name}")

        import torch
        from transformers import AutoTokenizer, AutoModel

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"BioBERT loaded on device: {self.device}")

    def generate_embedding(self, text: str) -> np.ndarray:
        torch = self.torch  

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():  
            outputs = self.model(**inputs)

        embedding = outputs.last_hidden_state.mean(dim=1)
        embedding = embedding.cpu().numpy().flatten()

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def generate_batch_embeddings(
        self,
        texts: List[str],
        batch_size: int = 8
    ) -> List[np.ndarray]:
        torch = self.torch  
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():  
                outputs = self.model(**inputs)

            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            batch_embeddings = batch_embeddings.cpu().numpy()

            for emb in batch_embeddings:
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                embeddings.append(emb)

        return embeddings
