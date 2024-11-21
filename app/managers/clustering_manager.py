from app.config import AppConfig
import numpy as np
from sklearn.cluster import KMeans
from typing import List
from app.schemas.langchain_embedding import DocumentEmbeddingModel

config = AppConfig.get_config()


class ClusteringManager:
    @classmethod
    async def perform_k_means(
        cls, document_embeddings: List[DocumentEmbeddingModel]
    ) -> List[int]:
        # Calculate an estimated number of clusters based on the number of samples
        vectors = np.array([doc.embedding for doc in document_embeddings])
        num_clusters = 3
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
        return kmeans.labels_.tolist()
