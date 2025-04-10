import numpy as np
import random

class VectorDB:
    def __init__(self):
        self.documents = {
            "doc1": "Depression is a common mental disorder with persistent sadness and lack of interest.",
            "doc2": "Cognitive Behavioral Therapy (CBT) is an effective treatment for anxiety and depression.",
            "doc3": "Mindfulness meditation helps in reducing stress and improving emotional regulation."
        }
        self.vectors = {doc: self._random_vector() for doc in self.documents}
    
    def _random_vector(self, dim=128):
        return np.random.rand(dim)  # Randomly initialized vector

    def retrieve(self, query):
        query_vector = self._random_vector()
        scores = {doc: np.dot(query_vector, vec) for doc, vec in self.vectors.items()}
        
        # Return top 2 most relevant documents
        top_docs = sorted(scores, key=scores.get, reverse=True)[:2]
        return [self.documents[doc] for doc in top_docs]
