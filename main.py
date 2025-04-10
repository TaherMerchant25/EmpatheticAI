# main.py
from fastapi import FastAPI, HTTPException
from vectordb import VectorDB
from gemini_api import GeminiFlash

app = FastAPI()
vector_db = VectorDB()
gemini = GeminiFlash()

@app.post("/chat/")
async def chat(user_query: str):
    try:
        # Retrieve relevant documents using RAG with randomized vectors
        relevant_docs = vector_db.retrieve(user_query)
        
        # Generate response using Gemini Flash 2.0
        response = gemini.generate_response(user_query, relevant_docs)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# vectordb.py
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

# gemini_api.py
import google.generativeai as genai

class GeminiFlash:
    def __init__(self):
        genai.configure(api_key= "AIzaSyDIjEYblJJ242NAnkc2nfn_Mufo1RbSKYA")
        self.model = genai.GenerativeModel("gemini-flash-2.0")
    
    def generate_response(self, query, retrieved_docs):
        prompt = f"""
        You are a psychiatrist chatbot. Use the following context to respond empathetically to the user's query:
        {retrieved_docs}
        
        User: {query}
        """
        response = self.model.generate_content(prompt)
        return response.text
