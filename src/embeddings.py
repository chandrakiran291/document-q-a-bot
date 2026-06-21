"""
embeddings.py

Custom Chroma-compatible embedding function for Google Gemini.
"""

import os
from typing import cast

import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings

# smaller dim keeps things fast/light for a demo project; max is 3072
EMBEDDING_DIMENSIONS = 768


class GeminiEmbeddingFunction(EmbeddingFunction):
    

    def __init__(self, api_key: str = None, model_name: str = "models/gemini-embedding-001",
                 task_type: str = "retrieval_document"):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it in your .env file "
                "(see .env.example)."
            )
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.task_type = task_type

    def __call__(self, input: Documents) -> Embeddings:
        response = genai.embed_content(
            model=self.model_name,
            content=list(input),
            task_type=self.task_type,
            output_dimensionality=EMBEDDING_DIMENSIONS,
        )
        return cast(Embeddings, response["embedding"])
