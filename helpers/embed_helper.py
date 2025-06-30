import numpy as np
import faiss
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

class EmbedHelper:
    def __init__(self):
        self.hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    
    def get_text_similar(self, target: str, texts: list[str], top_n=3):
        if len(texts) == 0:
            print("There is no information")
            return None

        self.target_vector = self.embed_single_text(target)
        self.text_vectors = self.embed_texts(texts)

        target_vector_np = np.array(self.target_vector).astype('float32').reshape(1, -1)
        text_vectors_np = np.array(self.text_vectors).astype('float32')
        
        dimension = text_vectors_np.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(text_vectors_np)
        
        distances, indices = index.search(target_vector_np, top_n)
        
        results = [(texts[i], float(distances[0][j])) for j, i in enumerate(indices[0])]
        return results

    def embed_single_text(self, text: str):
        if text is None:
            print('text is none')
            text = ""
        vectors = self.hf.embed_documents([text])
        return vectors[0]

    def embed_texts(self, texts: list[str]):
        cleaned_texts = [t if t is not None else "" for t in texts]
        vectors = self.hf.embed_documents(cleaned_texts)
        return vectors