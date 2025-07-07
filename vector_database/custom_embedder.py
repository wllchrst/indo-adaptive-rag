from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from chromadb import Documents, Embeddings, EmbeddingFunction

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

class IndoEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return model.encode(input, normalize_embeddings=True) 
