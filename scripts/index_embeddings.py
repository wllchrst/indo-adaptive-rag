"""
One-time script to add text_embedding (dense_vector) to all Elasticsearch indices.

Usage:
    python -m scripts.index_embeddings [--indices indoqa,hotpot,qasina] [--batch-size 100] [--dry-run]

Steps:
    1. Ensures each index has a 'text_embedding' dense_vector mapping.
    2. Fetches all documents missing the embedding field.
    3. Encodes text using paraphrase-multilingual-MiniLM-L12-v2.
    4. Bulk-updates documents with their embeddings.
"""

import argparse
import sys
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

from helpers import env_helper

INDICES = ['indoqa', 'hotpot', 'qasina']
EMBEDDING_FIELD = 'text_embedding'
EMBEDDING_DIMENSION = 384
EMBEDDING_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'


def ensure_mapping(es: Elasticsearch, index: str):
    mapping = es.indices.get_mapping(index=index)
    properties = mapping[index]['mappings'].get('properties', {})
    if EMBEDDING_FIELD not in properties:
        es.indices.put_mapping(
            index=index,
            properties={
                EMBEDDING_FIELD: {
                    'type': 'dense_vector',
                    'dims': EMBEDDING_DIMENSION,
                    'index': True,
                    'similarity': 'cosine',
                }
            }
        )
        print(f"[{index}] Added '{EMBEDDING_FIELD}' mapping")
    else:
        print(f"[{index}] '{EMBEDDING_FIELD}' mapping already exists")


def get_unembedded_docs(es: Elasticsearch, index: str, batch_size: int = 100) -> list:
    query = {
        "bool": {
            "must_not": [
                {"exists": {"field": EMBEDDING_FIELD}}
            ]
        }
    }
    hits = []
    from_ = 0
    while True:
        result = es.search(
            index=index,
            query=query,
            size=batch_size,
            _source=['text'],
            from_=from_,
        )
        batch = result['hits']['hits']
        if not batch:
            break
        hits.extend(batch)
        from_ += batch_size
        if len(batch) < batch_size:
            break
    return hits


def index_embeddings(es: Elasticsearch, indices: list[str], batch_size: int, dry_run: bool = False):
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    for index in indices:
        if not es.indices.exists(index=index):
            print(f"[{index}] Index does not exist, skipping")
            continue

        ensure_mapping(es, index)

        hits = get_unembedded_docs(es, index, batch_size=batch_size)
        total = len(hits)
        print(f"[{index}] Found {total} documents without embeddings")

        if total == 0:
            continue

        if dry_run:
            print(f"[{index}] Dry run: would embed {total} documents")
            continue

        texts = [hit['_source']['text'] for hit in hits]
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=batch_size,
                                  show_progress_bar=True)

        actions = []
        for hit, embedding in zip(hits, embeddings):
            actions.append({
                '_op_type': 'update',
                '_index': index,
                '_id': hit['_id'],
                'doc': {
                    EMBEDDING_FIELD: embedding.tolist(),
                }
            })

        success, errors = helpers.bulk(es, actions, raise_on_error=False)
        print(f"[{index}] Indexed {success}/{total} embeddings")
        if errors:
            for err in errors[:5]:
                print(f"  Error: {err}")


def main():
    parser = argparse.ArgumentParser(description="Add text embeddings to Elasticsearch indices")
    parser.add_argument('--indices', type=str, default=','.join(INDICES),
                        help=f"Comma-separated list of indices (default: {','.join(INDICES)})")
    parser.add_argument('--batch-size', type=int, default=100,
                        help="Batch size for embedding and bulk update (default: 100)")
    parser.add_argument('--dry-run', action='store_true',
                        help="Show what would be done without making changes")
    args = parser.parse_args()

    indices = [i.strip() for i in args.indices.split(',')]
    es = Elasticsearch(env_helper.ELASTIC_HOST)

    print(f"Elasticsearch: {env_helper.ELASTIC_HOST}")
    print(f"Indices: {indices}")
    print(f"Model: {EMBEDDING_MODEL_NAME}")
    print()

    index_embeddings(es, indices, args.batch_size, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
