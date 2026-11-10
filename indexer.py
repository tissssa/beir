import os
import random
import time

from beir.datasets.data_loader_hf import HFDataLoader
from beir.retrieval import models
from beir.retrieval.search.dense import FlatIPFaissSearch


COLLECTION = "BeIR/arguana"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = "./indexes"
INDEX_PREFIX = "arguana"
EXAMPLE_QUERY_COUNT = 5
EXAMPLE_RESULTS_COUNT = 3


class IndexableFlatIPFaissSearch(FlatIPFaissSearch):
    def encode(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        encode_output_path: str = "./embeddings/",
        overwrite: bool = False,
        query_filename: str = "queries.pkl",
        corpus_filename: str = "corpus.*.pkl",
        **kwargs,
    ) -> None:
        raise NotImplementedError("Utilise index() pour créer l'index FAISS.")

    def search_from_files(
        self,
        query_embeddings_file: str,
        corpus_embeddings_files: list[str],
        top_k: int,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        raise NotImplementedError("Cette classe utilise l'index FAISS sauvegardé.")


def main():
    dataset = HFDataLoader(hf_repo=COLLECTION, streaming=False, keep_in_memory=False)
    corpus_ds, queries_ds, _ = dataset.load(split="test")

    # Keep each document identifier (docno) paired with its title and text.
    corpus = {
        row["id"]: {
            "title": row.get("title") or "",
            "text": row.get("text") or "",
        }
        for row in corpus_ds
    }
    queries = {row["id"]: row["text"] for row in queries_ds}
    print(f"Collection : {COLLECTION}")
    print(f"Documents à indexer : {len(corpus)}")
    print(f"Requêtes de test disponibles : {len(queries)}")

    model = models.SentenceBERT(MODEL_NAME)
    indexer = IndexableFlatIPFaissSearch(model, batch_size=64, corpus_chunk_size=5000)

    indexing_start = time.perf_counter()
    indexer.index(corpus, score_function="cos_sim")
    indexing_seconds = time.perf_counter() - indexing_start

    os.makedirs(INDEX_DIR, exist_ok=True)
    save_start = time.perf_counter()
    indexer.save(INDEX_DIR, prefix=INDEX_PREFIX, ext="flat")
    save_seconds = time.perf_counter() - save_start

    index_files = [
        os.path.join(INDEX_DIR, f"{INDEX_PREFIX}.flat.faiss"),
        os.path.join(INDEX_DIR, f"{INDEX_PREFIX}.flat.tsv"),
    ]
    index_size_bytes = sum(os.path.getsize(path) for path in index_files)
    print(f"\nDurée de calcul de l'index : {indexing_seconds:.2f} secondes")
    print(f"Durée de sauvegarde : {save_seconds:.2f} secondes")
    print(f"Taille totale des fichiers d'index : {index_size_bytes / (1024**2):.2f} MiB")
    for path in index_files:
        print(f"  {path}: {os.path.getsize(path) / (1024**2):.2f} MiB")

    sample_ids = random.Random(42).sample(
        sorted(queries), min(EXAMPLE_QUERY_COUNT, len(queries))
    )
    sample_queries = {query_id: queries[query_id] for query_id in sample_ids}
    search_start = time.perf_counter()
    results = indexer.search(
        corpus,
        sample_queries,
        top_k=EXAMPLE_RESULTS_COUNT,
        score_function="cos_sim",
    )
    search_seconds = time.perf_counter() - search_start
    print(
        f"\nExemples de recherche ({len(sample_queries)} requêtes, "
        f"top {EXAMPLE_RESULTS_COUNT}; durée totale : {search_seconds:.2f} secondes)"
    )

    for query_id, query_text in sample_queries.items():
        print(f"\nRequête {query_id}: {query_text}")
        for rank, (doc_id, score) in enumerate(results[query_id].items(), start=1):
            document = corpus[doc_id]
            excerpt = " ".join(document["text"].split())[:240]
            print(
                f"  {rank}. docno={doc_id} score={score:.4f} "
                f"title={document['title']!r} text={excerpt!r}"
            )

    print(f"\nIndex enregistré dans {INDEX_DIR}")


if __name__ == "__main__":
    main()
