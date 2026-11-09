import json
import os

import joblib
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import CountVectorizer

from beir.datasets.data_loader_hf import HFDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval


COLLECTION = "BeIR/arguana"
INDEX_DIR = "./indexes/arguana_bm25"
TOP_K = 1000
K_VALUES = [1, 3, 5, 10, 100, 1000]
K1 = 1.2
B = 0.75


def main():
    dataset = HFDataLoader(hf_repo=COLLECTION, streaming=False, keep_in_memory=False)
    corpus_ds, queries_ds, qrels = dataset.load(split="test")

    doc_ids = [str(row["id"]) for row in corpus_ds]
    documents = [
        f'{row.get("title") or ""} {row.get("text") or ""}'.strip()
        for row in corpus_ds
    ]
    query_ids = [str(row["id"]) for row in queries_ds]
    query_texts = [row["text"] for row in queries_ds]
    print(f"Documents chargés : {len(doc_ids)}")
    print(f"Requêtes de test : {len(query_ids)}")

    vectorizer = CountVectorizer()
    document_counts = vectorizer.fit_transform(documents).tocsr()
    query_counts = vectorizer.transform(query_texts).tocsr()

    document_lengths = np.asarray(document_counts.sum(axis=1)).ravel()
    average_document_length = float(document_lengths.mean())
    document_frequency = np.asarray((document_counts > 0).sum(axis=0)).ravel()
    inverse_document_frequency = np.log(
        1 + (len(doc_ids) - document_frequency + 0.5) / (document_frequency + 0.5)
    )

    row_indices = np.repeat(np.arange(document_counts.shape[0]), np.diff(document_counts.indptr))
    term_frequencies = document_counts.data
    length_normalization = K1 * (
        1 - B + B * document_lengths[row_indices] / average_document_length
    )
    bm25_weights = (
        inverse_document_frequency[document_counts.indices]
        * term_frequencies
        * (K1 + 1)
        / (term_frequencies + length_normalization)
    )
    weighted_documents = csr_matrix(
        (bm25_weights, document_counts.indices, document_counts.indptr),
        shape=document_counts.shape,
    )

    scores = (query_counts @ weighted_documents.T).tocsr()
    results = {}
    for row_index, query_id in enumerate(query_ids):
        row = scores.getrow(row_index)
        if row.nnz == 0:
            results[query_id] = {}
            continue

        ranked = np.lexsort((row.indices, -row.data))[:TOP_K]
        results[query_id] = {
            doc_ids[row.indices[index]]: float(row.data[index])
            for index in ranked
        }

    os.makedirs(INDEX_DIR, exist_ok=True)
    save_npz(os.path.join(INDEX_DIR, "corpus_bm25.npz"), weighted_documents)
    joblib.dump(vectorizer, os.path.join(INDEX_DIR, "vectorizer.joblib"))
    with open(os.path.join(INDEX_DIR, "document_ids.json"), "w", encoding="utf-8") as file:
        json.dump(doc_ids, file)
    with open(os.path.join(INDEX_DIR, "parameters.json"), "w", encoding="utf-8") as file:
        json.dump(
            {"k1": K1, "b": B, "average_document_length": average_document_length},
            file,
            indent=2,
        )

    ndcg, mean_average_precision, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results, K_VALUES
    )
    print("\nRésultats BM25 sur ArguAna (split test)")
    for metric_group in (ndcg, mean_average_precision, recall, precision):
        for metric, score in metric_group.items():
            print(f"{metric}: {score:.4f}")
    print(f"\nIndex BM25 enregistré dans {INDEX_DIR}")


if __name__ == "__main__":
    main()
