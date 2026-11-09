import json
import os

import joblib
import numpy as np
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from beir.datasets.data_loader_hf import HFDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval


COLLECTION = "BeIR/arguana"
INDEX_DIR = "./indexes/arguana_tfidf"
TOP_K = 1000
K_VALUES = [1, 3, 5, 10, 100, 1000]


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

    vectorizer = TfidfVectorizer(stop_words="english")
    document_vectors = vectorizer.fit_transform(documents)
    query_vectors = vectorizer.transform(query_texts)
    scores = linear_kernel(query_vectors, document_vectors, dense_output=False)

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
    save_npz(os.path.join(INDEX_DIR, "corpus_tfidf.npz"), document_vectors)
    joblib.dump(vectorizer, os.path.join(INDEX_DIR, "vectorizer.joblib"))
    with open(os.path.join(INDEX_DIR, "document_ids.json"), "w", encoding="utf-8") as file:
        json.dump(doc_ids, file)

    ndcg, mean_average_precision, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results, K_VALUES
    )
    print("\nRésultats TF-IDF sur ArguAna (split test)")
    for metric_group in (ndcg, mean_average_precision, recall, precision):
        for metric, score in metric_group.items():
            print(f"{metric}: {score:.4f}")
    print(f"\nIndex TF-IDF enregistré dans {INDEX_DIR}")


if __name__ == "__main__":
    main()
