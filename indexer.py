import os

from beir.datasets.data_loader_hf import HFDataLoader
from beir.retrieval import models
from beir.retrieval.search.dense import FlatIPFaissSearch


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


collection = "BeIR/arguana"
# Charger le corpus
dataset = HFDataLoader(
    hf_repo=collection,
    streaming=False,
    keep_in_memory=False,
)
corpus_ds = dataset.load_corpus()

# Convertir le corpus au format attendu par l'indexeur BEIR
corpus = {
    row["id"]: {
        "title": row.get("title") or "",
        "text": row.get("text") or "",
    }
    for row in corpus_ds
}

print(f"Documents chargés : {len(corpus)}")

# Créer et enregistrer l'index vectoriel
model = models.SentenceBERT("sentence-transformers/all-MiniLM-L6-v2")
indexer = IndexableFlatIPFaissSearch(model, batch_size=64, corpus_chunk_size=5000)
indexer.index(corpus, score_function="cos_sim")

os.makedirs("./indexes", exist_ok=True)
indexer.save("./indexes", prefix="arguana", ext="flat")
print("Index enregistré dans ./indexes")
