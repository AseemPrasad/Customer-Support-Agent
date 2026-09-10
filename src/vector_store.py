from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DEFAULT_COLLECTION = "amazon_support_history"
EMBEDDING_MODEL = "BAAI/bge-m3"
BATCH_SIZE = 100
KB_DIR = "./amazon_kb"


class KnowledgeBase:
    """Vector store of historical AmazonHelp customer-agent pairs using ChromaDB + BGE-M3."""

    def __init__(
        self,
        persist_dir: str = KB_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        self.persist_dir = persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_fn
        )

    def count(self) -> int:
        """Return the number of stored documents."""
        return self.collection.count()

    def ingest(self, pairs_df: pd.DataFrame, force: bool = False) -> int:
        """
        Add customer/agent pairs to the collection.

        Each row adds customer_text as the document with metadata:
        agent_text, customer_id, tweet_id.

        Args:
            pairs_df: DataFrame with columns customer_text, agent_text,
                customer_id, tweet_id.
            force: If False and the collection already has documents,
                skip ingestion.

        Returns:
            Number of documents ingested (0 if skipped).
        """
        if self.collection.count() > 0 and not force:
            print(
                "Collection already contains documents; "
                "skipping ingestion (use force=True to re-ingest)."
            )
            return 0

        required = {"customer_text", "agent_text", "customer_id", "tweet_id"}
        missing = required - set(pairs_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        df = pairs_df.reset_index(drop=True)
        total = len(df)
        total_ingested = 0

        for start in range(0, total, BATCH_SIZE):
            chunk = df.iloc[start : start + BATCH_SIZE]
            ids = [str(int(t)) for t in chunk["tweet_id"]]
            documents = [str(t) for t in chunk["customer_text"]]
            metadatas: list[dict[str, Any]] = [
                {
                    "agent_text": str(row.agent_text),
                    "customer_id": str(row.customer_id),
                    "tweet_id": str(int(row.tweet_id)),
                }
                for row in chunk.itertuples(index=False)
            ]
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
            total_ingested += len(chunk)
            print(f"  Ingested {total_ingested:,} / {total:,}")

        print(f"Done. Collection now contains {self.collection.count():,} documents.")
        return total_ingested

    def search_similar(self, query: str, top_k: int = 2) -> list[dict[str, Any]]:
        """
        Return top_k documents most similar to query, with metadata.

        Args:
            query: Customer problem/issue text.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: id, document, distance, metadata.
        """
        if self.collection.count() == 0:
            print("Collection is empty; no results to return.")
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        out: list[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        for doc_id, doc, dist, meta in zip(ids, documents, distances, metadatas, strict=True):
            out.append(
                {
                    "id": doc_id,
                    "document": doc,
                    "distance": dist,
                    "metadata": meta,
                }
            )
        return out


def main() -> None:
    """CLI entry point for vector store ingestion and inspection."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ChromaDB knowledge base for Amazon support history"
    )
    parser.add_argument(
        "--ingest",
        default=None,
        metavar="PARQUET",
        help="Path to a pairs parquet to ingest (default none)",
    )
    parser.add_argument(
        "--dir", default=KB_DIR, help=f"Knowledge base directory (default {KB_DIR})"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Ingest only first N rows (for testing)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if collection is non-empty",
    )
    args = parser.parse_args()

    kb = KnowledgeBase(persist_dir=args.dir)
    print(f"Collection count: {kb.count():,}")

    if args.ingest:
        if not Path(args.ingest).exists():
            raise FileNotFoundError(args.ingest)
        df = pd.read_parquet(
            args.ingest, columns=["customer_text", "agent_text", "customer_id", "tweet_id"]
        )
        if args.max_rows:
            df = df.head(args.max_rows)
            print(f"Using first {len(df):,} rows.")
        kb.ingest(df, force=args.force)
        print(f"Final collection count: {kb.count():,}")


if __name__ == "__main__":
    main()
