import pandas as pd
import pytest

from src.vector_store import KnowledgeBase


@pytest.fixture()
def pairs_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_text": [
                "My package is delayed a week, where is it?",
                "The item I received was damaged.",
                "I was charged twice for my order.",
            ],
            "agent_text": [
                "@u1 Sorry for the delay, we are tracking this now.",
                "@u2 Apologies, please request a replacement.",
                "@u3 I will investigate the double charge.",
            ],
            "customer_id": ["1001", "1002", "1003"],
            "tweet_id": [101, 102, 103],
        }
    )


class TestKnowledgeBase:
    def test_ingest_and_search(self, tmp_path, pairs_df) -> None:
        kb = KnowledgeBase(persist_dir=str(tmp_path / "kb"))
        assert kb.count() == 0

        kb.ingest(pairs_df, force=True)
        assert kb.count() == 3

        results = kb.search_similar("When will my package arrive?", top_k=2)
        assert len(results) == 2
        top = results[0]
        assert "metadata" in top
        assert top["metadata"]["customer_id"] == "1001"

    def test_ingest_skips_without_force(self, tmp_path, pairs_df) -> None:
        kb = KnowledgeBase(persist_dir=str(tmp_path / "kb"))
        kb.ingest(pairs_df, force=True)
        n = kb.ingest(pairs_df)  # no force -> skip
        assert n == 0
        assert kb.count() == 3

    def test_force_reingest(self, tmp_path, pairs_df) -> None:
        kb = KnowledgeBase(persist_dir=str(tmp_path / "kb"))
        kb.ingest(pairs_df, force=True)
        kb.ingest(pairs_df, force=True)
        # upsert by tweet_id keeps collection at 3
        assert kb.count() == 3

    def test_missing_columns(self, tmp_path) -> None:
        kb = KnowledgeBase(persist_dir=str(tmp_path / "kb"))
        bad = pd.DataFrame({"customer_text": ["hi"], "agent_text": ["yo"]})
        with pytest.raises(ValueError):
            kb.ingest(bad)

    def test_search_empty_collection(self, tmp_path) -> None:
        kb = KnowledgeBase(persist_dir=str(tmp_path / "kb"))
        assert kb.search_similar("anything") == []
