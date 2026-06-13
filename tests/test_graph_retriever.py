from __future__ import annotations

import json

from grounded.retrieve.graph import GraphRetriever


class _FakeEmbedder:
    def encode(self, texts, show_progress: bool = False):
        del show_progress
        vecs = []
        for text in texts:
            lower = text.lower()
            vecs.append(
                [
                    1.0 if "relevant" in lower else 0.0,
                    1.0 if "query" in lower else 0.0,
                    1.0 if "irrelevant" in lower else 0.0,
                ]
            )
        return vecs


def test_graph_retriever_ranks_chunks_within_community() -> None:
    communities = [
        {
            "community_id": 0,
            "summary": "relevant query cluster",
            "paper_ids_json": json.dumps(["p1", "p2"]),
        }
    ]
    chunks_by_paper = {
        "p1": [
            {"chunk_id": "c1", "paper_id": "p1", "text": "irrelevant filler"},
            {"chunk_id": "c2", "paper_id": "p1", "text": "relevant query evidence"},
        ],
        "p2": [
            {"chunk_id": "c3", "paper_id": "p2", "text": "relevant query second"},
        ],
    }

    retriever = GraphRetriever(communities, chunks_by_paper, _FakeEmbedder())
    hits = retriever.search("relevant query", k_communities=1, k_chunks_per_community=2)

    assert [row["chunk_id"] for row in hits] == ["c2", "c3"]
    assert hits[0]["score"] >= hits[1]["score"]
    assert all("community_id" in row for row in hits)
