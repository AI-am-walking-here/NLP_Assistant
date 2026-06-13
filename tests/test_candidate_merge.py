from __future__ import annotations

from grounded.generate.baselines import _merge_candidates


def test_merge_candidates_can_cap_graph_contribution() -> None:
    vector_hits = [
        {"chunk_id": "v1", "paper_id": "p1"},
        {"chunk_id": "v2", "paper_id": "p2"},
        {"chunk_id": "v3", "paper_id": "p3"},
    ]
    graph_hits = [
        {"chunk_id": "g1", "paper_id": "g1"},
        {"chunk_id": "g2", "paper_id": "g2"},
        {"chunk_id": "g3", "paper_id": "g3"},
    ]

    merged = _merge_candidates(
        vector_hits,
        graph_hits,
        max_candidates=4,
        per_list_cap=[4, 1],
    )

    assert [row["chunk_id"] for row in merged] == ["v1", "v2", "v3", "g1"]
