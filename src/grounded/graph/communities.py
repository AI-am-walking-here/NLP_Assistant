"""M-6.4 — build entity graph and community summaries from extracted triples."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


def _node_id(entity: dict[str, Any]) -> str:
    return f"{entity['type']}:{entity['name']}"


def build_graph_from_triple_rows(rows: list[dict[str, Any]]) -> nx.Graph:
    g = nx.Graph()
    for row in rows:
        paper_id = row["paper_id"]
        entities = json.loads(row["entities_json"])
        relations = json.loads(row["relations_json"])
        nodes = [_node_id(e) for e in entities]
        for nid in nodes:
            if nid not in g:
                g.add_node(nid, paper_id=paper_id)
            else:
                g.nodes[nid]["paper_id"] = paper_id
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                g.add_edge(a, b, kind="cooccur", paper_id=paper_id)
        for rel in relations:
            head = f"method:{rel['head']}" if rel["head"] not in g else rel["head"]
            tail = rel["tail"]
            # relations use display names; map to typed nodes when possible
            for nid in nodes:
                if rel["head"] in nid:
                    head = nid
                if rel["tail"] in nid:
                    tail = nid
            if head in g and tail in g:
                g.add_edge(head, tail, kind=rel["relation"], paper_id=paper_id)
    return g


def detect_communities(g: nx.Graph) -> list[set[str]]:
    if g.number_of_nodes() == 0:
        return []
    try:
        from networkx.algorithms import community

        comps = list(community.greedy_modularity_communities(g))
        return [set(c) for c in comps]
    except Exception as exc:
        logger.warning("Community detection fallback to components: %s", exc)
        return [set(c) for c in nx.connected_components(g)]


def summarize_community(
    community_id: int,
    nodes: set[str],
    g: nx.Graph,
) -> dict[str, Any]:
    paper_ids: set[str] = set()
    labels: list[str] = []
    entity_counts: dict[str, int] = defaultdict(int)
    for nid in nodes:
        label = nid.split(":", 1)[-1][:80]
        labels.append(label)
        entity_counts[label] += 1
        pdata = g.nodes[nid]
        if pdata.get("paper_id"):
            paper_ids.add(pdata["paper_id"])
        for _u, _v, data in g.edges(nid, data=True):
            if data.get("paper_id"):
                paper_ids.add(data["paper_id"])
    top = [name for name, _count in sorted(entity_counts.items(), key=lambda item: (-item[1], item[0]))[:6]]
    summary = (
        f"Community {community_id} focuses on entities: "
        f"{', '.join(top)}"
        + (f" (+{len(labels) - len(top)} more)" if len(labels) > len(top) else "")
        + f", spanning {len(paper_ids)} papers."
    )
    return {
        "community_id": community_id,
        "summary": summary,
        "paper_ids": sorted(paper_ids),
        "n_entities": len(nodes),
        "n_papers": len(paper_ids),
        "extractor": "mock",
    }


def build_communities_from_triples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    g = build_graph_from_triple_rows(rows)
    logger.info(
        "Graph: %d nodes, %d edges",
        g.number_of_nodes(),
        g.number_of_edges(),
    )
    communities = detect_communities(g)
    out: list[dict[str, Any]] = []
    for cid, nodes in enumerate(communities):
        if not nodes:
            continue
        row = summarize_community(cid, nodes, g)
        row["paper_ids_json"] = json.dumps(row.pop("paper_ids"))
        out.append(row)
    return out


def write_communities_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def read_triples_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()
