# obsidian/graph.py - Basic graph construction from documents

import networkx as nx
from collections import Counter
from pathlib import Path

from .parser import ObsDoc


def load_corpus(obs_root: Path | str) -> list[ObsDoc]:
    """Load all markdown documents from vault directory."""
    obs_root = Path(obs_root)
    return [ObsDoc.from_path(fpath) for fpath in obs_root.glob("*.md")]


def build_graph(corpus: list[ObsDoc]) -> nx.DiGraph:
    """Build directed graph from document corpus."""
    G = nx.DiGraph()
    std_titles = []
    
    for doc in corpus:
        src = doc.title.lower()
        std_titles.append(src)
        edges = [(src, tgt) for tgt in doc.links]
        G.add_edges_from(edges)
    
    # Mark which nodes actually exist vs are just linked to
    nx.set_node_attributes(G, False, 'exists')
    nx.set_node_attributes(G, {title: True for title in std_titles}, 'exists')
    
    return G


def get_link_statistics(corpus: list[ObsDoc]) -> tuple[Counter, Counter]:
    """Calculate tag and indegree statistics."""
    tags = Counter()
    indegree = Counter()
    
    for doc in corpus:
        if doc.tags:
            tags += Counter(doc.tags)
        if doc.links:
            indegree += Counter(doc.links)
    
    return tags, indegree


def find_candidates(G: nx.DiGraph) -> dict[str, int]:
    """Find missing documents with their connection degrees."""
    candidates = [node for node, exists in nx.get_node_attributes(G, 'exists').items() 
                  if not exists]
    
    GU = G.to_undirected()
    cand_degree = {}
    for cand in candidates:
        cand_degree[cand] = len(list(nx.neighbors(GU, cand)))
    
    return cand_degree
