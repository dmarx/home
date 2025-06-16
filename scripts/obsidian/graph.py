# obsidian/graph.py - Graph construction utilities for Obsidian vault analysis

from collections import Counter
from pathlib import Path
from loguru import logger
import networkx as nx

from .parser import ObsidianDocument, load_vault_documents


class ObsidianGraph:
    """Builds and analyzes graph structure from Obsidian documents."""
    
    def __init__(self, documents: list[ObsidianDocument] | None = None):
        self.documents = documents or []
        self.graph = nx.DiGraph()
        self._existing_nodes = set()
        
    @classmethod
    def from_vault(cls, vault_path: Path | str) -> 'ObsidianGraph':
        """Create ObsidianGraph from vault directory.
        
        Args:
            vault_path: Path to Obsidian vault
            
        Returns:
            ObsidianGraph instance with loaded documents
        """
        documents = load_vault_documents(vault_path)
        return cls(documents)
    
    def build_graph(self) -> nx.DiGraph:
        """Build directed graph from document links.
        
        Returns:
            NetworkX DiGraph with document links
        """
        logger.info(f"Building graph from {len(self.documents)} documents")
        
        # Add all document nodes first
        for doc in self.documents:
            node_name = doc.node_name
            self.graph.add_node(
                node_name,
                title=doc.title,
                tags=doc.tags,
                exists=True,
                content=doc.body
            )
            self._existing_nodes.add(node_name)
        
        # Add edges for document links
        for doc in self.documents:
            source = doc.node_name
            for target in doc.links:
                self.graph.add_edge(source, target)
                # Add target node if it doesn't exist (phantom/missing documents)
                if target not in self._existing_nodes:
                    self.graph.add_node(target, exists=False)
        
        logger.info(f"Built graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        return self.graph
    
    def get_link_statistics(self) -> dict:
        """Calculate statistics about document links and tags.
        
        Returns:
            Dictionary with tag counts, indegree counts, and other metrics
        """
        tag_counter = Counter()
        indegree_counter = Counter()
        
        for doc in self.documents:
            if doc.tags:
                tag_counter.update(doc.tags)
            if doc.links:
                indegree_counter.update(doc.links)
        
        return {
            'total_documents': len(self.documents),
            'total_unique_links': len(indegree_counter),
            'most_common_tags': tag_counter.most_common(10),
            'most_linked_documents': indegree_counter.most_common(10),
            'existing_documents': len(self._existing_nodes),
            'phantom_documents': len(self.graph.nodes) - len(self._existing_nodes)
        }
    
    def find_candidate_documents(self, min_degree: int = 2) -> list[tuple[str, int]]:
        """Find missing documents that are linked to frequently.
        
        Args:
            min_degree: Minimum number of connections to be considered a candidate
            
        Returns:
            List of (document_name, degree) tuples for missing documents
        """
        if not self.graph:
            self.build_graph()
            
        undirected_graph = self.graph.to_undirected()
        candidates = []
        
        for node in self.graph.nodes():
            if not self.graph.nodes[node].get('exists', False):
                degree = len(list(nx.neighbors(undirected_graph, node)))
                if degree >= min_degree:
                    candidates.append((node, degree))
        
        # Sort by degree (most connected first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
    
    def get_communities(self, algorithm: str = 'louvain') -> dict[str, int]:
        """Detect communities in the document graph.
        
        Args:
            algorithm: Community detection algorithm ('louvain', 'greedy', etc.)
            
        Returns:
            Dictionary mapping node names to community IDs
        """
        if not self.graph:
            self.build_graph()
            
        # Convert to undirected for community detection
        undirected = self.graph.to_undirected()
        
        # Only include existing documents in community detection
        existing_subgraph = undirected.subgraph([
            node for node in undirected.nodes() 
            if undirected.nodes[node].get('exists', False)
        ])
        
        if algorithm == 'louvain':
            try:
                import community as community_louvain
                communities = community_louvain.best_partition(existing_subgraph)
            except ImportError:
                logger.warning("python-louvain not available, falling back to greedy")
                algorithm = 'greedy'
        
        if algorithm == 'greedy':
            community_generator = nx.algorithms.community.greedy_modularity_communities(
                existing_subgraph
            )
            communities = {}
            for i, community_set in enumerate(community_generator):
                for node in community_set:
                    communities[node] = i
        
        logger.info(f"Detected {len(set(communities.values()))} communities using {algorithm}")
        return communities
    
    def export_for_quartz(self, output_path: Path | str, communities: dict[str, int] | None = None):
        """Export community labels for Quartz graph visualization.
        
        Args:
            output_path: Path to write community labels
            communities: Optional pre-computed communities
        """
        if communities is None:
            communities = self.get_communities()
            
        output_path = Path(output_path)
        
        # Create mapping of document titles to community IDs
        title_to_community = {}
        for doc in self.documents:
            node_name = doc.node_name
            if node_name in communities:
                title_to_community[doc.title] = communities[node_name]
        
        # Write as YAML for easy integration with Quartz
        import yaml
        with output_path.open('w') as f:
            yaml.dump(title_to_community, f, default_flow_style=False)
        
        logger.info(f"Exported community labels for {len(title_to_community)} documents to {output_path}")
