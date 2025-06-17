# scripts/colorclass_processor.py - Add unique colorclass tags with community detection

from pathlib import Path
from loguru import logger
import yaml
import fire
from omegaconf import OmegaConf
import networkx as nx
from collections import Counter
import numpy as np
import sys
import random

# Import community detection algorithms from karateclub
from karateclub import (
    LabelPropagation, Louvain, LeidenAlgorithm, EgoNetSplitter,
    SCD, GEMSEC, BigClam, DANMF, NNSED, MNMF, ClusterONE,
    OverlappingCommunityDetection, CommunityDetection
)

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from obsidian import ObsDoc, build_graph, load_corpus


class ColorclassProcessor:
    """Processes Obsidian vault to add unique colorclass tags with community detection."""
    
    # Available community detection algorithms
    AVAILABLE_ALGORITHMS = {
        'label_propagation': LabelPropagation,
        'louvain': Louvain,
        'leiden': LeidenAlgorithm,
        'ego_net_splitter': EgoNetSplitter,
        'scd': SCD,
        'gemsec': GEMSEC,
        'bigclam': BigClam,
        'danmf': DANMF,
        'nnsed': NNSED,
        'mnmf': MNMF,
        'cluster_one': ClusterONE,
    }
    
    def __init__(self, config_path: str | None = None):
        """Initialize processor with optional config file."""
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: str | None) -> dict:
        """Load configuration from YAML file or use defaults."""
        default_config = OmegaConf.create({
            'source_tag': 'sod/root',  # Now used for filtering documents, not seeds
            'colorclass_prefix': 'colorclass',
            'dry_run': False,
            'backup_originals': True,
            'community_detection': {
                'algorithm': 'louvain',  # Default algorithm
                'algorithm_params': {    # Parameters passed to the algorithm
                    'seed': 42,
                    'resolution': 1.0    # For algorithms that support it
                },
                'min_community_size': 2,  # Minimum size for a community to get colorclass
                'filter_by_source_tag': False,  # Whether to only cluster documents with source_tag
                'naming_scheme': 'largest_node',  # 'cluster_id', 'largest_node', or 'sequential'
            }
        })
        
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                file_config = OmegaConf.load(config_path)
                return OmegaConf.merge(default_config, file_config)
            else:
                logger.warning(f"Config file not found: {config_path}, using defaults")
        
        return default_config
    
    def list_algorithms(self) -> list[str]:
        """List available community detection algorithms."""
        return list(self.AVAILABLE_ALGORITHMS.keys())
    
    def process_vault(
        self,
        vault_path: str,
        source_tag: str | None = None,
        dry_run: bool | None = None,
        algorithm: str | None = None
    ) -> dict[str, str]:
        """Process vault to add colorclass tags using community detection.
        
        Args:
            vault_path: Path to Obsidian vault directory
            source_tag: Tag to filter documents (overrides config)
            dry_run: If True, show what would be changed without modifying files
            algorithm: Community detection algorithm to use (overrides config)
            
        Returns:
            Dictionary mapping article names to their assigned colorclass tags
        """
        vault_path = Path(vault_path)
        source_tag = source_tag or self.config.source_tag
        dry_run = dry_run if dry_run is not None else self.config.dry_run
        algorithm = algorithm or self.config.community_detection.algorithm
        
        if algorithm not in self.AVAILABLE_ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {self.list_algorithms()}")
        
        logger.info(f"Processing vault: {vault_path}")
        logger.info(f"Algorithm: {algorithm}")
        logger.info(f"Source tag filter: {source_tag}")
        logger.info(f"Dry run: {dry_run}")
        
        # Load all documents
        _corpus = load_corpus(vault_path)

        # Prune cruft
        # WARNING: this deletes content.
        corpus = []
        for doc in _corpus:
            if doc.tags and any(['prune' in tag for tag in doc.tags]):
                Path(doc.fpath).unlink() 
                logger.warning(f"Pruned {doc.title}")
                continue
            else:
                corpus.append(doc)
        
        # # Filter documents if configured
        # if self.config.community_detection.filter_by_source_tag:
        #     filtered_corpus = []
        #     for doc in corpus:
        #         if doc.tags and source_tag in doc.tags:
        #             filtered_corpus.append(doc)
        #     logger.info(f"Filtered to {len(filtered_corpus)} documents with tag '{source_tag}'")
        #     clustering_corpus = filtered_corpus
        # else:
        #     clustering_corpus = corpus
        
        
        clustering_corpus = corpus
        
        # Run community detection
        assignments = self._detect_communities(clustering_corpus, algorithm)
        
        if not assignments:
            logger.warning("No community assignments generated")
            return {}
        
        # Apply changes to files
        if not dry_run:
            modified_count = self._apply_assignments(corpus, vault_path, assignments)
            logger.success(f"Modified {modified_count} files")
        else:
            logger.info("Dry run complete - no files modified")
        
        return assignments
    
    def _detect_communities(self, corpus: list[ObsDoc], algorithm: str) -> dict[str, str]:
        """Use community detection to assign colorclass tags."""
        logger.info(f"Starting community detection with {algorithm}...")
        
        # Build graph from corpus
        graph = build_graph(corpus)
        
        # Filter to existing documents only (no phantom nodes)
        existing_nodes = []
        for doc in corpus:
            if doc.node_name in graph.nodes:
                existing_nodes.append(doc.node_name)
        
        subgraph = graph.subgraph(existing_nodes).copy()
        logger.info(f"Clustering subgraph with {len(subgraph.nodes)} nodes, {len(subgraph.edges)} edges")
        
        if len(subgraph.nodes) < 2:
            logger.warning("Graph too small for community detection")
            return {}
        
        # Convert to undirected for most algorithms
        undirected_graph = subgraph.to_undirected()
        
        # Create mapping from node names to integer IDs
        node_to_id = {node: i for i, node in enumerate(undirected_graph.nodes)}
        id_to_node = {i: node for node, i in node_to_id.items()}
        
        # Convert NetworkX graph to integer-indexed graph
        edges = [(node_to_id[u], node_to_id[v]) for u, v in undirected_graph.edges]
        indexed_graph = nx.Graph()
        indexed_graph.add_nodes_from(range(len(node_to_id)))
        indexed_graph.add_edges_from(edges)
        
        # Initialize and run the selected algorithm
        try:
            algorithm_class = self.AVAILABLE_ALGORITHMS[algorithm]
            
            # Get algorithm parameters from config
            params = dict(self.config.community_detection.algorithm_params)
            
            # Filter parameters that the algorithm actually accepts
            import inspect
            algorithm_signature = inspect.signature(algorithm_class.__init__)
            valid_params = {
                key: value for key, value in params.items() 
                if key in algorithm_signature.parameters
            }
            
            logger.info(f"Using algorithm parameters: {valid_params}")
            model = algorithm_class(**valid_params)
            
            # Fit the model
            model.fit(indexed_graph)
            
            # Get community assignments
            if hasattr(model, 'get_memberships'):
                memberships = model.get_memberships()
            elif hasattr(model, 'get_cluster_centers'):
                # For overlapping community detection algorithms
                memberships = model.get_cluster_centers()
            else:
                raise AttributeError(f"Algorithm {algorithm} doesn't have expected output method")
            
            logger.info(f"Community detection completed with {algorithm}")
            
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return {}
        
        # Process community assignments
        assignments = self._process_memberships(memberships, id_to_node, undirected_graph)
        
        return assignments
    
    def _process_memberships(
        self, 
        memberships: dict[int, int], 
        id_to_node: dict[int, str],
        graph: nx.Graph
    ) -> dict[str, str]:
        """Process community memberships into colorclass assignments."""
        # Group nodes by community
        communities = {}
        for node_id, community_id in memberships.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(id_to_node[node_id])
        
        # Filter communities by minimum size
        min_size = self.config.community_detection.min_community_size
        filtered_communities = {
            comm_id: nodes for comm_id, nodes in communities.items() 
            if len(nodes) >= min_size
        }
        
        logger.info(f"Found {len(communities)} communities, {len(filtered_communities)} after size filtering")
        
        # Generate colorclass assignments
        assignments = {}
        naming_scheme = self.config.community_detection.naming_scheme
        
        if naming_scheme == 'cluster_id':
            # Use community ID as colorclass name
            for comm_id, nodes in filtered_communities.items():
                colorclass_tag = f"{self.config.colorclass_prefix}/cluster_{comm_id}"
                for node in nodes:
                    assignments[node] = colorclass_tag
                    
        elif naming_scheme == 'largest_node':
            # Use the node with highest degree as colorclass name
            for comm_id, nodes in filtered_communities.items():
                # Find node with highest degree in community
                max_degree = -1
                representative_node = nodes[0]
                for node in nodes:
                    degree = graph.degree(node)
                    if degree > max_degree:
                        max_degree = degree
                        representative_node = node
                
                colorclass_tag = f"{self.config.colorclass_prefix}/{representative_node}"
                for node in nodes:
                    assignments[node] = colorclass_tag
                    
        elif naming_scheme == 'sequential':
            # Use sequential numbering
            for i, (comm_id, nodes) in enumerate(filtered_communities.items()):
                colorclass_tag = f"{self.config.colorclass_prefix}/community_{i+1}"
                for node in nodes:
                    assignments[node] = colorclass_tag
        
        else:
            raise ValueError(f"Unknown naming scheme: {naming_scheme}")
        
        # Log assignments
        for comm_id, nodes in filtered_communities.items():
            sample_node = nodes[0]
            colorclass_tag = assignments[sample_node]
            logger.info(f"Community {comm_id} ({len(nodes)} nodes) → {colorclass_tag}")
            
        return assignments
    
    def _apply_assignments(self, corpus: list[ObsDoc], vault_path: Path, 
                          assignments: dict[str, str]) -> int:
        """Apply colorclass assignments to document files."""
        modified_count = 0
        
        for doc in corpus:
            if doc.node_name in assignments:
                colorclass_tag = assignments[doc.node_name]
                if self._add_colorclass_tag(doc, vault_path, colorclass_tag):
                    modified_count += 1
        
        return modified_count
    
    def _add_colorclass_tag(self, doc: ObsDoc, vault_path: Path, colorclass_tag: str) -> bool:
        """Add colorclass tag to a document's frontmatter.
        
        Args:
            doc: ObsDoc instance to modify
            vault_path: Path to vault directory
            colorclass_tag: The colorclass tag to add
            
        Returns:
            True if file was modified, False otherwise
        """
        file_path = doc.fpath if doc.fpath else vault_path / f"{doc.title}.md"
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        # Check if colorclass tag already exists
        existing_colorclass = None
        if doc.tags:
            for tag in doc.tags:
                if (tag is not None) and tag.startswith(f"{self.config.colorclass_prefix}/"):
                    existing_colorclass = tag
                    break
        
        if existing_colorclass == colorclass_tag:
            logger.debug(f"Colorclass tag already correct for {doc.title}")
            return False
        
        # Backup original if configured
        if self.config.backup_originals:
            backup_path = file_path.with_suffix('.md.bak')
            if not backup_path.exists():
                backup_path.write_text(file_path.read_text(encoding='utf-8'))
        
        # Read original content
        content = file_path.read_text(encoding='utf-8')
        
        # Parse and modify frontmatter
        frontmatter, body = self._extract_frontmatter_raw(content)
        
        if frontmatter is None:
            # No frontmatter - create new
            new_frontmatter = {'tags': [colorclass_tag]}
            new_content = self._rebuild_document(new_frontmatter, body)
        else:
            # Modify existing frontmatter
            if 'tags' not in frontmatter:
                frontmatter['tags'] = []
            elif not isinstance(frontmatter['tags'], list):
                frontmatter['tags'] = [frontmatter['tags']]
            
            # Remove existing colorclass tag if present
            frontmatter['tags'] = [
                tag for tag in frontmatter['tags'] 
                if (tag is not None) and (not tag.startswith(f"{self.config.colorclass_prefix}/"))
            ]
            
            # Add new colorclass tag
            frontmatter['tags'].append(colorclass_tag)
            
            new_content = self._rebuild_document(frontmatter, body)
        
        # Write modified content
        file_path.write_text(new_content, encoding='utf-8')
        logger.info(f"Added {colorclass_tag} to {doc.title}")
        return True
    
    def _extract_frontmatter_raw(self, content: str) -> tuple[dict | None, str]:
        """Extract frontmatter preserving original structure."""
        if not content.startswith('---'):
            return None, content
        
        try:
            _, frontmatter_text, body = content.split('---', 2)
            frontmatter = yaml.safe_load(frontmatter_text)
            return frontmatter, body
        except ValueError:
            return None, content
        except yaml.YAMLError as e:
            logger.warning(f"YAML parse error: {e}")
            return None, content
    
    def _rebuild_document(self, frontmatter: dict, body: str) -> str:
        """Rebuild document with modified frontmatter."""
        frontmatter_yaml = yaml.dump(
            frontmatter, 
            default_flow_style=False, 
            allow_unicode=True,
            sort_keys=False
        )
        return f"---\n{frontmatter_yaml}---{body}"
    
    def analyze_community_structure(self, vault_path: str, algorithm: str | None = None) -> dict:
        """Analyze community structure that would be detected."""
        vault_path = Path(vault_path)
        algorithm = algorithm or self.config.community_detection.algorithm
        
        if algorithm not in self.AVAILABLE_ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        logger.info(f"Analyzing community structure with algorithm: {algorithm}")
        
        corpus = load_corpus(vault_path)
        graph = build_graph(corpus)
        
        # Filter to existing documents
        existing_nodes = [doc.node_name for doc in corpus if doc.node_name in graph.nodes]
        subgraph = graph.subgraph(existing_nodes).to_undirected()
        
        if len(subgraph.nodes) < 2:
            return {'error': 'Graph too small for analysis'}
        
        # Create integer-indexed graph
        node_to_id = {node: i for i, node in enumerate(subgraph.nodes)}
        id_to_node = {i: node for node, i in node_to_id.items()}
        
        edges = [(node_to_id[u], node_to_id[v]) for u, v in subgraph.edges]
        indexed_graph = nx.Graph()
        indexed_graph.add_nodes_from(range(len(node_to_id)))
        indexed_graph.add_edges_from(edges)
        
        # Run community detection
        try:
            algorithm_class = self.AVAILABLE_ALGORITHMS[algorithm]
            params = dict(self.config.community_detection.algorithm_params)
            
            # Filter valid parameters
            import inspect
            algorithm_signature = inspect.signature(algorithm_class.__init__)
            valid_params = {
                key: value for key, value in params.items() 
                if key in algorithm_signature.parameters
            }
            
            model = algorithm_class(**valid_params)
            model.fit(indexed_graph)
            memberships = model.get_memberships()
            
        except Exception as e:
            return {'error': f'Community detection failed: {e}'}
        
        # Analyze results
        communities = {}
        for node_id, community_id in memberships.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(id_to_node[node_id])
        
        # Calculate statistics
        community_sizes = [len(nodes) for nodes in communities.values()]
        min_size = self.config.community_detection.min_community_size
        valid_communities = [nodes for nodes in communities.values() if len(nodes) >= min_size]
        
        analysis = {
            'total_documents': len(corpus),
            'clustered_documents': len(existing_nodes),
            'total_communities': len(communities),
            'valid_communities': len(valid_communities),
            'community_size_stats': {
                'min': min(community_sizes) if community_sizes else 0,
                'max': max(community_sizes) if community_sizes else 0,
                'mean': sum(community_sizes) / len(community_sizes) if community_sizes else 0,
                'sizes': sorted(community_sizes, reverse=True)[:10]  # Top 10 sizes
            },
            'coverage': len([n for nodes in valid_communities for n in nodes]) / len(existing_nodes) if existing_nodes else 0,
            'algorithm': algorithm,
            'parameters': valid_params
        }
        
        logger.info(f"Community analysis: {analysis}")
        return analysis


def main():
    """CLI entry point for colorclass processor."""
    logger.add("colorclass_processor.log", rotation="1 MB")
    fire.Fire(ColorclassProcessor)


if __name__ == "__main__":
    main()
