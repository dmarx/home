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

# Import community detection algorithms from karateclub (excluding louvain/leiden)
from karateclub import (
    LabelPropagation, EgoNetSplitter, SCD, GEMSEC, BigClam, 
    DANMF, NNSED, MNMF, 
    #ClusterONE
)

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from obsidian import ObsDoc, build_graph, load_corpus


class ColorclassProcessor:
    """Processes Obsidian vault to add unique colorclass tags with community detection."""
    
    # Available community detection algorithms
    NETWORKX_ALGORITHMS = {
        'louvain': 'louvain_communities',
        'leiden': 'leiden_communities',
    }
    
    KARATECLUB_ALGORITHMS = {
        'label_propagation': LabelPropagation,
        'ego_net_splitter': EgoNetSplitter,
        'scd': SCD,
        'gemsec': GEMSEC,
        'bigclam': BigClam,
        'danmf': DANMF,
        'nnsed': NNSED,
        'mnmf': MNMF,
        #'cluster_one': ClusterONE,
    }
    
    def __init__(self, config_path: str | None = None):
        """Initialize processor with optional config file."""
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: str | None) -> dict:
        """Load configuration from YAML file or use defaults."""
        default_config = OmegaConf.create({
            'colorclass_prefix': 'colorclass',
            'dry_run': False,
            'backup_originals': True,
            'community_detection': {
                'algorithm': 'louvain',  # Default algorithm
                'algorithm_params': {    # Parameters passed to the algorithm
                    'seed': 42,
                    'resolution': 1.0,   # For louvain/leiden
                    'threshold': 1e-07,  # For leiden
                    'max_comm_size': 0,  # For leiden (0 = no limit)
                },
                'min_community_size': 2,  # Minimum size for a community to get colorclass
                'naming_scheme': 'cluster_id',  # 'cluster_id', 'largest_node', or 'sequential'
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
        return list(self.NETWORKX_ALGORITHMS.keys()) + list(self.KARATECLUB_ALGORITHMS.keys())
    
    def process_vault(
        self,
        vault_path: str,
        dry_run: bool | None = None,
        algorithm: str | None = None
    ) -> dict[str, str]:
        """Process vault to add colorclass tags using community detection.
        
        Args:
            vault_path: Path to Obsidian vault directory
            dry_run: If True, show what would be changed without modifying files
            algorithm: Community detection algorithm to use (overrides config)
            
        Returns:
            Dictionary mapping article names to their assigned colorclass tags
        """
        vault_path = Path(vault_path)
        dry_run = dry_run if dry_run is not None else self.config.dry_run
        algorithm = algorithm or self.config.community_detection.algorithm
        
        all_algorithms = list(self.NETWORKX_ALGORITHMS.keys()) + list(self.KARATECLUB_ALGORITHMS.keys())
        if algorithm not in all_algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {all_algorithms}")
        
        logger.info(f"Processing vault: {vault_path}")
        logger.info(f"Algorithm: {algorithm}")
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
        
        # Run community detection on all documents
        assignments = self._detect_communities(corpus, algorithm)
        
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
        
        # Convert to undirected for algorithms
        undirected_graph = subgraph.to_undirected()
        
        # Run algorithm based on type
        if algorithm in self.NETWORKX_ALGORITHMS:
            communities = self._run_networkx_algorithm(undirected_graph, algorithm)
        else:
            communities = self._run_karateclub_algorithm(undirected_graph, algorithm)
        
        if not communities:
            logger.error("Community detection failed to produce results")
            return {}
        
        # Process community assignments
        assignments = self._process_communities(communities, undirected_graph)
        
        return assignments
    
    def _run_networkx_algorithm(self, graph: nx.Graph, algorithm: str) -> list[set]:
        """Run NetworkX community detection algorithm."""
        try:
            params = dict(self.config.community_detection.algorithm_params)
            
            if algorithm == 'louvain':
                # NetworkX louvain_communities parameters
                nx_params = {}
                if 'seed' in params:
                    nx_params['seed'] = params['seed']
                if 'resolution' in params:
                    nx_params['resolution'] = params['resolution']
                
                logger.info(f"Running NetworkX Louvain with parameters: {nx_params}")
                communities = nx.community.louvain_communities(graph, **nx_params)
                
            elif algorithm == 'leiden':
                # Check if Leiden is available in this NetworkX version
                if hasattr(nx.community, 'leiden_communities'):
                    nx_params = {}
                    if 'seed' in params:
                        nx_params['seed'] = params['seed']
                    if 'resolution' in params:
                        nx_params['resolution'] = params['resolution']
                    if 'threshold' in params:
                        nx_params['threshold'] = params['threshold']
                    if 'max_comm_size' in params and params['max_comm_size'] > 0:
                        nx_params['max_comm_size'] = params['max_comm_size']
                    
                    logger.info(f"Running NetworkX Leiden with parameters: {nx_params}")
                    communities = nx.community.leiden_communities(graph, **nx_params)
                else:
                    logger.error("Leiden algorithm not available in this NetworkX version")
                    return []
            
            logger.info(f"NetworkX {algorithm} found {len(communities)} communities")
            return list(communities)
            
        except Exception as e:
            logger.error(f"NetworkX {algorithm} failed: {e}")
            return []
    
    def _run_karateclub_algorithm(self, graph: nx.Graph, algorithm: str) -> list[list]:
        """Run karateclub community detection algorithm."""
        try:
            # Create mapping from node names to integer IDs
            node_to_id = {node: i for i, node in enumerate(graph.nodes)}
            id_to_node = {i: node for node, i in node_to_id.items()}
            
            # Convert NetworkX graph to integer-indexed graph
            edges = [(node_to_id[u], node_to_id[v]) for u, v in graph.edges]
            indexed_graph = nx.Graph()
            indexed_graph.add_nodes_from(range(len(node_to_id)))
            indexed_graph.add_edges_from(edges)
            
            # Initialize and run the selected algorithm
            algorithm_class = self.KARATECLUB_ALGORITHMS[algorithm]
            params = dict(self.config.community_detection.algorithm_params)
            
            # Filter parameters that the algorithm actually accepts
            import inspect
            algorithm_signature = inspect.signature(algorithm_class.__init__)
            valid_params = {
                key: value for key, value in params.items() 
                if key in algorithm_signature.parameters
            }
            
            logger.info(f"Running karateclub {algorithm} with parameters: {valid_params}")
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
            
            # Convert back to node names and group by community
            communities_dict = {}
            for node_id, community_id in memberships.items():
                if community_id not in communities_dict:
                    communities_dict[community_id] = []
                communities_dict[community_id].append(id_to_node[node_id])
            
            communities = list(communities_dict.values())
            logger.info(f"Karateclub {algorithm} found {len(communities)} communities")
            return communities
            
        except Exception as e:
            logger.error(f"Karateclub {algorithm} failed: {e}")
            return []
    
    def _process_communities(self, communities: list, graph: nx.Graph) -> dict[str, str]:
        """Process communities into colorclass assignments."""
        # Filter communities by minimum size
        min_size = self.config.community_detection.min_community_size
        filtered_communities = [
            community for community in communities 
            if len(community) >= min_size
        ]
        
        logger.info(f"Found {len(communities)} communities, {len(filtered_communities)} after size filtering")
        
        # Generate colorclass assignments
        assignments = {}
        naming_scheme = self.config.community_detection.naming_scheme
        
        for i, community in enumerate(filtered_communities):
            # Convert community to list if it's a set (from NetworkX)
            nodes = list(community)
            
            if naming_scheme == 'cluster_id':
                # Use community index as colorclass name
                colorclass_tag = f"{self.config.colorclass_prefix}/cluster_{i}"
                
            elif naming_scheme == 'largest_node':
                # Use the node with highest degree as colorclass name
                max_degree = -1
                representative_node = nodes[0]
                for node in nodes:
                    degree = graph.degree(node)
                    if degree > max_degree:
                        max_degree = degree
                        representative_node = node
                
                colorclass_tag = f"{self.config.colorclass_prefix}/{representative_node}"
                
            elif naming_scheme == 'sequential':
                # Use sequential numbering
                colorclass_tag = f"{self.config.colorclass_prefix}/community_{i+1}"
            
            else:
                raise ValueError(f"Unknown naming scheme: {naming_scheme}")
            
            # Assign colorclass to all nodes in community
            for node in nodes:
                assignments[node] = colorclass_tag
            
            logger.info(f"Community {i} ({len(nodes)} nodes) → {colorclass_tag}")
            
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
        
        all_algorithms = list(self.NETWORKX_ALGORITHMS.keys()) + list(self.KARATECLUB_ALGORITHMS.keys())
        if algorithm not in all_algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        logger.info(f"Analyzing community structure with algorithm: {algorithm}")
        
        corpus = load_corpus(vault_path)
        graph = build_graph(corpus)
        
        # Filter to existing documents
        existing_nodes = [doc.node_name for doc in corpus if doc.node_name in graph.nodes]
        subgraph = graph.subgraph(existing_nodes).to_undirected()
        
        if len(subgraph.nodes) < 2:
            return {'error': 'Graph too small for analysis'}
        
        # Run community detection
        try:
            if algorithm in self.NETWORKX_ALGORITHMS:
                communities = self._run_networkx_algorithm(subgraph, algorithm)
            else:
                communities = self._run_karateclub_algorithm(subgraph, algorithm)
                
            if not communities:
                return {'error': 'Community detection failed'}
                
        except Exception as e:
            return {'error': f'Community detection failed: {e}'}
        
        # Calculate statistics
        community_sizes = [len(community) for community in communities]
        min_size = self.config.community_detection.min_community_size
        valid_communities = [community for community in communities if len(community) >= min_size]
        
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
            'coverage': len([n for community in valid_communities for n in community]) / len(existing_nodes) if existing_nodes else 0,
            'algorithm': algorithm,
            'algorithm_type': 'networkx' if algorithm in self.NETWORKX_ALGORITHMS else 'karateclub'
        }
        
        logger.info(f"Community analysis: {analysis}")
        return analysis


def main():
    """CLI entry point for colorclass processor."""
    logger.add("colorclass_processor.log", rotation="1 MB")
    fire.Fire(ColorclassProcessor)


if __name__ == "__main__":
    main()
