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

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from obsidian import ObsDoc, build_graph, load_corpus


class ColorclassProcessor:
    """Processes Obsidian vault to add unique colorclass tags with NetworkX community detection."""
    
    # Available NetworkX community detection algorithms
    AVAILABLE_ALGORITHMS = {
        'louvain': 'louvain_communities',
        'leiden': 'leiden_communities', 
        'greedy_modularity': 'greedy_modularity_communities',
        'girvan_newman': 'girvan_newman',
        'label_propagation': 'asyn_lpa_communities',
        'kernighan_lin': 'kernighan_lin_bisection',
    }
    
    def __init__(self, config_path: str | None = None):
        """Initialize processor with optional config file."""
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: str | None) -> dict:
        """Load configuration from YAML file or use defaults."""
        default_config = OmegaConf.create({
            'colorclass_prefix': 'colorclass',
            'dry_run': False,
            'backup_originals': False,
            'community_detection': {
                'algorithm': 'louvain',  # Default algorithm
                'algorithm_params': {    # Parameters passed to the algorithm
                    'seed': 42,
                    'resolution': 1.0,   # For louvain/leiden
                    'threshold': 1e-07,  # For leiden
                    'max_comm_size': 0,  # For leiden (0 = no limit)
                    'weight': None,      # Edge weight attribute name
                    'max_levels': None,  # For girvan_newman (None = all levels)
                },
                'min_community_size': 5,  # Minimum size for a community to get colorclass
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
        available = []
        for algo_name, nx_func_name in self.AVAILABLE_ALGORITHMS.items():
            if hasattr(nx.community, nx_func_name):
                available.append(algo_name)
            else:
                logger.debug(f"Algorithm {algo_name} ({nx_func_name}) not available in this NetworkX version")
        return available
    
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
        
        available_algorithms = self.list_algorithms()
        if algorithm not in available_algorithms:
            raise ValueError(f"Unknown or unavailable algorithm: {algorithm}. Available: {available_algorithms}")
        
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
        """Use NetworkX community detection to assign colorclass tags."""
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
        
        # Run the selected algorithm
        communities = self._run_networkx_algorithm(undirected_graph, algorithm)
        
        if not communities:
            logger.error("Community detection failed to produce results")
            return {}
        
        # Process community assignments
        assignments = self._process_communities(communities, undirected_graph)
        
        return assignments
    
    def _run_networkx_algorithm(self, graph: nx.Graph, algorithm: str) -> list:
        """Run NetworkX community detection algorithm."""
        try:
            params = dict(self.config.community_detection.algorithm_params)
            nx_func_name = self.AVAILABLE_ALGORITHMS[algorithm]
            
            if not hasattr(nx.algorithms.community, nx_func_name):
                logger.error(f"Algorithm {algorithm} ({nx_func_name}) not available in NetworkX")
                return []
            
            func = getattr(nx.algorithms.community, nx_func_name)
            
            # Prepare parameters based on algorithm
            if algorithm == 'louvain':
                nx_params = {}
                if 'seed' in params and params['seed'] is not None:
                    nx_params['seed'] = params['seed']
                if 'resolution' in params and params['resolution'] is not None:
                    nx_params['resolution'] = params['resolution']
                if 'weight' in params and params['weight'] is not None:
                    nx_params['weight'] = params['weight']
                
                logger.info(f"Running NetworkX Louvain with parameters: {nx_params}")
                communities = func(graph, **nx_params)
                
            elif algorithm == 'leiden':
                nx_params = {}
                if 'seed' in params and params['seed'] is not None:
                    nx_params['seed'] = params['seed']
                if 'resolution' in params and params['resolution'] is not None:
                    nx_params['resolution'] = params['resolution']
                if 'threshold' in params and params['threshold'] is not None:
                    nx_params['threshold'] = params['threshold']
                if 'max_comm_size' in params and params['max_comm_size'] is not None and params['max_comm_size'] > 0:
                    nx_params['max_comm_size'] = params['max_comm_size']
                
                logger.info(f"Running NetworkX Leiden with parameters: {nx_params}")
                communities = func(graph, **nx_params)
                
            elif algorithm == 'greedy_modularity':
                nx_params = {}
                if 'weight' in params and params['weight'] is not None:
                    nx_params['weight'] = params['weight']
                if 'resolution' in params and params['resolution'] is not None:
                    nx_params['resolution'] = params['resolution']
                
                logger.info(f"Running NetworkX Greedy Modularity with parameters: {nx_params}")
                communities = func(graph, **nx_params)
                
            elif algorithm == 'girvan_newman':
                nx_params = {}
                if 'weight' in params and params['weight'] is not None:
                    nx_params['weight'] = params['weight']
                
                logger.info(f"Running NetworkX Girvan-Newman with parameters: {nx_params}")
                # Girvan-Newman returns a generator of community divisions
                communities_gen = func(graph, **nx_params)
                
                # Get the best division (or up to max_levels)
                max_levels = params.get('max_levels', 10)  # Default to 10 levels
                if max_levels is None:
                    max_levels = 10
                
                best_communities = None
                best_modularity = -1
                
                for i, division in enumerate(communities_gen):
                    if i >= max_levels:
                        break
                    modularity = nx.algorithms.community.modularity(graph, division)
                    if modularity > best_modularity:
                        best_modularity = modularity
                        best_communities = division
                
                communities = best_communities if best_communities else []
                logger.info(f"Girvan-Newman best modularity: {best_modularity}")
                
            elif algorithm == 'label_propagation':
                nx_params = {}
                if 'seed' in params and params['seed'] is not None:
                    nx_params['seed'] = params['seed']
                if 'weight' in params and params['weight'] is not None:
                    nx_params['weight'] = params['weight']
                
                logger.info(f"Running NetworkX Label Propagation with parameters: {nx_params}")
                communities = func(graph, **nx_params)
                
            elif algorithm == 'kernighan_lin':
                # This is a bisection algorithm, so we'll apply it recursively
                logger.info("Running NetworkX Kernighan-Lin (recursive bisection)")
                communities = self._recursive_kernighan_lin(graph, params)
                
            else:
                logger.error(f"Unknown algorithm implementation: {algorithm}")
                return []
            
            if communities:
                communities_list = list(communities)
                logger.info(f"NetworkX {algorithm} found {len(communities_list)} communities")
                return communities_list
            else:
                logger.warning(f"NetworkX {algorithm} returned no communities")
                return []
                
        except Exception as e:
            logger.error(f"NetworkX {algorithm} failed: {e}")
            return []
    
    def _recursive_kernighan_lin(self, graph: nx.Graph, params: dict, max_depth: int = 4) -> list:
        """Apply Kernighan-Lin bisection recursively to create multiple communities."""
        communities = []
        
        def bisect_graph(g, depth=0):
            if len(g.nodes) < 4 or depth >= max_depth:  # Stop if too small or too deep
                communities.append(set(g.nodes))
                return
            
            try:
                # Apply Kernighan-Lin bisection
                partition = nx.algorithms.community.kernighan_lin_bisection(g, seed=params.get('seed'))
                
                # If partition is successful and creates meaningful split
                if len(partition[0]) > 1 and len(partition[1]) > 1:
                    # Recursively bisect each partition
                    subgraph1 = g.subgraph(partition[0]).copy()
                    subgraph2 = g.subgraph(partition[1]).copy()
                    bisect_graph(subgraph1, depth + 1)
                    bisect_graph(subgraph2, depth + 1)
                else:
                    # Can't split meaningfully, add as single community
                    communities.append(set(g.nodes))
            except:
                # If bisection fails, add as single community
                communities.append(set(g.nodes))
        
        bisect_graph(graph)
        return communities
    
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
            # Convert community to list if it's a set
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
        
        available_algorithms = self.list_algorithms()
        if algorithm not in available_algorithms:
            raise ValueError(f"Unknown or unavailable algorithm: {algorithm}")
        
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
            communities = self._run_networkx_algorithm(subgraph, algorithm)
                
            if not communities:
                return {'error': 'Community detection failed'}
                
        except Exception as e:
            return {'error': f'Community detection failed: {e}'}
        
        # Calculate statistics
        community_sizes = [len(community) for community in communities]
        min_size = self.config.community_detection.min_community_size
        valid_communities = [community for community in communities if len(community) >= min_size]
        
        # Calculate modularity for the detected communities
        try:
            modularity = nx.algorithms.community.modularity(subgraph, communities)
        except:
            modularity = None
        
        analysis = {
            'total_documents': len(corpus),
            'clustered_documents': len(existing_nodes),
            'total_communities': len(communities),
            'valid_communities': len(valid_communities),
            'modularity': modularity,
            'community_size_stats': {
                'min': min(community_sizes) if community_sizes else 0,
                'max': max(community_sizes) if community_sizes else 0,
                'mean': sum(community_sizes) / len(community_sizes) if community_sizes else 0,
                'sizes': sorted(community_sizes, reverse=True)[:10]  # Top 10 sizes
            },
            'coverage': len([n for community in valid_communities for n in community]) / len(existing_nodes) if existing_nodes else 0,
            'algorithm': algorithm
        }
        
        logger.info(f"Community analysis: {analysis}")
        return analysis


def main():
    """CLI entry point for colorclass processor."""
    logger.add("colorclass_processor.log", rotation="1 MB")
    fire.Fire(ColorclassProcessor)


if __name__ == "__main__":
    main()
