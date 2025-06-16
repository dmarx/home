# scripts/colorclass_processor.py - Add unique colorclass tags with label propagation

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

from karateclub import LabelPropagation

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from obsidian import ObsDoc, build_graph, load_corpus



class SemiSupervisedLabelPropagation:
    """Modified Label Propagation that supports initial seed labels."""
    
    def __init__(self, seed: int = 42, iterations: int = 100):
        self.seed = seed
        self.iterations = iterations
        
    def _set_seed(self):
        """Set random seed for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        
    def _make_a_pick(self, neighbors, unlabeled_value=-1):
        """Choose a label from neighboring nodes, ignoring unlabeled ones."""
        scores = {}
        for neighbor in neighbors:
            neighbor_label = self._labels[neighbor]
            # Skip unlabeled neighbors
            if neighbor_label == unlabeled_value:
                continue
            if neighbor_label in scores:
                scores[neighbor_label] += 1
            else:
                scores[neighbor_label] = 1
        
        # If no labeled neighbors, return unlabeled
        if not scores:
            return unlabeled_value
            
        # Return most common label (random tie-breaking)
        max_score = max(scores.values())
        top_labels = [label for label, score in scores.items() if score == max_score]
        return random.choice(top_labels)
    
    def _do_a_propagation(self, unlabeled_value=-1):
        """Do one round of label propagation, only updating unlabeled nodes."""
        # Shuffle nodes for randomness
        nodes_to_update = [node for node in self._nodes if self._labels[node] == unlabeled_value]
        random.shuffle(nodes_to_update)
        
        new_labels = self._labels.copy()
        
        for node in nodes_to_update:
            neighbors = list(nx.neighbors(self._graph, node))
            if neighbors:  # Only update if node has neighbors
                pick = self._make_a_pick(neighbors, unlabeled_value)
                new_labels[node] = pick
                
        self._labels = new_labels
    
    def fit(self, graph: nx.Graph, initial_labels: dict[int, int] | None = None):
        """Fit label propagation with optional initial seed labels.
        
        Args:
            graph: NetworkX graph to cluster
            initial_labels: Dict mapping node IDs to initial labels.
                           Nodes not in dict are considered unlabeled.
        """
        self._set_seed()
        self._graph = graph
        self._nodes = list(self._graph.nodes())
        
        # Initialize labels
        unlabeled_value = -1
        if initial_labels:
            self._labels = {node: initial_labels.get(node, unlabeled_value) 
                          for node in self._nodes}
        else:
            # Original behavior: all nodes get unique labels
            self._labels = {node: i for i, node in enumerate(self._nodes)}
        
        # Run propagation iterations
        for iteration in range(self.iterations):
            old_labels = self._labels.copy()
            self._do_a_propagation(unlabeled_value)
            
            # Check for convergence (no changes in unlabeled nodes)
            changes = sum(1 for node in self._nodes 
                         if old_labels[node] != self._labels[node] 
                         and old_labels[node] == unlabeled_value)
            
            if changes == 0:
                logger.debug(f"Label propagation converged after {iteration + 1} iterations")
                break
                
    def get_memberships(self) -> dict[int, int]:
        """Get cluster membership of nodes."""
        return self._labels.copy()


class ColorclassProcessor:
    """Processes Obsidian vault to add unique colorclass tags with label propagation."""
    
    def __init__(self, config_path: str | None = None):
        """Initialize processor with optional config file."""
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: str | None) -> dict:
        """Load configuration from YAML file or use defaults."""
        default_config = OmegaConf.create({
            'source_tag': 'sod/root',
            'colorclass_prefix': 'colorclass',
            'dry_run': False,
            'backup_originals': True,
            'label_propagation': {
                'enabled': True,
                'max_iterations': 100,
                'min_connections': 2,  # Minimum connections to receive propagated label
                'seed': 42  # Random seed for reproducibility
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
    
    def process_vault(
        self,
        vault_path: str,
        source_tag: str | None = None,
        dry_run: bool | None = None,
        enable_propagation: bool | None = None
    ) -> dict[str, str]:
        """Process vault to add colorclass tags with optional label propagation.
        
        Args:
            vault_path: Path to Obsidian vault directory
            source_tag: Tag to search for (overrides config)
            dry_run: If True, show what would be changed without modifying files
            enable_propagation: Enable label propagation (overrides config)
            
        Returns:
            Dictionary mapping article names to their assigned colorclass tags
        """
        vault_path = Path(vault_path)
        source_tag = source_tag or self.config.source_tag
        dry_run = dry_run if dry_run is not None else self.config.dry_run
        enable_propagation = (enable_propagation if enable_propagation is not None 
                            else self.config.label_propagation.enabled)
        
        logger.info(f"Processing vault: {vault_path}")
        logger.info(f"Source tag: {source_tag}")
        logger.info(f"Label propagation: {enable_propagation}")
        logger.info(f"Dry run: {dry_run}")
        
        # Load all documents
        corpus = load_corpus(vault_path)
        
        # Phase 1: Add colorclass tags to source articles
        seed_assignments = self._add_seed_colorclass_tags(corpus, source_tag)
        
        if not seed_assignments:
            logger.warning("No seed articles found - skipping label propagation")
            return {}
        
        # Phase 2: Label propagation (if enabled)
        all_assignments = seed_assignments.copy()
        if enable_propagation:
            propagated_assignments = self._propagate_labels(corpus, seed_assignments)
            all_assignments.update(propagated_assignments)
            logger.info(f"Label propagation added {len(propagated_assignments)} new assignments")
        
        # Phase 3: Apply changes to files
        if not dry_run:
            modified_count = self._apply_assignments(corpus, vault_path, all_assignments)
            logger.success(f"Modified {modified_count} files")
        else:
            logger.info("Dry run complete - no files modified")
        
        return all_assignments
    
    def _add_seed_colorclass_tags(self, corpus: list[ObsDoc], source_tag: str) -> dict[str, str]:
        """Add colorclass tags to articles with source tag (seed nodes)."""
        target_docs = []
        for doc in corpus:
            if doc.tags and source_tag in doc.tags:
                target_docs.append(doc)
        
        logger.info(f"Found {len(target_docs)} seed documents with tag '{source_tag}'")
        
        # Generate unique colorclass assignments for seed articles
        assignments = {}
        for doc in target_docs:
            colorclass_tag = f"{self.config.colorclass_prefix}/{doc.node_name}"
            assignments[doc.node_name] = colorclass_tag
            logger.info(f"  Seed: {doc.title} → {colorclass_tag}")
        
        return assignments
    
    def _propagate_labels(self, corpus: list[ObsDoc], seed_assignments: dict[str, str]) -> dict[str, str]:
        """Use modified label propagation to spread colorclass tags to connected nodes."""
        logger.info("Starting label propagation...")
        
        # Build graph from corpus
        graph = build_graph(corpus)
        
        # Filter to existing documents only (no phantom nodes)
        existing_nodes = []
        for doc in corpus:
            if doc.node_name in graph.nodes:
                existing_nodes.append(doc.node_name)
        
        subgraph = graph.subgraph(existing_nodes).copy()
        logger.info(f"Propagating on subgraph with {len(subgraph.nodes)} nodes, {len(subgraph.edges)} edges")
        
        if len(subgraph.nodes) < 2:
            logger.warning("Graph too small for label propagation")
            return {}
        
        # Convert to undirected for label propagation
        undirected_graph = subgraph.to_undirected()
        
        # Create mapping from node names to integer IDs
        node_to_id = {node: i for i, node in enumerate(undirected_graph.nodes)}
        id_to_node = {i: node for node, i in node_to_id.items()}
        
        # Convert NetworkX graph to integer-indexed graph
        edges = [(node_to_id[u], node_to_id[v]) for u, v in undirected_graph.edges]
        indexed_graph = nx.Graph()
        indexed_graph.add_nodes_from(range(len(node_to_id)))
        indexed_graph.add_edges_from(edges)
        
        # Prepare initial labels for seed nodes
        initial_labels = {}
        label_to_colorclass = {}
        next_label_id = 0
        
        # Assign integer labels to seed nodes
        for node_name, colorclass_tag in seed_assignments.items():
            if node_name in node_to_id:
                initial_labels[node_to_id[node_name]] = next_label_id
                label_to_colorclass[next_label_id] = colorclass_tag
                next_label_id += 1
        
        # Run label propagation
        try:
            model = SemiSupervisedLabelPropagation(
                seed=self.config.label_propagation.seed,
                iterations=self.config.label_propagation.max_iterations
            )
            model.fit(indexed_graph, initial_labels)
            predicted_labels = model.get_memberships()
            
            logger.info(f"Label propagation completed")
            
        except Exception as e:
            logger.error(f"Label propagation failed: {e}")
            return {}
        
        # Extract propagated assignments
        propagated_assignments = {}
        min_connections = self.config.label_propagation.min_connections
        
        for node_id, predicted_label in predicted_labels.items():
            node_name = id_to_node[node_id]
            
            # Skip if already has seed assignment or is unlabeled
            if node_name in seed_assignments or predicted_label == -1:
                continue
            
            # Skip if predicted label is unknown
            if predicted_label not in label_to_colorclass:
                continue
            
            # Check minimum connections requirement
            node_degree = undirected_graph.degree(node_name)
            if node_degree < min_connections:
                continue
            
            # Check if node already has a colorclass tag
            doc = next((d for d in corpus if d.node_name == node_name), None)
            if doc and doc.tags:
                existing_colorclass = any(tag.startswith(f"{self.config.colorclass_prefix}/") 
                                        for tag in doc.tags)
                if existing_colorclass:
                    continue
            
            propagated_assignments[node_name] = label_to_colorclass[predicted_label]
            logger.info(f"  Propagated: {node_name} → {label_to_colorclass[predicted_label]}")
        
        return propagated_assignments
    
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
        #file_path = vault_path / f"{doc.title}.md"
        
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
    
    def analyze_propagation_potential(self, vault_path: str, source_tag: str | None = None) -> dict:
        """Analyze how many nodes could receive propagated labels."""
        vault_path = Path(vault_path)
        source_tag = source_tag or self.config.source_tag
        
        logger.info(f"Analyzing propagation potential for tag: {source_tag}")
        
        corpus = load_corpus(vault_path)
        graph = build_graph(corpus)
        
        # Find seed nodes
        seed_nodes = set()
        for doc in corpus:
            if doc.tags and source_tag in doc.tags:
                seed_nodes.add(doc.node_name)
        
        # Analyze connectivity
        undirected = graph.to_undirected()
        existing_nodes = {doc.node_name for doc in corpus if doc.node_name in graph.nodes}
        subgraph = undirected.subgraph(existing_nodes)
        
        # Find nodes reachable from seeds
        reachable_nodes = set()
        for seed in seed_nodes:
            if seed in subgraph:
                reachable_nodes.update(nx.single_source_shortest_path_length(subgraph, seed).keys())
        
        # Count potential targets
        potential_targets = reachable_nodes - seed_nodes
        
        # Analyze by degree
        degree_dist = Counter()
        for node in potential_targets:
            degree = subgraph.degree(node)
            degree_dist[degree] += 1
        
        analysis = {
            'total_documents': len(corpus),
            'seed_nodes': len(seed_nodes),
            'reachable_nodes': len(reachable_nodes),
            'potential_targets': len(potential_targets),
            'coverage_ratio': len(reachable_nodes) / len(existing_nodes) if existing_nodes else 0,
            'degree_distribution': dict(degree_dist.most_common(10))
        }
        
        logger.info(f"Propagation analysis: {analysis}")
        return analysis


def main():
    """CLI entry point for colorclass processor."""
    logger.add("colorclass_processor.log", rotation="1 MB")
    fire.Fire(ColorclassProcessor)


if __name__ == "__main__":
    main()
