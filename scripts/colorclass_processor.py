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

from karateclub import LabelPropagation

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from obsidian import ObsDoc, build_graph, load_corpus


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
                'propagation_strength': 0.7  # How much weight to give to neighbor labels
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
        if enable_propagation and HAS_KARATECLUB:
            propagated_assignments = self._propagate_labels(corpus, seed_assignments)
            all_assignments.update(propagated_assignments)
            logger.info(f"Label propagation added {len(propagated_assignments)} new assignments")
        elif enable_propagation and not HAS_KARATECLUB:
            logger.warning("Label propagation requested but karateclub not available")
        
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
        """Use label propagation to spread colorclass tags to connected nodes."""
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
        
        # Convert NetworkX graph to integer-indexed graph for karateclub
        edges = [(node_to_id[u], node_to_id[v]) for u, v in undirected_graph.edges]
        karate_graph = nx.Graph()
        karate_graph.add_nodes_from(range(len(node_to_id)))
        karate_graph.add_edges_from(edges)
        
        # Prepare initial labels (-1 for unlabeled, unique IDs for labeled)
        initial_labels = np.full(len(node_to_id), -1, dtype=int)
        label_to_colorclass = {}
        next_label_id = 0
        
        # Assign labels to seed nodes
        for node_name, colorclass_tag in seed_assignments.items():
            if node_name in node_to_id:
                initial_labels[node_to_id[node_name]] = next_label_id
                label_to_colorclass[next_label_id] = colorclass_tag
                next_label_id += 1
        
        # Run label propagation
        try:
            model = LabelPropagation()
            model.fit(karate_graph, initial_labels)
            predicted_labels = model.get_memberships()
            
            logger.info(f"Label propagation completed with {len(set(predicted_labels))} communities")
            
        except Exception as e:
            logger.error(f"Label propagation failed: {e}")
            return {}
        
        # Extract propagated assignments
        propagated_assignments = {}
        min_connections = self.config.label_propagation.min_connections
        
        for node_id, predicted_label in enumerate(predicted_labels):
            node_name = id_to_node[node_id]
            
            # Skip if already has seed assignment
            if node_name in seed_assignments:
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
        file_path = vault_path / f"{doc.title}.md"
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        # Check if colorclass tag already exists
        existing_colorclass = None
        if doc.tags:
            for tag in doc.tags:
                if tag.startswith(f"{self.config.colorclass_prefix}/"):
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
                if not tag.startswith(f"{self.config.colorclass_prefix}/")
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
