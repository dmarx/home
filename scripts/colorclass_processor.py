# scripts/colorclass_processor.py - Add unique colorclass tags to articles by source tag

from pathlib import Path
from loguru import logger
import yaml
import fire
from omegaconf import OmegaConf
import sys


# install local tools
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))



from obsidian import ObsDoc, load_corpus


class ColorclassProcessor:
    """Processes Obsidian vault to add unique colorclass tags based on source tags."""
    
    def __init__(self, config_path: str | None = None):
        """Initialize processor with optional config file."""
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: str | None) -> dict:
        """Load configuration from YAML file or use defaults."""
        default_config = {
            'source_tag': 'sod/root',
            'colorclass_prefix': 'colorclass',
            'dry_run': False,
            'backup_originals': True
        }
        
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
        dry_run: bool | None = None
    ) -> dict[str, str]:
        """Process vault to add colorclass tags to articles with source tag.
        
        Args:
            vault_path: Path to Obsidian vault directory
            source_tag: Tag to search for (overrides config)
            dry_run: If True, show what would be changed without modifying files
            
        Returns:
            Dictionary mapping article names to their assigned colorclass tags
        """
        vault_path = Path(vault_path)
        source_tag = source_tag or self.config.source_tag
        dry_run = dry_run if dry_run is not None else self.config.dry_run
        
        logger.info(f"Processing vault: {vault_path}")
        logger.info(f"Source tag: {source_tag}")
        logger.info(f"Dry run: {dry_run}")
        
        # Load all documents
        corpus = load_corpus(vault_path)
        
        # Find documents with the source tag
        target_docs = []
        for doc in corpus:
            if doc.tags and source_tag in doc.tags:
                target_docs.append(doc)
        
        logger.info(f"Found {len(target_docs)} documents with tag '{source_tag}'")
        
        if not target_docs:
            logger.warning("No documents found with the specified tag")
            return {}
        
        # Generate colorclass assignments
        assignments = {}
        for doc in target_docs:
            colorclass_tag = f"{self.config.colorclass_prefix}/{doc.node_name}"
            assignments[doc.title] = colorclass_tag
            logger.info(f"  {doc.title} → {colorclass_tag}")
        
        if dry_run:
            logger.info("Dry run complete - no files modified")
            return assignments
        
        # Apply changes to files
        modified_count = 0
        for doc in target_docs:
            if self._add_colorclass_tag(doc, vault_path):
                modified_count += 1
        
        logger.success(f"Modified {modified_count} files")
        return assignments
    
    def _add_colorclass_tag(self, doc: ObsDoc, vault_path: Path) -> bool:
        """Add colorclass tag to a document's frontmatter.
        
        Args:
            doc: ObsDoc instance to modify
            vault_path: Path to vault directory
            
        Returns:
            True if file was modified, False otherwise
        """
        file_path = vault_path / f"{doc.title}.md"
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        # Generate the new colorclass tag
        colorclass_tag = f"{self.config.colorclass_prefix}/{doc.node_name}"
        
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


def main():
    """CLI entry point for colorclass processor."""
    logger.add("colorclass_processor.log", rotation="1 MB")
    fire.Fire(ColorclassProcessor)


if __name__ == "__main__":
    main()
