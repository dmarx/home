# obsidian/parser.py - Core document parsing utilities

import re
import yaml
from pathlib import Path


def read_yaml(txt: str) -> dict:
    """Parse YAML text and return as dictionary."""
    return yaml.load(txt, yaml.Loader)


def extract_frontmatter(doc: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and body from markdown document."""
    front, body = {}, doc
    if doc.startswith('---'):
        front, body = doc.split("---", 2)[1:]
        front = read_yaml(front)
    return front, body


def get_wikilinks(text: str) -> list[str]:
    """Extract all wikilinks from text using [[link]] pattern."""
    links_pat = re.compile(r"\[\[(.+?)\]\]")
    return re.findall(links_pat, text)


def clean_links(wikilinks: list[str], collect_aliases: bool = False) -> list[str]:
    """Canonicalize aliases, standardize case."""
    if collect_aliases:
        raise NotImplementedError
    outv = []
    for link in wikilinks:
        if '|' in link:
            try:
                link, alias = link.split('|')
            except:
                print(link)
                raise
        outv.append(link.lower())
    return outv


class ObsDoc:
    """Represents a single Obsidian document."""
    
    def __init__(self, title: str, raw: str, fpath: Path | str | None = None):
        self.title = title
        self.raw = raw
        self.frontmatter, self.body = extract_frontmatter(raw)
        if 'title' in self.frontmatter:
            self.title = self.frontmatter['title']
        self.links = clean_links(get_wikilinks(self.body))
        self.tags = self.frontmatter.get('tags',[])
        self.fpath=fpath

    @property
    def node_name(self) -> str:
        """Canonicalized title for graph nodes."""
        return self.title.lower()

    @classmethod
    def from_path(cls, fpath: Path | str) -> 'ObsDoc':
        """Create ObsDoc from file path."""
        fpath = Path(fpath)
        with fpath.open() as f:
            try:
                return cls(fpath.stem, f.read(), fpath=fpath)
            except Exception as e:
                print(fpath)
                raise e
