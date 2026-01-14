"""
Document Indexer for ML Study Notes

Creates a BM25 index of all markdown documentation for offline search.
"""

import os
import re
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("Please install rank-bm25: pip install rank-bm25")
    raise


@dataclass
class Document:
    """Represents an indexed document section."""
    path: str
    title: str
    section: str
    content: str
    language: str  # 'en' or 'zh'

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'Document':
        return cls(**d)


class DocumentIndex:
    """BM25-based document index for offline search."""

    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = Path(docs_dir)
        self.documents: List[Document] = []
        self.bm25: Optional[BM25Okapi] = None
        self.index_path = Path(".claude/index.pkl")

    def _extract_sections(self, content: str, filepath: Path) -> List[Tuple[str, str]]:
        """Extract sections from markdown content."""
        sections = []
        current_section = "Introduction"
        current_content = []

        for line in content.split('\n'):
            if line.startswith('## '):
                if current_content:
                    sections.append((current_section, '\n'.join(current_content)))
                current_section = line[3:].strip()
                current_content = []
            elif line.startswith('# '):
                if current_content:
                    sections.append((current_section, '\n'.join(current_content)))
                current_section = line[2:].strip()
                current_content = []
            else:
                current_content.append(line)

        if current_content:
            sections.append((current_section, '\n'.join(current_content)))

        return sections

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Remove LaTeX
        text = re.sub(r'\$[^$]+\$', ' MATH ', text)
        text = re.sub(r'\\\([^)]+\\\)', ' MATH ', text)
        text = re.sub(r'\\\[[^\]]+\\\]', ' MATH ', text)

        # Remove code blocks
        text = re.sub(r'```[^`]+```', ' CODE ', text)
        text = re.sub(r'`[^`]+`', ' CODE ', text)

        # Remove markdown formatting
        text = re.sub(r'[#*_\[\](){}|<>]', ' ', text)

        # Tokenize
        tokens = re.findall(r'\b\w+\b', text.lower())

        return tokens

    def build_index(self) -> int:
        """Build index from all markdown files in docs directory."""
        self.documents = []

        for lang in ['en', 'zh']:
            lang_dir = self.docs_dir / lang
            if not lang_dir.exists():
                continue

            for md_file in lang_dir.rglob('*.md'):
                try:
                    content = md_file.read_text(encoding='utf-8')
                    relative_path = str(md_file.relative_to(self.docs_dir))

                    # Get title from first heading
                    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
                    title = title_match.group(1) if title_match else md_file.stem

                    # Extract sections
                    sections = self._extract_sections(content, md_file)

                    for section_title, section_content in sections:
                        if len(section_content.strip()) > 50:  # Skip very short sections
                            doc = Document(
                                path=relative_path,
                                title=title,
                                section=section_title,
                                content=section_content,
                                language=lang
                            )
                            self.documents.append(doc)

                except Exception as e:
                    print(f"Error processing {md_file}: {e}")

        # Build BM25 index
        tokenized_docs = [self._tokenize(doc.content) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        # Save index
        self._save_index()

        return len(self.documents)

    def _save_index(self):
        """Save index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'documents': [doc.to_dict() for doc in self.documents],
            'bm25': self.bm25
        }

        with open(self.index_path, 'wb') as f:
            pickle.dump(data, f)

    def load_index(self) -> bool:
        """Load index from disk."""
        if not self.index_path.exists():
            return False

        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)

            self.documents = [Document.from_dict(d) for d in data['documents']]
            self.bm25 = data['bm25']
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False

    def search(self, query: str, top_k: int = 5, language: Optional[str] = None) -> List[Tuple[Document, float]]:
        """Search for documents matching query."""
        if self.bm25 is None:
            if not self.load_index():
                self.build_index()

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top results
        results = []
        for idx, score in enumerate(scores):
            doc = self.documents[idx]
            if language and doc.language != language:
                continue
            results.append((doc, score))

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]


def main():
    """CLI for building the index."""
    import argparse

    parser = argparse.ArgumentParser(description="Build document index for Q&A system")
    parser.add_argument("--docs-dir", default="docs", help="Documentation directory")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild index")
    args = parser.parse_args()

    index = DocumentIndex(args.docs_dir)

    if args.rebuild or not index.load_index():
        print("Building index...")
        count = index.build_index()
        print(f"Indexed {count} document sections")
    else:
        print(f"Loaded existing index with {len(index.documents)} sections")


if __name__ == "__main__":
    main()
