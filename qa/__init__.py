"""
Q&A System for ML Study Notes

Provides:
- BM25-based document search (index.py)
- Question answering interface (ask.py)
- Content validation (validate.py)
- Patch proposal management (patch.py)
"""

from .index import DocumentIndex, Document
from .validate import DocumentValidator, ValidationError

__all__ = ['DocumentIndex', 'Document', 'DocumentValidator', 'ValidationError']
