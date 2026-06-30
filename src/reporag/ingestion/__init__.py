"""Ingestion module.

Handles repository cloning, source file parsing, and code entity extraction.
"""

from src.reporag.ingestion.chunker import CodeChunk, SemanticChunker
from src.reporag.ingestion.parser import ASTNode, ASTParser
from src.reporag.ingestion.symbol_extractor import Symbol, SymbolExtractor

__all__ = [
    "ASTNode",
    "ASTParser",
    "CodeChunk",
    "SemanticChunker",
    "Symbol",
    "SymbolExtractor",
]
