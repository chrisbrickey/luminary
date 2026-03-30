"""Pydantic schemas to validate data structures.

This module re-exports all schemas for backward compatibility.
Schemas are organized by domain in submodules:
- ingestion: WikisourceCollection
- vectorstore: ChunkInfo
- chat: ChatResponse
- eval: MetricResult
"""

from src.schemas.chat import ChatResponse
from src.schemas.eval import MetricResult
from src.schemas.ingestion import WikisourceCollection
from src.schemas.vectorstore import ChunkInfo

__all__ = [
    "ChatResponse",
    "ChunkInfo",
    "MetricResult",
    "WikisourceCollection",
]
