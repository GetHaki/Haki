import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel


class GraphNode(BaseModel):
    id: str
    kind: Literal["subject", "fact", "event"]
    label: str
    status: str | None = None
    meta: dict[str, Any] = {}


class GraphEdge(BaseModel):
    source: str
    target: str
    kind: Literal["has_fact", "supersedes", "derived_from"]


class GraphResponse(BaseModel):
    subject_id: str
    project_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
