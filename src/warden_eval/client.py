from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from warden_eval.schema import RoleEnum


class CitationItem(BaseModel):
    citation_id: Optional[int] = None
    doc_id: str
    chunk_index: int
    source_url: str = ""
    quoted_snippet: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    caller_role: str
    answer: str
    citations: List[CitationItem] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    trace_id: Optional[str] = None


class WardenAPIClient:
    def __init__(self, base_url: str = "http://localhost:8080", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def query(self, query_text: str, role: RoleEnum) -> QueryResponse:
        url = f"{self.base_url}/query"
        headers = {
            "Content-Type": "application/json",
            "X-User-Role": role.value if isinstance(role, RoleEnum) else str(role),
        }
        payload = {"query": query_text, "stream": False}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return QueryResponse.model_validate(data)
