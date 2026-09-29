from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class RoleEnum(str, Enum):
    EMPLOYEE = "Employee"
    MANAGER = "Manager"
    HR_ADMIN = "HR-Admin"


class GoldenEvalItem(BaseModel):
    eval_id: str = Field(..., description="Unique evaluation item identifier (e.g., EVAL-001)")
    query: str = Field(..., min_length=5, description="HR policy query string")
    caller_role: RoleEnum = Field(..., description="Target security permission tier")
    ground_truth_answer: str = Field(..., min_length=5, description="Ground truth answer text")
    ground_truth_doc_ids: List[str] = Field(..., min_length=1, description="List of ground truth document IDs")
    ground_truth_chunks: List[int] = Field(..., min_length=1, description="List of ground truth chunk indices")


class GoldenEvalDataset(BaseModel):
    items: List[GoldenEvalItem] = Field(..., min_length=1)
