import json
from pathlib import Path
import pytest
from warden_eval.schema import GoldenEvalDataset, GoldenEvalItem, RoleEnum


def test_schema_model_validation():
    item = GoldenEvalItem(
        eval_id="EVAL-001",
        query="How many days of bereavement leave am I entitled to?",
        caller_role=RoleEnum.EMPLOYEE,
        ground_truth_answer="Employees are eligible for up to 5 consecutive paid days off.",
        ground_truth_doc_ids=["DOC-HR-LEAVE-2026"],
        ground_truth_chunks=[2, 3],
    )
    assert item.eval_id == "EVAL-001"
    assert item.caller_role == RoleEnum.EMPLOYEE
    assert len(item.ground_truth_doc_ids) == 1
    assert item.ground_truth_chunks == [2, 3]


def test_golden_dataset_structure_and_counts():
    dataset_path = Path(__file__).parents[2] / "eval" / "datasets" / "eval_golden_50.json"
    assert dataset_path.exists(), f"Dataset file missing at {dataset_path}"

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 50, f"Expected exactly 50 items, found {len(data)}"

    # Parse via Pydantic model
    validated_items = [GoldenEvalItem.model_validate(item) for item in data]
    dataset = GoldenEvalDataset(items=validated_items)
    assert len(dataset.items) == 50

    role_counts = {
        RoleEnum.EMPLOYEE: 0,
        RoleEnum.MANAGER: 0,
        RoleEnum.HR_ADMIN: 0,
    }
    eval_ids = set()

    for item in dataset.items:
        role_counts[item.caller_role] += 1
        assert item.eval_id not in eval_ids, f"Duplicate eval_id: {item.eval_id}"
        eval_ids.add(item.eval_id)
        assert len(item.query.strip()) > 10
        assert len(item.ground_truth_answer.strip()) > 10
        assert len(item.ground_truth_doc_ids) > 0
        assert len(item.ground_truth_chunks) > 0

    assert role_counts[RoleEnum.EMPLOYEE] == 20, f"Expected 20 Employee queries, got {role_counts[RoleEnum.EMPLOYEE]}"
    assert role_counts[RoleEnum.MANAGER] == 15, f"Expected 15 Manager queries, got {role_counts[RoleEnum.MANAGER]}"
    assert role_counts[RoleEnum.HR_ADMIN] == 15, f"Expected 15 HR-Admin queries, got {role_counts[RoleEnum.HR_ADMIN]}"
