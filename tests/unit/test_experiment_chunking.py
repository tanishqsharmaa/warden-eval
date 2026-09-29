from warden_eval.experiments.chunking_benchmark import (
    ChunkingExperimentResult,
    run_chunking_experiment,
)
from warden_eval.schema import GoldenEvalItem, RoleEnum


def test_chunking_experiment_dry_run(tmp_path):
    report_file = tmp_path / "experiment_2_chunking_comparison.md"

    items = [
        GoldenEvalItem(
            eval_id="EVAL-001",
            query="Bereavement leave days?",
            caller_role=RoleEnum.EMPLOYEE,
            ground_truth_answer="5 days",
            ground_truth_doc_ids=["DOC-HR-LEAVE-2026"],
            ground_truth_chunks=[2],
        ),
        GoldenEvalItem(
            eval_id="EVAL-002",
            query="PTO annual accrual?",
            caller_role=RoleEnum.EMPLOYEE,
            ground_truth_answer="18 days",
            ground_truth_doc_ids=["DOC-HR-LEAVE-2026"],
            ground_truth_chunks=[1],
        ),
    ]

    result: ChunkingExperimentResult = run_chunking_experiment(
        items=items,
        report_path=report_file,
        dry_run=True,
    )

    assert result.recursive_precision > 0.0
    assert result.recursive_recall > 0.0
    assert result.semantic_precision > 0.0
    assert result.semantic_recall > 0.0
    assert report_file.exists()

    content = report_file.read_text(encoding="utf-8")
    assert "Experiment 2: Recursive vs. Semantic Chunking" in content
    assert "Recursive 512-Token" in content
    assert "Semantic Distance Chunking" in content
