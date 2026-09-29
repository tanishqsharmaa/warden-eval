from warden_eval.experiments.rrf_survival import RrfSurvivalResult, run_rrf_survival_experiment
from warden_eval.schema import GoldenEvalItem, RoleEnum


def test_rrf_survival_experiment_dry_run(tmp_path):
    report_file = tmp_path / "experiment_1_rrf_survival.md"

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

    result: RrfSurvivalResult = run_rrf_survival_experiment(
        items=items,
        report_path=report_file,
        dry_run=True,
    )

    assert result.dense_only_hit_at_5 > 0.0
    assert result.hybrid_rrf_hit_at_5 > 0.0
    assert result.dense_only_mrr_at_5 > 0.0
    assert result.hybrid_rrf_mrr_at_5 > 0.0
    assert report_file.exists()

    content = report_file.read_text(encoding="utf-8")
    assert "Experiment 1: RRF Fusion Survival Post-Truncation" in content
    assert "Dense-Only (BGE-Base)" in content
    assert "Hybrid RRF (BGE + BM25)" in content
