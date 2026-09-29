from pathlib import Path
from unittest.mock import patch

import pytest

from warden_eval.client import QueryResponse
from warden_eval.runner import run_evaluation_pipeline
from warden_eval.schema import GoldenEvalItem, RoleEnum


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ci_synthetic_failure_gate_blocks_merge(tmp_path):
    """
    Asserts that if Faithfulness regresses below the 0.90 threshold,
    the pipeline returns exit code 1 and logs the failure in the scorecard.
    """
    items = [
        GoldenEvalItem(
            eval_id="EVAL-001",
            query="Bereavement leave policy question?",
            caller_role=RoleEnum.EMPLOYEE,
            ground_truth_answer="Eligible for 5 days.",
            ground_truth_doc_ids=["DOC-HR-LEAVE-2026"],
            ground_truth_chunks=[2],
        )
    ]
    output_dir = tmp_path / "ci_reports"

    # Injecting synthetic hallucination regression (Faithfulness drops to 0.72)
    synthetic_failing_scores = {
        "faithfulness": 0.72,  # FAILS (Threshold: 0.90)
        "answer_relevancy": 0.91,
        "context_precision": 0.88,
        "context_recall": 0.84,
    }

    with patch("warden_eval.runner.load_dataset", return_value=items), \
         patch("warden_eval.client.WardenAPIClient.query") as mock_query, \
         patch("warden_eval.ragas_gates.RagasGateEvaluator.compute_metrics", return_value=synthetic_failing_scores):

        mock_query.return_value = QueryResponse(
            query="Bereavement leave policy question?",
            caller_role="Employee",
            answer="Hallucinated answer without factual backing.",
            citations=[],
            metrics={},
            trace_id="ci-trace-id",
        )

        exit_code, scorecard = await run_evaluation_pipeline(
            dataset_path="mock.json",
            base_url="http://localhost:8080",
            output_dir=output_dir,
            enforce_gates=True,
        )

        # 1. Exit code MUST be 1 to block CI pipeline
        assert exit_code == 1, f"Expected exit code 1 for failing gate, got {exit_code}"

        # 2. Scorecard must declare failure and list faithfulness
        assert scorecard["passed"] is False
        assert "faithfulness" in scorecard["failing_metrics"]

        # 3. Artifacts must be written
        json_artifact = output_dir / "ragas_scorecard.json"
        md_artifact = output_dir / "ragas_scorecard.md"
        assert json_artifact.exists()
        assert md_artifact.exists()

        md_content = md_artifact.read_text(encoding="utf-8")
        assert "FAILED: RAGAS quality gates regressed" in md_content
        assert "Pull Request blocked from deployment" in md_content


@pytest.mark.integration
def test_workflow_file_conforms_to_spec():
    workflow_path = Path(__file__).parents[2] / ".github" / "workflows" / "ragas-gate.yml"
    assert workflow_path.exists(), f"Workflow file missing at {workflow_path}"

    content = workflow_path.read_text(encoding="utf-8")
    assert "name: RAGAS Quality Gate CI" in content
    assert "eval_golden_50.json" in content
    assert "--enforce-gates" in content
    assert "docker compose" in content
