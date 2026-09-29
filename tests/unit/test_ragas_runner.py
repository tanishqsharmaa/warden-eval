from unittest.mock import AsyncMock, patch

import pytest

from warden_eval.client import CitationItem, QueryResponse, WardenAPIClient
from warden_eval.ragas_gates import GATE_THRESHOLDS, RagasGateEvaluator
from warden_eval.runner import run_evaluation_pipeline
from warden_eval.schema import GoldenEvalItem, RoleEnum


def test_gate_thresholds():
    assert GATE_THRESHOLDS["faithfulness"] == 0.90
    assert GATE_THRESHOLDS["answer_relevancy"] == 0.85
    assert GATE_THRESHOLDS["context_precision"] == 0.80
    assert GATE_THRESHOLDS["context_recall"] == 0.80


def test_evaluator_pass_criteria():
    evaluator = RagasGateEvaluator(thresholds=GATE_THRESHOLDS)
    scores = {
        "faithfulness": 0.94,
        "answer_relevancy": 0.89,
        "context_precision": 0.85,
        "context_recall": 0.82,
    }
    result = evaluator.evaluate_scores(scores)
    assert result.passed is True
    assert len(result.failing_metrics) == 0


def test_evaluator_fail_criteria():
    evaluator = RagasGateEvaluator(thresholds=GATE_THRESHOLDS)
    scores = {
        "faithfulness": 0.88,  # Below 0.90
        "answer_relevancy": 0.89,
        "context_precision": 0.85,
        "context_recall": 0.78,  # Below 0.80
    }
    result = evaluator.evaluate_scores(scores)
    assert result.passed is False
    assert "faithfulness" in result.failing_metrics
    assert "context_recall" in result.failing_metrics


@pytest.mark.asyncio
async def test_warden_api_client_query(monkeypatch):
    client = WardenAPIClient(base_url="http://localhost:8080")

    mock_response = {
        "query": "How many days of bereavement leave am I entitled to?",
        "caller_role": "Employee",
        "answer": "Full-time employees receive up to 5 days [Doc: DOC-HR-LEAVE-2026, Chunk: 2].",
        "citations": [
            {
                "citation_id": 1,
                "doc_id": "DOC-HR-LEAVE-2026",
                "chunk_index": 2,
                "source_url": "file:///data/policies/pto_policy_2026.md",
            }
        ],
        "metrics": {"total_latency_ms": 45.2, "cache_hit": False},
        "trace_id": "test-trace-id",
    }

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.json = lambda: mock_response
        mock_resp.raise_for_status = lambda: None
        mock_post.return_value = mock_resp
        res = await client.query("How many days of bereavement leave am I entitled to?", RoleEnum.EMPLOYEE)
        assert res.answer.startswith("Full-time employees")
        assert len(res.citations) == 1
        assert res.citations[0].doc_id == "DOC-HR-LEAVE-2026"
        assert res.trace_id == "test-trace-id"


@pytest.mark.asyncio
async def test_run_pipeline_success(tmp_path):
    # Mock items
    items = [
        GoldenEvalItem(
            eval_id="EVAL-001",
            query="Bereavement leave query?",
            caller_role=RoleEnum.EMPLOYEE,
            ground_truth_answer="5 days paid leave.",
            ground_truth_doc_ids=["DOC-HR-LEAVE-2026"],
            ground_truth_chunks=[2],
        )
    ]

    output_dir = tmp_path / "reports"

    with patch("warden_eval.runner.load_dataset", return_value=items), \
         patch("warden_eval.client.WardenAPIClient.query") as mock_query, \
         patch("warden_eval.ragas_gates.RagasGateEvaluator.compute_metrics") as mock_compute:

        mock_query.return_value = QueryResponse(
            query="Bereavement leave query?",
            caller_role="Employee",
            answer="5 days paid leave.",
            citations=[CitationItem(citation_id=1, doc_id="DOC-HR-LEAVE-2026", chunk_index=2, source_url="")],
            metrics={"total_latency_ms": 25.0, "cache_hit": False},
            trace_id="trace-123"
        )
        mock_compute.return_value = {
            "faithfulness": 0.95,
            "answer_relevancy": 0.92,
            "context_precision": 0.88,
            "context_recall": 0.85,
        }

        exit_code, scorecard = await run_evaluation_pipeline(
            dataset_path="mock.json",
            base_url="http://localhost:8080",
            output_dir=output_dir,
            enforce_gates=True,
        )

        assert exit_code == 0
        assert scorecard["passed"] is True
        assert (output_dir / "ragas_scorecard.json").exists()
        assert (output_dir / "ragas_scorecard.md").exists()


@pytest.mark.asyncio
async def test_run_pipeline_failure_exit_code(tmp_path):
    items = [
        GoldenEvalItem(
            eval_id="EVAL-001",
            query="Bereavement leave query?",
            caller_role=RoleEnum.EMPLOYEE,
            ground_truth_answer="5 days paid leave.",
            ground_truth_doc_ids=["DOC-HR-LEAVE-2026"],
            ground_truth_chunks=[2],
        )
    ]
    output_dir = tmp_path / "reports"

    with patch("warden_eval.runner.load_dataset", return_value=items), \
         patch("warden_eval.client.WardenAPIClient.query") as mock_query, \
         patch("warden_eval.ragas_gates.RagasGateEvaluator.compute_metrics") as mock_compute:

        mock_query.return_value = QueryResponse(
            query="Bereavement leave query?",
            caller_role="Employee",
            answer="Irrelevant answer.",
            citations=[],
            metrics={},
            trace_id="trace-123"
        )
        mock_compute.return_value = {
            "faithfulness": 0.50, # Failing
            "answer_relevancy": 0.50,
            "context_precision": 0.50,
            "context_recall": 0.50,
        }

        exit_code, scorecard = await run_evaluation_pipeline(
            dataset_path="mock.json",
            base_url="http://localhost:8080",
            output_dir=output_dir,
            enforce_gates=True,
        )

        assert exit_code == 1
        assert scorecard["passed"] is False
