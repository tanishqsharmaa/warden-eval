from warden_eval.bakeoff import BakeOffHarness, BakeOffResult, PipelineConfig, PipelineType
from warden_eval.metrics import calculate_hit_at_k, calculate_mrr_at_k, calculate_ndcg_at_k
from warden_eval.schema import GoldenEvalItem, RoleEnum


def test_ranking_metrics_calculation():
    # Target chunks: [2, 3]
    # Prediction: [1, 2, 4, 5, 3]
    pred = [1, 2, 4, 5, 3]
    relevant = {2, 3}

    hit5 = calculate_hit_at_k(pred, relevant, k=5)
    assert hit5 == 1.0

    hit1 = calculate_hit_at_k(pred, relevant, k=1)
    assert hit1 == 0.0

    # First relevant item is at rank 2 (1-based index) -> 1/2 = 0.5
    mrr5 = calculate_mrr_at_k(pred, relevant, k=5)
    assert mrr5 == 0.5

    # NDCG@5
    ndcg5 = calculate_ndcg_at_k(pred, relevant, k=5)
    assert 0.0 < ndcg5 <= 1.0


def test_bakeoff_harness_execution():
    items = [
        GoldenEvalItem(
            eval_id="EVAL-001",
            query="Bereavement leave days?",
            caller_role=RoleEnum.EMPLOYEE,
            ground_truth_answer="5 days",
            ground_truth_doc_ids=["DOC-HR-LEAVE-2026"],
            ground_truth_chunks=[2, 3],
        ),
        GoldenEvalItem(
            eval_id="EVAL-002",
            query="PTO annual accrual?",
            caller_role=RoleEnum.EMPLOYEE,
            ground_truth_answer="18 days",
            ground_truth_doc_ids=["DOC-HR-LEAVE-2026"],
            ground_truth_chunks=[0, 1],
        ),
    ]

    # Mock candidate generator returning raw candidates for each item
    def mock_retrieval_fn(query: str, role: RoleEnum):
        # 10 raw candidate chunks
        return [
            {"doc_id": "DOC-HR-LEAVE-2026", "chunk_index": i, "rrf_score": 1.0 / (i + 1)}
            for i in range(10)
        ]

    # Mock rerankers
    def mock_laya_reranker(query: str, candidates: list):
        # Simulates Laya ModernBERT prioritizing chunks 2 and 3
        sorted_cands = sorted(
            candidates,
            key=lambda c: 0.95 if c["chunk_index"] in [2, 3, 0, 1] else 0.10,
            reverse=True,
        )
        return sorted_cands, 41.5  # 41.5ms latency

    def mock_bge_reranker(query: str, candidates: list):
        # Simulates BGE-Reranker-v2-m3 prioritizing chunks
        sorted_cands = sorted(
            candidates,
            key=lambda c: 0.94 if c["chunk_index"] in [2, 3, 0, 1] else 0.12,
            reverse=True,
        )
        return sorted_cands, 128.0  # 128ms latency

    harness = BakeOffHarness(
        pipelines=[
            PipelineConfig(
                pipeline_id="Pipeline A (Baseline)",
                pipeline_type=PipelineType.BASELINE_RAW_RRF,
                reranker_fn=None,
            ),
            PipelineConfig(
                pipeline_id="Pipeline B (Production Standard)",
                pipeline_type=PipelineType.PRODUCTION_LAYA,
                reranker_fn=mock_laya_reranker,
            ),
            PipelineConfig(
                pipeline_id="Pipeline C (Comparative Benchmark)",
                pipeline_type=PipelineType.COMPARATIVE_BGE,
                reranker_fn=mock_bge_reranker,
            ),
        ]
    )

    results: BakeOffResult = harness.run(items, retrieval_fn=mock_retrieval_fn)

    assert len(results.pipeline_summaries) == 3
    # Verify Laya achieves low latency
    laya_summary = next(s for s in results.pipeline_summaries if "Pipeline B" in s.pipeline_id)
    assert laya_summary.avg_latency_ms < 50.0
    assert laya_summary.hit_at_5 > 0.80

    # Verify report generated
    assert len(results.markdown_report) > 100
    assert "Pipeline A (Baseline)" in results.markdown_report
    assert "Pipeline B (Production Standard)" in results.markdown_report
    assert "Pipeline C (Comparative Benchmark)" in results.markdown_report
