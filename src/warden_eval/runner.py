import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from warden_eval.client import WardenAPIClient
from warden_eval.ragas_gates import RagasGateEvaluator
from warden_eval.schema import GoldenEvalItem


def load_dataset(dataset_path: str | Path) -> List[GoldenEvalItem]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    return [GoldenEvalItem.model_validate(item) for item in raw_items]


async def run_evaluation_pipeline(
    dataset_path: str | Path = "eval/datasets/eval_golden_50.json",
    base_url: str = "http://localhost:8080",
    output_dir: str | Path = "eval/reports",
    enforce_gates: bool = True,
    delay_ms: float = 50.0,
) -> Tuple[int, Dict[str, Any]]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    items = load_dataset(dataset_path)
    client = WardenAPIClient(base_url=base_url)
    evaluator = RagasGateEvaluator()

    records = []
    consecutive_failures = 0
    print(f"\n[warden-eval] Starting evaluation over {len(items)} queries against {base_url}...")

    for i, item in enumerate(items, 1):
        try:
            resp = await client.query(item.query, item.caller_role)
            consecutive_failures = 0
            records.append({
                "eval_id": item.eval_id,
                "query": item.query,
                "caller_role": item.caller_role.value if hasattr(item.caller_role, "value") else str(item.caller_role),
                "ground_truth_answer": item.ground_truth_answer,
                "ground_truth_doc_ids": item.ground_truth_doc_ids,
                "ground_truth_chunks": item.ground_truth_chunks,
                "predicted_answer": resp.answer,
                "citations": [c.model_dump() for c in resp.citations],
                "latency_ms": resp.metrics.get("total_latency_ms", 0.0),
                "trace_id": resp.trace_id,
            })
            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000.0)
        except Exception as e:
            consecutive_failures += 1
            if consecutive_failures >= 5:
                raise ConnectionError("Circuit breaker triggered: 5 consecutive queries failed") from e

            print(f"[warden-eval] Query {item.eval_id} failed: {e}")
            records.append({
                "eval_id": item.eval_id,
                "query": item.query,
                "caller_role": item.caller_role.value if hasattr(item.caller_role, "value") else str(item.caller_role),
                "ground_truth_answer": item.ground_truth_answer,
                "ground_truth_doc_ids": item.ground_truth_doc_ids,
                "ground_truth_chunks": item.ground_truth_chunks,
                "predicted_answer": "",
                "citations": [],
                "error": str(e),
            })

    scores = evaluator.compute_metrics(records)
    eval_result = evaluator.evaluate_scores(scores, total_queries=len(items))

    scorecard_data = {
        "passed": eval_result.passed,
        "scores": eval_result.scores,
        "thresholds": eval_result.thresholds,
        "failing_metrics": eval_result.failing_metrics,
        "total_queries": eval_result.total_queries,
        "records": records,
    }

    # Save artifacts
    json_file = out_path / "ragas_scorecard.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(scorecard_data, f, indent=2)

    md_file = out_path / "ragas_scorecard.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(eval_result.markdown_report)

    print("\n" + eval_result.markdown_report)
    print(f"\n[warden-eval] Scorecard saved to {json_file} and {md_file}")

    exit_code = 0 if (eval_result.passed or not enforce_gates) else 1
    return exit_code, scorecard_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Warden Automated RAGAS Quality Gate Runner")
    parser.add_argument("--dataset", default="eval/datasets/eval_golden_50.json", help="Path to golden dataset")
    parser.add_argument("--base-url", default="http://localhost:8080", help="API Gateway URL")
    parser.add_argument("--output-dir", default="eval/reports", help="Directory for scorecard artifacts")
    parser.add_argument(
        "--enforce-gates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit 1 on quality regression",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    exit_code, _ = asyncio.run(
        run_evaluation_pipeline(
            dataset_path=args.dataset,
            base_url=args.base_url,
            output_dir=args.output_dir,
            enforce_gates=args.enforce_gates,
        )
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
