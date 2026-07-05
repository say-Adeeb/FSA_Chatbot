"""Evaluation harness: retrieval metrics (hit-rate@k, MRR) + LLM-as-judge answer
quality, run against the real pipeline (real embeddings, real Groq calls).

Run from the project root:  python -m evaluation.run_eval
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from app.retrieval.llm import default_fallback_message, generate_answer
from app.retrieval.rag_pipeline import _detect_boost_terms, extract_course_name
from app.retrieval.vectordb import load_index, search_documents
from evaluation.golden_dataset import GOLDEN_DATASET
from evaluation.judge import judge_answer
from evaluation.metrics import hit_rate_at_k, reciprocal_rank

logging.basicConfig(level=logging.WARNING)

RESULTS_DIR = Path(__file__).parent / "results"


def evaluate_item(item: dict, k: int) -> dict:
    question = item["question"]
    detected_course = extract_course_name(question)

    search_query = (
        f"{question}\n{detected_course} curriculum modules topics syllabus"
        if detected_course != "the course"
        else question
    )
    retrieved = search_documents(
        search_query,
        k=k,
        course=detected_course if detected_course != "the course" else None,
        boost_terms=_detect_boost_terms(question),
    )

    if retrieved:
        answer = generate_answer(question, retrieved, course=detected_course)
    else:
        answer = default_fallback_message(detected_course)

    result = {
        "id": item["id"],
        "type": item["type"],
        "question": question,
        "detected_course": detected_course,
        "answer": answer,
    }

    if item["type"] == "course":
        keywords = item["keywords"]
        result["expected_course"] = item["course"]
        result["course_detected_correctly"] = detected_course == item["course"]
        result["hit_at_k"] = hit_rate_at_k(retrieved, keywords, k)
        result["reciprocal_rank"] = reciprocal_rank(retrieved, keywords)
    else:
        result["correctly_refused"] = "couldn't find" in answer.lower()

    context = "\n\n".join(retrieved)
    result["judge"] = judge_answer(question, context, answer)
    return result


def passed(r: dict) -> bool:
    if r["type"] == "course":
        return r["course_detected_correctly"] and r["hit_at_k"] == 1
    return r["correctly_refused"]


def summarize(results: list[dict]) -> dict:
    course_results = [r for r in results if r["type"] == "course"]
    off_topic_results = [r for r in results if r["type"] == "off_topic"]
    summary = {"n_total": len(results)}

    if course_results:
        summary["course_detection_accuracy"] = round(
            sum(r["course_detected_correctly"] for r in course_results) / len(course_results), 3
        )
        summary["hit_rate_at_k"] = round(
            sum(r["hit_at_k"] for r in course_results) / len(course_results), 3
        )
        summary["mrr"] = round(
            sum(r["reciprocal_rank"] for r in course_results) / len(course_results), 3
        )

    if off_topic_results:
        summary["refusal_accuracy"] = round(
            sum(r["correctly_refused"] for r in off_topic_results) / len(off_topic_results), 3
        )

    judged = [r["judge"] for r in results if r["judge"].get("grounded") is not None]
    if judged:
        summary["groundedness_rate"] = round(sum(j["grounded"] for j in judged) / len(judged), 3)
        summary["relevance_rate"] = round(sum(j["relevant"] for j in judged) / len(judged), 3)

    summary["overall_pass_rate"] = round(sum(passed(r) for r in results) / len(results), 3)
    return summary


def print_report(results: list[dict], summary: dict) -> None:
    print("\n=== Per-question results ===")
    for r in results:
        marker = "PASS" if passed(r) else "FAIL"
        print(f"[{marker}] {r['id']}: {r['question']}")
        if r["type"] == "course":
            print(
                f"    course_detected={r['detected_course']!r} "
                f"(expected {r['expected_course']!r}), "
                f"hit@k={r['hit_at_k']}, rr={r['reciprocal_rank']:.2f}"
            )
        else:
            print(f"    detected_course={r['detected_course']!r}, correctly_refused={r['correctly_refused']}")
        print(f"    judge: {r['judge']}")

    print("\n=== Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation harness.")
    parser.add_argument("--k", type=int, default=settings.RETRIEVAL_K)
    args = parser.parse_args()

    load_index()
    results = [evaluate_item(item, args.k) for item in GOLDEN_DATASET]
    summary = summarize(results)
    print_report(results, summary)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(f"\nSaved detailed report to {out_path}")


if __name__ == "__main__":
    main()
