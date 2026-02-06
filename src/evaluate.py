from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_test_data(path: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            records.append(
                {
                    "query": row["query"].strip(),
                    "positive": row["positive"].strip(),
                    "negative": row["negative"].strip(),
                }
            )
    return records


def compute_metrics(
    model: SentenceTransformer,
    test_data: List[Dict[str, str]],
) -> Dict[str, float]:
    queries = [item["query"] for item in test_data]
    positives = [item["positive"] for item in test_data]
    negatives = [item["negative"] for item in test_data]

    print("Encoding queries...")
    query_embs = model.encode(queries, normalize_embeddings=True, show_progress_bar=True)
    print("Encoding positives...")
    pos_embs = model.encode(positives, normalize_embeddings=True, show_progress_bar=True)
    print("Encoding negatives...")
    neg_embs = model.encode(negatives, normalize_embeddings=True, show_progress_bar=True)

    # Compute cosine similarities
    pos_scores = np.array(
        [cosine_similarity([q], [p])[0, 0] for q, p in zip(query_embs, pos_embs)]
    )
    neg_scores = np.array(
        [cosine_similarity([q], [n])[0, 0] for q, n in zip(query_embs, neg_embs)]
    )

    # Accuracy: positive > negative
    accuracy = (pos_scores > neg_scores).mean()

    # Mean scores
    mean_pos_score = pos_scores.mean()
    mean_neg_score = neg_scores.mean()
    mean_diff = mean_pos_score - mean_neg_score

    metrics = {
        "accuracy": float(accuracy),
        "mean_positive_score": float(mean_pos_score),
        "mean_negative_score": float(mean_neg_score),
        "mean_score_diff": float(mean_diff),
    }

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned BGE-M3 model")
    parser.add_argument("--model_path", type=Path, default=Path("outputs/bge-m3-kr"))
    parser.add_argument("--test_data", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--sample_size", type=int, default=500, help="Number of samples to evaluate")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model = SentenceTransformer(str(args.model_path))

    print(f"Loading test data from {args.test_data}...")
    test_data = load_test_data(args.test_data)

    # Sample for faster eval
    if len(test_data) > args.sample_size:
        import random
        random.seed(42)
        test_data = random.sample(test_data, args.sample_size)

    print(f"Evaluating on {len(test_data)} samples...")
    metrics = compute_metrics(model, test_data)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy (positive > negative): {metrics['accuracy']:.4f}")
    print(f"Mean Positive Score: {metrics['mean_positive_score']:.4f}")
    print(f"Mean Negative Score: {metrics['mean_negative_score']:.4f}")
    print(f"Mean Score Difference: {metrics['mean_score_diff']:.4f}")


if __name__ == "__main__":
    main()
