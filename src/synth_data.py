from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Dict


QUERY_TEMPLATES = [
    "{term}의 정의는 뭐야?",
    "{term} 원리 설명해줘",
    "{term}가 회로에서 하는 역할은?",
    "{term} 관련 핵심 개념 정리해줘",
    "{term}의 특징과 용도",
    "{term}는 어디에 쓰이나요?",
    "{term}와 관련된 기본 법칙 알려줘",
    "{term}의 장단점은?",
]

PASSAGE_TEMPLATES = [
    "{term}는 전기/전자 시스템에서 핵심 구성요소로, 회로의 동작과 성능에 직접적인 영향을 준다.",
    "{term}는 신호 처리나 전력 변환에서 중요한 역할을 하며, 설계 시 파라미터 선택이 중요하다.",
    "{term}는 회로 안정성과 효율에 영향을 주는 요소로, 관련 규격과 신뢰성 요구사항을 고려해야 한다.",
    "{term}는 특정 동작 조건에서 원하는 전기적 특성을 확보하기 위해 사용된다.",
    "{term}는 시스템 전력 품질과 EMI/EMC 요구사항을 만족하기 위해 적절한 설계가 필요하다.",
]

NEGATIVE_TEMPLATES = [
    "{other}는 {term}과 다른 목적을 가진 부품으로 동작 원리가 다르다.",
    "{other}는 {term}의 대체재가 아니며, 회로에서 수행하는 기능이 구분된다.",
    "{other}는 다른 물리 현상을 기반으로 하여 {term}과 직접적인 연관이 적다.",
]

ACRONYM_QUERY_TEMPLATES = [
    "{acronym}이 뭐야?",
    "{acronym} 풀네임 알려줘",
    "{acronym} 의미 설명해줘",
    "{acronym} 정의는?",
]

ACRONYM_POSITIVE_TEMPLATES = [
    "{acronym}은/는 {full}의 약자로, {definition}",
    "{acronym} = {full}. {definition}",
    "{full}({acronym})는 {definition}",
]


def read_terms(path: Path) -> List[str]:
    terms = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [t for t in terms if t]


def read_acronyms(path: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if {"acronym", "full", "definition"}.issubset(row):
                records.append(
                    {
                        "acronym": row["acronym"].strip(),
                        "full": row["full"].strip(),
                        "definition": row["definition"].strip(),
                    }
                )
    return records


def synthesize_examples(terms: List[str], n_pairs_per_term: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    examples: List[Dict[str, str]] = []
    for term in terms:
        for _ in range(n_pairs_per_term):
            query = rng.choice(QUERY_TEMPLATES).format(term=term)
            positive = rng.choice(PASSAGE_TEMPLATES).format(term=term)
            other = rng.choice([t for t in terms if t != term])
            negative = rng.choice(NEGATIVE_TEMPLATES).format(term=term, other=other)
            examples.append(
                {
                    "query": query,
                    "positive": positive,
                    "negative": negative,
                    "term": term,
                    "type": "term",
                }
            )
    rng.shuffle(examples)
    return examples


def synthesize_acronym_examples(
    acronyms: List[Dict[str, str]],
    terms: List[str],
    n_pairs_per_acronym: int,
    seed: int,
) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    examples: List[Dict[str, str]] = []
    for item in acronyms:
        acronym = item["acronym"]
        full = item["full"]
        definition = item["definition"]
        for _ in range(n_pairs_per_acronym):
            query = rng.choice(ACRONYM_QUERY_TEMPLATES).format(acronym=acronym)
            positive = rng.choice(ACRONYM_POSITIVE_TEMPLATES).format(
                acronym=acronym, full=full, definition=definition
            )
            other = rng.choice([t for t in terms if t not in {acronym, full}])
            negative = rng.choice(NEGATIVE_TEMPLATES).format(term=acronym, other=other)
            examples.append(
                {
                    "query": query,
                    "positive": positive,
                    "negative": negative,
                    "acronym": acronym,
                    "full": full,
                    "type": "acronym",
                }
            )
    rng.shuffle(examples)
    return examples


def save_jsonl(records: Iterable[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize Korean electronics embedding dataset")
    parser.add_argument("--terms", type=Path, default=Path("data/terms_electronics_ko.txt"))
    parser.add_argument("--acronyms", type=Path, default=Path("data/acronyms_ko.jsonl"))
    parser.add_argument("--train-output", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--test-output", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--pairs-per-term", type=int, default=6)
    parser.add_argument("--pairs-per-acronym", type=int, default=8)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    terms = read_terms(args.terms)
    examples = synthesize_examples(terms, n_pairs_per_term=args.pairs_per_term, seed=args.seed)
    if args.acronyms.exists():
        acronym_records = read_acronyms(args.acronyms)
        examples.extend(
            synthesize_acronym_examples(
                acronym_records,
                terms,
                n_pairs_per_acronym=args.pairs_per_acronym,
                seed=args.seed + 1,
            )
        )
    
    # Train/test split
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    split_idx = int(len(examples) * (1 - args.test_split))
    train_examples = examples[:split_idx]
    test_examples = examples[split_idx:]
    
    save_jsonl(train_examples, args.train_output)
    save_jsonl(test_examples, args.test_output)
    print(f"Saved {len(train_examples)} train records to {args.train_output}")
    print(f"Saved {len(test_examples)} test records to {args.test_output}")


if __name__ == "__main__":
    main()
