from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BGE-M3 for Korean electronics embeddings")
    parser.add_argument("--train_data", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default="BAAI/bge-m3")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/bge-m3-kr"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()

    # CRITICAL: Disable MPS BEFORE importing torch
    if args.use_cpu:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    # Import torch AFTER setting environment variables
    import torch
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    if args.use_cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    model = SentenceTransformer(args.model_name, device=device)
    model.max_seq_length = args.max_seq_length

    # Load examples
    examples: List[InputExample] = []
    with args.train_data.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            query = row["query"].strip()
            positive = row["positive"].strip()
            examples.append(InputExample(texts=[query, positive]))

    train_dataloader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = max(10, int(len(train_dataloader) * args.epochs * 0.1))

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        use_amp=False,
        optimizer_params={"lr": args.lr},
        output_path=str(args.output_dir),
        show_progress_bar=True,
    )

    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
