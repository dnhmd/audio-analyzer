import argparse
import csv
import json
import os
import time
import numpy as np
import requests
from pathlib import Path

API_URL = "http://localhost:8001/api/v1/analyze"

GENDER_MAP = {
    "male": "male",
    "female": "female",
    "m": "male",
    "f": "female"
}

def evaluate(dataset_dir: str, clips_tsv: str, limit: int = 100):
    results = []
    gender_correct = 0
    gender_total = 0
    confidence_sum = 0.0

    with open(clips_tsv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = [r for r in reader if r.get("gender") in GENDER_MAP][:limit]

    print(f"Evaluating {len(rows)} clips...")

    for i, row in enumerate(rows):
        clip_path = Path(dataset_dir) / row["path"]
        if not clip_path.exists():
            continue

        true_gender = GENDER_MAP[row["gender"]]

        with open(clip_path, "rb") as f:
            audio_bytes = f.read()

        start = time.monotonic()
        try:
            resp = requests.post(
                API_URL,
                data={"contact_id": f"eval-{i}"},
                files={"audio": (clip_path.name, audio_bytes, "audio/mpeg")},
                timeout=30
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            print(f"  SKIP {clip_path.name}: {e}")
            continue

        pred_gender = result["gender"]["prediction"]
        pred_conf = result["gender"]["confidence"]
        quality = result["audio_quality"]
        processing_ms = result["processing_ms"]

        if pred_gender != "unknown":
            gender_total += 1
            confidence_sum += pred_conf
            if pred_gender == true_gender:
                gender_correct += 1

        results.append({
            "file": clip_path.name,
            "true_gender": true_gender,
            "pred_gender": pred_gender,
            "confidence": pred_conf,
            "quality": quality,
            "processing_ms": processing_ms,
            "correct": pred_gender == true_gender
        })

        print(f"  [{i+1}/{len(rows)}] {clip_path.name}: true={true_gender} pred={pred_gender} conf={pred_conf:.2f} quality={quality} {processing_ms}ms")

    print("\n=== RESULTS ===")
    print(f"Clips evaluated:     {len(results)}")
    print(f"Gender accuracy:     {gender_correct}/{gender_total} = {gender_correct/max(gender_total,1)*100:.1f}%")
    print(f"Avg confidence:      {confidence_sum/max(gender_total,1):.3f}")
    print(f"Coverage (non-unk):  {gender_total/max(len(results),1)*100:.1f}%")

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to eval_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, help="Path to Common Voice clip directory")
    parser.add_argument("--tsv", required=True, help="Path to validated.tsv")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    evaluate(args.dataset_dir, args.tsv, args.limit)