#!/usr/bin/env python3
"""CER evaluation: compare ASR hypothesis to reference text files.

Reads pack_parquet output from the three eval runs, matches each cut back
to its reference .txt file, then prints a per-model CER summary.

Usage:
    cd /data/xiaoqinfeng/code/VoxKitchen
    python experiments/children-asr/compare_cer.py
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path


DATA_ROOT = Path("/data/xiaoqinfeng/code/VoxKitchen/1200小时童声语音数据")
WORK_ROOT = Path("/data/xiaoqinfeng/code/VoxKitchen/work")

EVAL_RUNS = {
    "paraformer": "children-eval-paraformer",
    "sensevoice": "children-eval-sensevoice",
    "qwen3": "children-eval-qwen3",
}


# ---------------------------------------------------------------------------
# Reference text loader
# ---------------------------------------------------------------------------

def load_references() -> dict[str, str]:
    """Build {wav_stem → normalized_ref_text} from all labeled subdirs."""
    refs: dict[str, str] = {}
    for subdir in sorted(DATA_ROOT.iterdir()):
        if not subdir.is_dir() or subdir.name == "source":
            continue
        for wav in subdir.glob("*.wav"):
            # Find the matching .txt (same subdir, any name)
            txts = list(subdir.glob("*.txt"))
            if not txts:
                continue
            raw = txts[0].read_text(encoding="utf-8").strip()
            # Format: "0001 参观海洋馆" — strip leading seq number
            text = re.sub(r"^\d+\s+", "", raw)
            refs[wav.stem] = normalize_text(text)
    return refs


def normalize_text(text: str) -> str:
    """Strip tags/punctuation/spaces; keep only CJK chars + alphanumerics."""
    # Strip SenseVoice emotion/language tags: <|zh|><|HAPPY|>...
    text = re.sub(r"<\|[^|]*\|>", "", text)
    text = unicodedata.normalize("NFKC", text)
    # Remove all non-CJK non-alphanumeric chars (punctuation, spaces, etc.)
    text = re.sub(r"[^\w一-鿿㐀-䶿]", "", text)
    text = text.lower()
    return text


# ---------------------------------------------------------------------------
# CER
# ---------------------------------------------------------------------------

def edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance at character level."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def cer(hyp: str, ref: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(normalize_text(hyp), normalize_text(ref)) / len(ref)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_run(model_name: str, run_name: str, refs: dict[str, str]) -> None:
    import pyarrow.parquet as pq

    parquet_dirs = list((WORK_ROOT / run_name).rglob("metadata.parquet"))
    if not parquet_dirs:
        print(f"[{model_name}] No parquet found — run eval_{model_name}.yaml first")
        return

    table = pq.read_table(parquet_dirs[0]).to_pydict()
    audio_paths = table.get("audio_path", [])
    texts = table.get("text", [])

    results = []
    for path, hyp in zip(audio_paths, texts):
        if path is None:
            continue
        stem = Path(path).stem
        ref = refs.get(stem)
        if ref is None:
            continue
        hyp_norm = normalize_text(hyp or "")
        c = cer(hyp_norm, ref)
        results.append((stem, ref, hyp_norm, c))

    if not results:
        print(f"[{model_name}] No matching reference found")
        return

    avg_cer = sum(r[3] for r in results) / len(results)
    print(f"\n{'='*60}")
    print(f"Model: {model_name}   CER: {avg_cer:.1%}  ({len(results)} samples)")
    print(f"{'='*60}")
    for stem, ref, hyp, c in results:
        mark = "OK" if c < 0.1 else ("~" if c < 0.3 else "POOR")
        print(f"  [{mark}] {stem}")
        print(f"    REF: {ref}")
        print(f"    HYP: {hyp}")
        print(f"    CER: {c:.1%}")


def main() -> None:
    refs = load_references()
    if not refs:
        print("ERROR: No reference files found under", DATA_ROOT)
        return
    print(f"Loaded {len(refs)} reference transcriptions")

    for model_name, run_name in EVAL_RUNS.items():
        evaluate_run(model_name, run_name, refs)

    print("\nDone. Lower CER = better. Run the winning model's exp_<model>.yaml on source/.")


if __name__ == "__main__":
    main()
