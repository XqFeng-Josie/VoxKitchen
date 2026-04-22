#!/usr/bin/env python3
"""Multi-ASR demo 结果分析.

读取 demo_multi_asr pipeline 的最后一个 stage 输出，展示：
- 三个模型的转录并排对比
- SenseVoice 的 emotion / audio_event / language（存于 supervision.custom）
- 与参考文本的 CER（有标注样本）

用法:
    cd /data/xiaoqinfeng/code/VoxKitchen
    python experiments/children-asr/analyze_multi_asr.py
"""

from __future__ import annotations

import gzip
import json
import re
import unicodedata
from pathlib import Path

DATA_ROOT  = Path("/data/xiaoqinfeng/code/VoxKitchen/1200小时童声语音数据")
CUTS_JSONL = Path("work/multi-asr-demo/04_asr_qwen3/cuts.jsonl.gz")

# demo_multi_asr.yaml 中的 stage name → 展示名（supervision.id = {cut_id}__{stage_name}_{n}）
STAGE_LABELS = {
    "asr_paraformer": "Paraformer",
    "asr_sensevoice": "SenseVoice",
    "asr_qwen3":      "Qwen3-ASR",
}


# ---------------------------------------------------------------------------
# 参考文本
# ---------------------------------------------------------------------------

def load_refs() -> dict[str, str]:
    """{ wav_stem → 参考文本 }，文本格式 "0001 参观海洋馆" 去掉序号前缀。"""
    refs: dict[str, str] = {}
    for sub in sorted(DATA_ROOT.iterdir()):
        if not sub.is_dir() or sub.name == "source":
            continue
        for wav in sub.glob("*.wav"):
            txts = list(sub.glob("*.txt"))
            if txts:
                raw = txts[0].read_text(encoding="utf-8").strip()
                refs[wav.stem] = re.sub(r"^\d+\s+", "", raw)
    return refs


# ---------------------------------------------------------------------------
# CER
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<\|[^|]*\|>")


def normalize(text: str) -> str:
    text = _TAG_RE.sub("", text or "")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w一-鿿㐀-䶿]", "", text)
    return re.sub(r"\s+", "", text).lower()


def cer(hyp: str, ref: str) -> float:
    h, r = list(normalize(hyp)), list(normalize(ref))
    if not r:
        return 0.0 if not h else 1.0
    n, m = len(h), len(r)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            dp[j] = prev if h[i - 1] == r[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m] / m


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_cuts(path: Path) -> list[dict]:
    cuts = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("__type__") == "cut":
                cuts.append(d)
    return cuts


def by_model(supervisions: list[dict]) -> dict[str, dict]:
    """{ 'Paraformer': sup_dict, … }  keyed by display label.

    supervision.id format: ``{cut_id}__{stage_name}_{n}``
    Stage name is the second token after splitting on ``__``.
    """
    result: dict[str, dict] = {}
    for s in supervisions:
        sid = s.get("id", "")
        # supervision.id = "{cut_id}__{stage_name}_{n}"
        # cut_id itself may contain "__" (e.g. "000001__rs16000"),
        # so take the LAST "__" segment as stage_name (+ optional counter).
        last = sid.rsplit("__", 1)[-1]          # "asr_paraformer_0"
        stage = last.rsplit("_", 1)[0] if last[-1:].isdigit() else last
        label = STAGE_LABELS.get(stage)
        if label:
            result[label] = s
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CUTS_JSONL.exists():
        print(f"找不到 {CUTS_JSONL}，请先执行:")
        print("  vkit docker run experiments/children-asr/demo_multi_asr.yaml \\")
        print("    -m /data/xiaoqinfeng/code/VoxKitchen/1200小时童声语音数据")
        return

    refs  = load_refs()
    cuts  = load_cuts(CUTS_JSONL)
    cer_accum: dict[str, list[float]] = {m: [] for m in STAGE_LABELS.values()}

    print("=" * 72)
    print("  Multi-ASR Demo — Paraformer · SenseVoice · Qwen3-ASR 对比")
    print("=" * 72)

    for cut in cuts:
        ap   = cut.get("recording", {}).get("sources", [{}])[0].get("source", "")
        dur  = cut.get("duration", 0)
        sups = by_model(cut.get("supervisions", []))
        if not sups:
            continue

        stem = re.sub(r"[_]{1,2}rs\d+$", "", Path(ap).stem)
        ref  = refs.get(stem)

        print(f"\n{'─'*72}")
        print(f"  {stem}  ({dur:.1f}s)" + (f"  参考: {ref}" if ref else ""))
        print()

        for label in ["Paraformer", "SenseVoice", "Qwen3-ASR"]:
            s = sups.get(label)
            if s is None:
                print(f"  [{label:<12}] —")
                continue

            text   = s.get("text") or ""
            custom = s.get("custom") or {}

            cer_str = ""
            if ref:
                c = cer(text, ref)
                cer_accum[label].append(c)
                cer_str = f"  CER={c:.0%}"

            meta = ""
            if label == "SenseVoice":
                lang    = s.get("language") or "—"
                emotion = custom.get("emotion", "—")
                event   = custom.get("audio_event", "—")
                meta = f"  [lang={lang}  emotion={emotion}  event={event}]"

            print(f"  [{label:<12}] {text}{cer_str}{meta}")

    print(f"\n{'='*72}")
    print("  CER 汇总")
    print(f"{'─'*72}")
    for label, scores in cer_accum.items():
        if scores:
            print(f"  {label:<14}  {sum(scores)/len(scores):.1%}  ({len(scores)} 样本)")

    print(f"\n  输出文件: work/multi-asr-demo/05_pack/parquet_output/metadata.parquet")
    print(f"  supervisions 列: JSON 数组，每项含 id / text / language / custom")


if __name__ == "__main__":
    main()
