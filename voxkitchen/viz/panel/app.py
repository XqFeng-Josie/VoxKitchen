"""Local Gradio panel for interactive CutSet exploration."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def create_app(manifest_path: str) -> Any:
    import gradio as gr

    from voxkitchen.schema.cutset import CutSet
    from voxkitchen.viz.stats import compute_cutset_stats

    cuts = CutSet.from_jsonl_gz(Path(manifest_path))
    all_cuts = list(cuts)
    stats = compute_cutset_stats(cuts)

    # Collect distinct filter values
    languages: set[str] = set()
    speakers: set[str] = set()
    for c in all_cuts:
        for s in c.supervisions:
            if s.language:
                languages.add(s.language)
            if s.speaker:
                speakers.add(s.speaker)
    lang_choices = sorted(languages)
    speaker_choices = sorted(speakers)

    max_dur = max((c.duration for c in all_cuts), default=120.0)
    max_dur = min(max_dur, 300.0)  # cap slider range

    def _cut_summary(c: Any) -> dict[str, str]:
        text = next((s.text for s in c.supervisions if s.text), "")
        speaker = next((s.speaker for s in c.supervisions if s.speaker), "")
        language = next((s.language for s in c.supervisions if s.language), "")
        snr = c.metrics.get("snr", "")
        return {
            "id": c.id,
            "duration": f"{c.duration:.2f}",
            "text": text[:80],
            "speaker": speaker,
            "language": language,
            "snr": str(snr),
        }

    def _match_filters(
        c: Any, min_d: float, max_d: float, lang: str, spk: str, snr_min: float
    ) -> bool:
        if not (min_d <= c.duration <= max_d):
            return False
        if lang:
            cl = next((s.language for s in c.supervisions if s.language), "")
            if cl != lang:
                return False
        if spk:
            cs = next((s.speaker for s in c.supervisions if s.speaker), "")
            if cs != spk:
                return False
        if snr_min > 0:
            snr_val = c.metrics.get("snr")
            if snr_val is None or float(snr_val) < snr_min:
                return False
        return True

    def filter_cuts(
        min_d: float, max_d: float, lang: str, spk: str, snr_min: float
    ) -> list[list[str]]:
        rows = []
        for c in all_cuts:
            if not _match_filters(c, min_d, max_d, lang, spk, snr_min):
                continue
            a = _cut_summary(c)
            rows.append([a["id"], a["duration"], a["text"], a["speaker"], a["language"], a["snr"]])
        return rows

    def get_cut_detail(cut_id: str) -> str:
        for c in all_cuts:
            if c.id == cut_id:
                lines = [
                    f"**Cut:** {c.id}",
                    f"**Recording:** {c.recording_id}",
                    f"**Start:** {c.start:.3f}s  **Duration:** {c.duration:.3f}s",
                ]
                if c.metrics:
                    lines.append(
                        "**Metrics:** " + ", ".join(f"{k}={v}" for k, v in c.metrics.items())
                    )
                if c.provenance:
                    p = c.provenance
                    lines.append(f"**Provenance:** {p.generated_by} (from {p.source_cut_id})")
                for i, s in enumerate(c.supervisions):
                    lines.append(
                        f"**Supervision {i}:** text={s.text!r} speaker={s.speaker} "
                        f"lang={s.language} gender={s.gender}"
                    )
                return "\n\n".join(lines)
        return f"Cut `{cut_id}` not found."

    # Fixed export path — reused across clicks, no temp file leak.
    _export_path = Path(manifest_path).parent / "_export_subset.jsonl.gz"

    def export_filtered(
        min_d: float, max_d: float, lang: str, spk: str, snr_min: float
    ) -> str | None:
        import gzip
        import json

        filtered = [c for c in all_cuts if _match_filters(c, min_d, max_d, lang, spk, snr_min)]
        if not filtered:
            return None
        with gzip.open(str(_export_path), "wt", encoding="utf-8") as f:
            for c in filtered:
                f.write(json.dumps(c.model_dump(mode="json"), ensure_ascii=False) + "\n")
        return str(_export_path)

    with gr.Blocks(title="VoxKitchen Explorer") as app:
        gr.Markdown(
            f"# VoxKitchen Explorer\n"
            f"**{stats['count']} cuts** | {stats['total_duration_s']:.1f}s total"
        )

        with gr.Row():
            min_dur = gr.Slider(0, max_dur, value=0, step=0.1, label="Min duration (s)")
            max_dur_sl = gr.Slider(0, max_dur, value=max_dur, step=0.1, label="Max duration (s)")
        with gr.Row():
            lang_dd = gr.Dropdown(["", *lang_choices], value="", label="Language")
            spk_dd = gr.Dropdown(["", *speaker_choices], value="", label="Speaker")
            snr_sl = gr.Slider(0, 50, value=0, step=1, label="Min SNR (dB)")

        filter_inputs = [min_dur, max_dur_sl, lang_dd, spk_dd, snr_sl]

        table = gr.Dataframe(
            headers=["ID", "Duration", "Text", "Speaker", "Language", "SNR"],
            value=filter_cuts(0, max_dur, "", "", 0),
        )

        for inp in filter_inputs:
            inp.change(filter_cuts, filter_inputs, table)  # type: ignore[attr-defined]

        gr.Markdown("---\n### Cut Detail")
        with gr.Row():
            detail_input = gr.Textbox(label="Enter Cut ID", placeholder="paste a cut ID...")
            detail_btn = gr.Button("Show Detail")
        detail_output = gr.Markdown()
        detail_btn.click(get_cut_detail, [detail_input], detail_output)

        gr.Markdown("---")
        export_btn = gr.Button("Export filtered subset")
        export_file = gr.File(label="Download")
        export_btn.click(export_filtered, filter_inputs, export_file)

    return app


def launch(manifest_path: str, port: int = 7860) -> None:
    """Create and launch the Gradio panel."""
    app = create_app(manifest_path)
    app.launch(server_name="127.0.0.1", server_port=port)
