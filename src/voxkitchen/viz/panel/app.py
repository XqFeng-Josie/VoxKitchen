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

    def get_table_data(min_dur: float, max_dur: float) -> list[list[str]]:
        rows = []
        for c in all_cuts:
            if not (min_dur <= c.duration <= max_dur):
                continue
            text = next((s.text for s in c.supervisions if s.text), "")
            speaker = next((s.speaker for s in c.supervisions if s.speaker), "")
            language = next((s.language for s in c.supervisions if s.language), "")
            snr = c.metrics.get("snr", "")
            rows.append([c.id, f"{c.duration:.2f}", text[:80], speaker, language, str(snr)])
        return rows

    with gr.Blocks(title="VoxKitchen Explorer") as app:
        gr.Markdown(
            f"# VoxKitchen Explorer\n**{stats['count']} cuts** | {stats['total_duration_s']:.1f}s total"
        )
        with gr.Row():
            min_dur = gr.Slider(0, 120, value=0, step=0.1, label="Min duration (s)")
            max_dur = gr.Slider(0, 120, value=120, step=0.1, label="Max duration (s)")
        table = gr.Dataframe(
            headers=["ID", "Duration", "Text", "Speaker", "Language", "SNR"],
            value=get_table_data(0, 120),
        )
        for slider in [min_dur, max_dur]:
            slider.change(get_table_data, [min_dur, max_dur], table)

    return app


def launch(manifest_path: str, port: int = 7860) -> None:
    """Create and launch the Gradio panel."""
    app = create_app(manifest_path)
    app.launch(server_name="127.0.0.1", server_port=port)
