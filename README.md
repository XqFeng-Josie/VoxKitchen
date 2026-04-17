<p align="center">
  <img src="voxkitchen_logo.svg" width="400" alt="VoxKitchen logo">
</p>

<p align="center">
  <a href="https://github.com/XqFeng-Josie/VoxKitchen/actions/workflows/ci.yml"><img src="https://github.com/XqFeng-Josie/VoxKitchen/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
</p>

Declarative speech data processing toolkit. Write a YAML recipe, run `vkit run`, get training-ready data.

> **Status:** Pre-alpha. API is unstable.

## Quickstart

```bash
pip install -e .
vkit init my-project && cd my-project
vkit run pipeline.yaml
```

Or with Docker (all 51 operators pre-installed, no dependency issues):

```bash
docker build -t voxkitchen .
docker run --rm voxkitchen run examples/pipelines/demo-no-asr.yaml
```

## How it works

A pipeline is a YAML file: **ingest** raw audio, pass it through **stages**, each stage transforms the data:

```yaml
version: "0.1"
name: my-pipeline
work_dir: ./work/${run_id}
ingest:
  source: dir
  args:
    root: /data/raw_audio
stages:
  - name: resample
    op: resample
    args: { target_sr: 16000 }
  - name: vad
    op: silero_vad
    args: { threshold: 0.5 }
  - name: asr
    op: faster_whisper_asr
    args: { model: large-v3, compute_type: float16 }
  - name: filter
    op: quality_score_filter
    args:
      conditions: ["duration > 1", "duration < 30", "metrics.snr > 10"]
  - name: pack
    op: pack_huggingface
    args: { output_dir: ./output/hf_dataset }
```

```bash
vkit run pipeline.yaml
vkit inspect run work/          # stage summary
vkit inspect cuts work/05_pack/cuts.jsonl.gz   # data statistics
```

## Install

### Docker (recommended)

```bash
docker build -t voxkitchen .

# Process your data
docker run --rm -v /data/raw_audio:/data voxkitchen run pipeline.yaml

# GPU support
docker run --rm --gpus all -v /data:/data voxkitchen run pipeline.yaml

# Interactive shell
docker run --rm -it --entrypoint bash voxkitchen
```

<!-- TODO: pre-built image available at:
```bash
docker pull ghcr.io/voxkitchen/voxkitchen:latest
```
-->

### pip

```bash
conda create -n voxkitchen python=3.11 -y && conda activate voxkitchen

# Core (21 operators: resample, ffmpeg_convert, snr_estimate, pack_jsonl, etc.)
pip install -e .

# Add what you need
pip install -e ".[asr,segment,pitch,dnsmos,pack]"

# Or everything
pip install -e ".[all]"
```

<details>
<summary>All extras groups</summary>

| Group | Operators enabled | Dependencies |
|-------|-------------------|--------------|
| `audio` | `speed_perturb` | torch, torchaudio |
| `segment` | `silero_vad`, `webrtc_vad`, `silence_split` | webrtcvad, librosa, torchaudio |
| `asr` | `faster_whisper_asr`, `whisperx_asr` | faster-whisper |
| `whisper` | `whisper_openai_asr`, `whisper_langid` | openai-whisper |
| `funasr` | `paraformer_asr`, `sensevoice_asr`, `emotion_recognize` | funasr |
| `wenet` | `wenet_asr` | wenet (GitHub) |
| `pitch` | `pitch_stats` | pyworld |
| `dnsmos` | `dnsmos_score`, `utmos_score` | speechmos |
| `quality` | `audio_fingerprint_dedup` | simhash |
| `diarize` | `pyannote_diarize` | pyannote.audio (needs `HF_TOKEN`) |
| `classify` | `speechbrain_langid` | speechbrain |
| `gender` | `gender_classify` (ina method) | inaSpeechSegmenter |
| `speaker` | `speaker_embed` | wespeaker (GitHub) |
| `enhance` | `speech_enhance` | deepfilternet |
| `align` | `forced_align`, `qwen3_asr` | qwen-asr |
| `codec` | `codec_tokenize` | encodec, descript-audio-codec |
| `tts-kokoro` | `tts_kokoro` | kokoro, misaki |
| `tts-chattts` | `tts_chattts` | ChatTTS |
| `tts-cosyvoice` | `tts_cosyvoice` | modelscope |
| `tts-fish-speech` | `tts_fish_speech` | fish-speech (GitHub) |
| `pack` | `pack_huggingface`, `pack_webdataset`, `pack_parquet` | datasets, webdataset, pyarrow |
| `viz` | HTML report | jinja2, plotly |
| `viz-panel` | Gradio panel | gradio |

</details>

<details>
<summary>GPU setup</summary>

Install PyTorch matching your CUDA driver **before** installing extras:

```bash
nvidia-smi | head -3                # check CUDA version
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> Do NOT use bare `pip install torch` — it may install a CUDA build incompatible with your driver.

</details>

### Configuration

Some operators require API tokens:

```bash
cp .env.example .env   # then edit .env
```

| Variable | Required by | How to get |
|----------|-------------|------------|
| `HF_TOKEN` | `pyannote_diarize` | [HuggingFace tokens](https://huggingface.co/settings/tokens) + accept the [pyannote model agreement](https://huggingface.co/pyannote/speaker-diarization-3.1) |

For Docker, pass via `-e HF_TOKEN=hf_xxx` or mount `-v $(pwd)/.env:/app/.env`.

## Operators

51 built-in operators across 8 categories:

| Category | Count | Operators |
|----------|-------|-----------|
| **Audio** | 5 | `resample`, `ffmpeg_convert`, `channel_merge`, `loudness_normalize`, `identity` |
| **Segmentation** | 4 | `silero_vad`, `webrtc_vad`, `fixed_segment`, `silence_split` |
| **Augmentation** | 4 | `speed_perturb`, `volume_perturb`, `noise_augment`, `reverb_augment` |
| **Annotation** | 17 | `faster_whisper_asr`, `whisper_openai_asr`, `whisperx_asr`, `paraformer_asr`, `sensevoice_asr`, `wenet_asr`, `qwen3_asr`, `pyannote_diarize`, `speechbrain_langid`, `whisper_langid`, `gender_classify`, `speaker_embed`, `speech_enhance`, `forced_align`, `emotion_recognize`, `codec_tokenize`, `mel_extract` |
| **Quality** | 11 | `snr_estimate`, `dnsmos_score`, `utmos_score`, `pitch_stats`, `clipping_detect`, `bandwidth_estimate`, `duration_filter`, `audio_fingerprint_dedup`, `quality_score_filter`, `speaker_similarity`, `cer_wer` |
| **Synthesize** | 4 | `tts_kokoro`, `tts_chattts`, `tts_cosyvoice`, `tts_fish_speech` |
| **Pack** | 6 | `pack_manifest`, `pack_jsonl`, `pack_huggingface`, `pack_webdataset`, `pack_parquet`, `pack_kaldi` |

```bash
vkit operators                  # list all
vkit operators show silero_vad  # config fields + YAML example
```

## CLI

```
vkit init <path>                    Scaffold a new project
vkit init <path> -t tts             Use a template (tts, asr, cleaning, speaker)
vkit run <yaml>                     Execute a pipeline
vkit run <yaml> --dry-run           Validate without executing
vkit validate <yaml>                Check YAML syntax
vkit download <recipe> --root <dir> Download a dataset
vkit operators                      List all operators
vkit operators show <name>          Operator detail + config
vkit recipes                        List available dataset recipes
vkit inspect cuts <path>            CutSet statistics
vkit inspect run <work_dir>         Pipeline run summary
vkit viz <path>                     Gradio interactive explorer
```

## Python API

For quick tasks without a YAML pipeline:

```python
from voxkitchen.tools import audio_info, transcribe, detect_speech, estimate_snr

audio_info("speech.wav")       # AudioInfo(sample_rate=16000, duration=3.2, ...)
transcribe("speech.wav")       # [Segment(start=0.0, end=3.2, text="Hello world")]
detect_speech("speech.wav")    # [SpeechSegment(start=0.5, end=2.8)]
estimate_snr("speech.wav")     # 18.3
```

<details>
<summary>More examples</summary>

```python
from voxkitchen.tools import (
    extract_speaker_embedding, enhance_speech, align_words, synthesize,
)

# Speaker embedding (requires: pip install voxkitchen[speaker])
emb = extract_speaker_embedding("speaker.wav")   # 512-d vector

# Speech enhancement (requires: pip install voxkitchen[enhance])
enhance_speech("noisy.wav", "clean.wav")

# Forced alignment (requires: pip install voxkitchen[align])
align_words("speech.wav", "hello world")
# [{"text": "hello", "start": 0.12, "end": 0.58}, ...]

# TTS synthesis (requires: pip install voxkitchen[tts-kokoro])
synthesize("Hello world!", "output.wav", engine="kokoro")

# Voice cloning (requires: pip install voxkitchen[tts-cosyvoice])
synthesize("你好", "clone.wav", engine="cosyvoice",
           reference_audio="ref.wav", reference_text="参考文本")
```

</details>

## Key features

- **Resumable** — each stage checkpoints; crashes resume from last good stage
- **Error tolerant** — bad cuts logged to `_errors.jsonl` and skipped
- **GC** — intermediate audio cleaned up automatically
- **Provenance** — every Cut tracks which operator produced it
- **Extensible** — register custom operators via Python entry_points

## Examples

See [`examples/pipelines/`](examples/pipelines/) for ready-to-run YAML files.

<details>
<summary>Troubleshooting</summary>

| Symptom | Solution |
|---------|----------|
| `torch.cuda.is_available()` returns False | PyTorch CUDA mismatch — reinstall matching version |
| `faster_whisper_asr` deadlocks on macOS | Use `whisper_openai_asr` instead |
| `pyannote_diarize` returns 403 | Accept the [model agreement](https://huggingface.co/pyannote/speaker-diarization-3.1) on HuggingFace |
| `silero_vad` hangs on first run | Pre-download: `python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad')"` |

</details>

## License

Apache 2.0. See [LICENSE](LICENSE).
