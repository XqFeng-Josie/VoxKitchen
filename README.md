# VoxKitchen

Declarative speech data processing toolkit. Write a YAML recipe, run `vkit run`, get training-ready data.

> **Status:** Pre-alpha. API is unstable.

## Requirements

- Python 3.10+
- ffmpeg (for audio format conversion)
- GPU is **optional** — most operators run on CPU. GPU accelerates ASR, VAD, and diarization.

<details>
<summary>GPU setup (only if you have NVIDIA GPU)</summary>

Install PyTorch matching your CUDA driver version:

```bash
nvidia-smi | head -3                # check CUDA version
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124  # example for CUDA 12.4
```

> Do NOT use bare `pip install torch` — it may install a CUDA build incompatible with your driver.

</details>

## Install

```bash
# Create virtual environment
conda create -n voxkitchen python=3.11 -y
conda activate voxkitchen

# Install core (ffmpeg_convert, resample, snr_estimate, etc.)
pip install -e .

# Install extras as needed:
#
#   ASR engines
pip install -e ".[asr]"           # faster-whisper (CTranslate2, GPU recommended)
pip install -e ".[whisper]"       # OpenAI whisper (pure PyTorch, macOS-safe)
pip install -e ".[funasr]"        # Paraformer, SenseVoice (FunASR)
pip install -e ".[wenet]"         # WeNet ASR
#
#   Audio analysis
pip install -e ".[audio]"         # torch + torchaudio
pip install -e ".[segment]"       # webrtcvad, librosa, torchaudio (VAD, silence split)
pip install -e ".[pitch]"         # PyWorld pitch analysis
pip install -e ".[dnsmos]"        # DNSMOS P.835/P.808 + UTMOS quality scores
pip install -e ".[quality]"       # simhash (audio deduplication)
#
#   Speaker & language
pip install -e ".[diarize]"       # pyannote speaker diarization (needs HF_TOKEN, see below)
pip install -e ".[classify]"      # SpeechBrain language ID
pip install -e ".[gender]"        # inaSpeechSegmenter gender detection
pip install -e ".[speaker]"       # WeSpeaker speaker embeddings
#
#   Enhancement & alignment
pip install -e ".[enhance]"       # DeepFilterNet speech denoising
pip install -e ".[align]"         # CTC forced alignment
#
#   Output & visualization
pip install -e ".[pack]"          # HuggingFace datasets, WebDataset, Parquet
pip install -e ".[viz]"           # HTML report (Jinja2 + Plotly)
pip install -e ".[viz-panel]"     # Gradio interactive panel

# Or pick what you need
pip install -e ".[asr,whisper,pitch,dnsmos,segment,diarize]"

# Or install everything at once
pip install -e ".[all]"
```

### Configuration

Some operators require API tokens. Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Variable | Required by | How to get |
|----------|-------------|------------|
| `HF_TOKEN` | `pyannote_diarize` | [HuggingFace](https://huggingface.co/settings/tokens) — also accept the [pyannote model agreement](https://huggingface.co/pyannote/speaker-diarization-3.1) |

### Troubleshooting

| Symptom | Solution |
|---------|----------|
| `torch.cuda.is_available()` returns False | PyTorch CUDA mismatch — install matching version: `pip install torch --index-url .../cuXXX` |
| `faster_whisper_asr` deadlocks on macOS | Use `whisper_openai_asr` instead (pure PyTorch, no CTranslate2) |
| `pyannote_diarize` returns 403 | Accept the [model agreement](https://huggingface.co/pyannote/speaker-diarization-3.1) on HuggingFace |
| `silero_vad` hangs on first run | Network issue — pre-download: `python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad')"` |

## 30-second quickstart

```bash
# Scaffold a project
vkit init my-project && cd my-project

# Run the pipeline
vkit run pipeline.yaml

# Inspect results
vkit inspect run work/
vkit inspect cuts work/01_pack/cuts.jsonl.gz
open work/report.html
```

## How it works

A pipeline is a YAML file: **ingest** raw audio, pass it through **stages** (operators), each stage transforms the data:

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

## Operators

41 built-in operators across 7 categories:

| Category | Operators |
|----------|-----------|
| **Audio** | `resample`, `ffmpeg_convert`, `channel_merge`, `loudness_normalize` |
| **Segmentation** | `silero_vad`, `webrtc_vad`, `fixed_segment`, `silence_split` |
| **Augmentation** | `speed_perturb`, `volume_perturb`, `noise_augment`, `reverb_augment` |
| **Annotation** | `faster_whisper_asr`, `whisper_openai_asr`, `whisperx_asr`, `paraformer_asr`, `sensevoice_asr`, `wenet_asr`, `pyannote_diarize`, `speechbrain_langid`, `whisper_langid`, `gender_classify`, `speaker_embed`, `speech_enhance`, `forced_align` |
| **Quality** | `snr_estimate`, `dnsmos_score`, `utmos_score`, `pitch_stats`, `clipping_detect`, `bandwidth_estimate`, `duration_filter`, `audio_fingerprint_dedup`, `quality_score_filter` |
| **Pack** | `pack_manifest`, `pack_jsonl`, `pack_huggingface`, `pack_webdataset`, `pack_parquet`, `pack_kaldi` |

```bash
# List all operators
vkit operators

# Show config fields + YAML example for any operator
vkit operators show silero_vad
```

## CLI commands

```
vkit init <path>              Scaffold a new project
vkit run <yaml>               Execute a pipeline
vkit run <yaml> --dry-run     Validate without executing
vkit validate <yaml>          Check YAML syntax
vkit ingest --source dir ...  Build CutSet without a pipeline
vkit inspect cuts <path>      CutSet statistics
vkit inspect run <work_dir>   Pipeline stage summary
vkit inspect trace <id> --in <work_dir>   Trace a cut's provenance
vkit inspect errors <work_dir>            Show per-cut errors
vkit operators                List all operators
vkit operators show <name>    Operator detail + config
vkit viz <path>               Gradio interactive explorer
```

## Python tools API

For quick tasks without writing a YAML pipeline:

```python
from voxkitchen.tools import (
    audio_info, transcribe, detect_speech, estimate_snr,
    extract_speaker_embedding, enhance_speech, align_words,
)

audio_info("speech.wav")
# AudioInfo(sample_rate=16000, duration=3.2, num_channels=1, format='WAV')

transcribe("speech.wav", model="tiny")
# [Segment(start=0.0, end=3.2, text="Hello world")]

detect_speech("speech.wav", method="silero")
# [SpeechSegment(start=0.5, end=2.8)]

estimate_snr("speech.wav")
# 18.3

# Speaker embedding (requires: pip install voxkitchen[speaker])
emb = extract_speaker_embedding("speaker.wav")
# [0.12, -0.34, 0.56, ...]  (512-d vector)

# Speech enhancement (requires: pip install voxkitchen[enhance])
enhance_speech("noisy.wav", "clean.wav", aggressiveness=0.5)

# Forced alignment (requires: pip install voxkitchen[align])
align_words("speech.wav", "hello world")
# [{"text": "hello", "start": 0.12, "end": 0.58}, ...]
```

## Key features

- **Resumable** — each stage checkpoints with `_SUCCESS` marker; crashes resume from last good stage
- **Error tolerant** — bad cuts are logged to `_errors.jsonl` and skipped, pipeline continues
- **GC** — intermediate derived audio is cleaned up automatically (`--keep-intermediates` to disable)
- **Provenance** — every Cut tracks which operator produced it from which parent
- **Extensible** — register custom operators via Python entry_points

## Examples

See [`examples/pipelines/`](examples/pipelines/) for ready-to-run YAML files.

## License

Apache 2.0. See [LICENSE](LICENSE).
