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

## Install

Two install paths. Both give you the same `vkit` CLI.

### Docker — runs any pipeline

Every operator works regardless of combination. Models are pre-baked
into the image. **This is the path to use if you don't want to think
about dependencies.**

```bash
docker pull ghcr.io/xqfeng-josie/voxkitchen:slim   # CPU, ~3 GB
# or :asr / :diarize / :tts / :fish-speech / :latest — see tag matrix below
```

### pip — fast local iteration, narrow pipelines, library embedding

Works for pipelines whose operators share one **dependency cluster**
(e.g. all core, or core + ASR, or core + TTS). Also how you embed
`voxkitchen.tools.transcribe(...)` into your own Python code.

```bash
conda create -n voxkitchen python=3.11 -y && conda activate voxkitchen
pip install -e ".[asr,pack]"   # pick only the extras your pipeline uses
```

**Known limitation**: `pip install voxkitchen[all]` will fail at the pip
resolver — pyannote 4, funasr, and fish-speech demand mutually exclusive
torch/numpy versions. For cross-cluster pipelines use Docker, which
ships all the clusters as isolated envs in one image.

## Quickstart

The same `vkit` CLI works both locally and against a Docker image —
prefix with `docker` to run inside a container:

### With Docker

```bash
vkit docker run examples/pipelines/demo-no-asr.yaml   # run in container
vkit docker doctor                                     # per-env health report
vkit docker run --tag asr my.yaml                      # switch image tag
vkit docker build latest                               # build image locally
```

`vkit docker` auto-handles the non-trivial flags — non-root `--user`,
`./work` and `./data` mounts, `./.env` loading, GPU autodetection.
Raw `docker run` form (for CI / non-pip users) is in
[Install reference](#install-reference) below.

### With pip (local)

```bash
vkit init my-project -t asr   # scaffold a starter ASR pipeline
cd my-project
vkit run pipeline.yaml
```

Templates: `tts`, `asr`, `cleaning`, `speaker`. Or write your own —
every pipeline is a YAML file (next section shows the structure).

## How it works

A pipeline is a YAML file: **ingest** raw audio, pass it through
**stages**, each stage transforms the data.

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

After a run:

```bash
vkit inspect run work/                         # stage summary
vkit inspect cuts work/05_pack/cuts.jsonl.gz   # data statistics
vkit doctor                                    # per-env operator availability
```

## Install reference

### Docker tag matrix

| Tag | Contains | GPU | Size |
|-----|----------|-----|------|
| `voxkitchen:slim`        | core env only (VAD, quality, pack, speaker embed, enhancement) | no  | ~3 GB |
| `voxkitchen:asr`         | core + ASR family (faster-whisper, funasr, qwen3, forced alignment) | yes | ~10 GB |
| `voxkitchen:diarize`     | core + pyannote speaker diarization | yes | ~5 GB |
| `voxkitchen:tts`         | core + kokoro / ChatTTS / CosyVoice | yes | ~10 GB |
| `voxkitchen:fish-speech` | core + fish-speech (isolated torch 2.8 stack) | yes | ~6 GB |
| `voxkitchen:latest`      | all five envs merged (cross-cluster pipelines) | yes | ~25 GB |

`voxkitchen:latest` contains five isolated Python environments in one
image. VoxKitchen already checkpoints each pipeline stage to disk, so
the runner dispatches stages across envs via subprocess — users write
ordinary pipelines and the multi-env is invisible. See
[`docs/architecture/multi-env.md`](docs/architecture/multi-env.md) for
the full design.

### pip extras — which combine cleanly

Install any subset whose groups map to a **single cluster**:

| Cluster | Safe to combine in one `pip install` |
|---|---|
| Core | `audio`, `segment`, `quality`, `pack`, `pitch`, `dnsmos`, `classify`, `enhance`, `codec`, `speaker`, `viz`, `viz-panel` |
| ASR | core + `asr`, `whisper`, `funasr`, `align`, `wenet` |
| Diarize | core + `diarize` |
| TTS | core + `tts-kokoro`, `tts-chattts`, `tts-cosyvoice` |
| Fish-Speech | `tts-fish-speech` (**do not mix with anything else** — needs torch 2.8) |

Mixing across clusters (`diarize + funasr`, `tts-cosyvoice + tts-fish-speech`,
etc.) is the case where pip fails and Docker wins. The authoritative
mapping is [`voxkitchen/runtime/env_resolver.py`](voxkitchen/runtime/env_resolver.py).

<details>
<summary>All extras groups (operators and deps per group)</summary>

| Group | Operators enabled | Dependencies |
|-------|-------------------|--------------|
| `audio` | `speed_perturb` | torch, torchaudio |
| `segment` | `silero_vad`, `webrtc_vad`, `silence_split` | webrtcvad, librosa, torchaudio |
| `asr` | `faster_whisper_asr`, `whisperx_asr` | faster-whisper |
| `whisper` | `whisper_openai_asr`, `whisper_langid` | openai-whisper |
| `funasr` | `paraformer_asr`, `sensevoice_asr`, `emotion_recognize` | funasr |
| `wenet` | `wenet_asr` | wenet (GitHub) |
| `pitch` | `pitch_stats` | pyworld |
| `dnsmos` | `dnsmos_score`, `utmos_score` | speechmos, onnxruntime |
| `quality` | `audio_fingerprint_dedup` | simhash |
| `diarize` | `pyannote_diarize` | pyannote.audio (needs `HF_TOKEN`) |
| `classify` | `speechbrain_langid` | speechbrain |
| `gender` | `gender_classify` (ina method — opt-in only; conflicts with ASR/TTS stacks) | inaSpeechSegmenter |
| `speaker` | `speaker_embed` | wespeaker (GitHub) |
| `enhance` | `speech_enhance` | deepfilternet |
| `align` | `forced_align`, `qwen3_asr` | qwen-asr |
| `codec` | `codec_tokenize` | encodec, descript-audio-codec |
| `tts-kokoro` | `tts_kokoro` | kokoro, misaki |
| `tts-chattts` | `tts_chattts` | ChatTTS |
| `tts-cosyvoice` | `tts_cosyvoice` | modelscope |
| `tts-fish-speech` | `tts_fish_speech` (operator currently parked; tracks fish-speech 1.x API) | fish-speech (GitHub) |
| `pack` | `pack_huggingface`, `pack_webdataset`, `pack_parquet` | datasets, webdataset, pyarrow |
| `viz` | HTML report | jinja2, plotly |
| `viz-panel` | Gradio panel | gradio |

GPU note for pip: install PyTorch for your CUDA version **first**, then
extras:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[asr,pitch]"
```

</details>

<details>
<summary>Docker without the wrapper — for CI, k8s, or image-only usage</summary>

If you pulled the image without cloning this repo (e.g. a CI job against
a registry tag), you need to spell out what the wrapper was doing:

```bash
docker run --rm --gpus all \
    --user $(id -u):$(id -g) -e HOME=/tmp \
    --env-file .env \
    -v $(pwd)/work:/app/work \
    -v $(pwd)/data:/data \
    ghcr.io/xqfeng-josie/voxkitchen:latest run pipeline.yaml
```

`--user $(id -u):$(id -g)` + `-e HOME=/tmp` so container-written files
end up with your UID. Without it, `./work` fills with root-owned files
you can't delete without sudo. Recovery in that case:

```bash
docker run --rm -v $(pwd)/work:/work alpine \
    chown -R $(id -u):$(id -g) /work
```

</details>

<details>
<summary>Building Docker images locally (air-gapped hosts, custom CUDA, arm64)</summary>

```bash
# Via vkit CLI (reads HF_TOKEN from ./.env automatically):
vkit docker build latest         # → voxkitchen:latest (~25 GB)
vkit docker build slim           # → voxkitchen:slim   (~3 GB)

# Or raw:
docker build --target latest -f docker/Dockerfile -t voxkitchen:latest .
docker build --target latest -f docker/Dockerfile \
    --build-arg HF_TOKEN=hf_xxx -t voxkitchen:latest .

# Or via the shell wrappers (same effect, no pip install needed):
scripts/vkit-build.sh latest
scripts/vkit-build.sh slim
```

Full build docs: [`docs/docker-build.md`](docs/docker-build.md).

</details>

## Configuration

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
vkit doctor                         Report operator availability + model cache
vkit viz <path>                     Gradio interactive explorer
```

## Python tools API

For quick tasks without a YAML pipeline, or to embed VoxKitchen
functions in your own Python code. Needs [pip install](#pip--fast-local-iteration-narrow-pipelines-library-embedding)
with the specific extras each call uses:

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
