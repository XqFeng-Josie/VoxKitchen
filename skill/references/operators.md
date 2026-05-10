# VoxKitchen Operator Selection Guide

Prefer live discovery when available:

```bash
vkit operators
vkit operators show <operator>
```

Inside the repository, `docs/reference/operators.md` is the authoritative
generated reference. This file is a compact selection guide for agents.

## Common Pipeline Shapes

### Cleaning

Use image `slim`.

```yaml
- name: resample
  op: resample
  args: { target_sr: 16000, target_channels: 1 }
- name: snr
  op: snr_estimate
- name: clipping
  op: clipping_detect
- name: filter
  op: quality_score_filter
  args:
    conditions:
      - "duration > 1"
      - "duration < 30"
      - "metrics.snr > 10"
      - "metrics.clipping_ratio < 0.01"
- name: pack
  op: pack_jsonl
```

### ASR Training Data

Use image `asr` unless `vkit validate` recommends another tag.

```yaml
- name: resample
  op: resample
  args: { target_sr: 16000, target_channels: 1 }
- name: vad
  op: silero_vad
  args: { threshold: 0.5 }
- name: asr
  op: faster_whisper_asr
  args: { model: large-v3, compute_type: float16 }
- name: filter
  op: quality_score_filter
  args:
    conditions: ["duration > 1", "duration < 30"]
- name: pack
  op: pack_huggingface
```

For Chinese ASR, consider `qwen3_asr`, `sensevoice_asr`, or `paraformer_asr`.

### Speaker Analysis

Use `latest` for the default template because it mixes diarization with other
annotation families. Use `diarize` for a pure pyannote pipeline.

```yaml
- name: resample
  op: resample
  args: { target_sr: 16000, target_channels: 1 }
- name: diarize
  op: pyannote_diarize
  args: { model: pyannote/speaker-diarization-3.1 }
- name: pack
  op: pack_jsonl
```

`pyannote_diarize` needs `HF_TOKEN` in `./.env` and accepted model terms on
HuggingFace.

### TTS Data Preparation

Use `latest` for templates that transcribe or align with ASR before packing.
Use `tts` only for pure synthesis/tts-family stages.

```yaml
- name: denoise
  op: speech_enhance
  args: { aggressiveness: 0.3 }
- name: vad
  op: silero_vad
- name: asr
  op: qwen3_asr
  args: { model: Qwen/Qwen3-ASR-0.6B, return_timestamps: true }
- name: pack
  op: pack_jsonl
```

## Operator Families

| Family | Operators to consider | Typical image |
|---|---|---|
| Audio prep | `resample`, `ffmpeg_convert`, `channel_merge`, `loudness_normalize` | `slim` |
| Segmentation | `silero_vad`, `webrtc_vad`, `fixed_segment`, `silence_split` | `slim` |
| Augmentation | `speed_perturb`, `volume_perturb`, `noise_augment`, `reverb_augment` | `slim` |
| ASR/alignment | `faster_whisper_asr`, `whisper_openai_asr`, `whisperx_asr`, `qwen3_asr`, `paraformer_asr`, `sensevoice_asr`, `wenet_asr`, `forced_align` | `asr` |
| Diarization | `pyannote_diarize` | `diarize` |
| Speaker/language | `speaker_embed`, `speechbrain_langid`, `whisper_langid`, `gender_classify` | `slim` or `asr` |
| Enhancement/features | `speech_enhance`, `codec_tokenize`, `mel_extract` | `slim` |
| Quality | `snr_estimate`, `dnsmos_score`, `utmos_score`, `pitch_stats`, `clipping_detect`, `bandwidth_estimate`, `duration_filter`, `quality_score_filter`, `audio_fingerprint_dedup`, `speaker_similarity`, `cer_wer` | `slim` |
| TTS | `tts_kokoro`, `tts_chattts`, `tts_cosyvoice`, `tts_fish_speech` | `tts` or `fish-speech` |
| Packing | `pack_manifest`, `pack_jsonl`, `pack_huggingface`, `pack_webdataset`, `pack_parquet`, `pack_kaldi` | `slim` |

## Selection Rules

- Use `silero_vad` as the default VAD. Use `webrtc_vad` only when speed and a
  simpler CPU path matter more than accuracy.
- Prefer `faster_whisper_asr` for general multilingual ASR, `qwen3_asr` or
  `paraformer_asr` for Chinese-heavy data, and `whisper_openai_asr` for macOS
  compatibility.
- Use `quality_score_filter` for most filtering because it can combine duration,
  SNR, clipping, MOS, language, and custom metrics.
- Choose `pack_jsonl` for simple manifests, `pack_huggingface` for HuggingFace
  training workflows, `pack_kaldi` for Kaldi-style training, and
  `pack_webdataset` for streaming at scale.
- Add `noise_augment` or `reverb_augment` only when the user already has noise or
  RIR files under `./data/...`.
- Always validate generated YAML with `vkit validate pipeline.yaml`.
