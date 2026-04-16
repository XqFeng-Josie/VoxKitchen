# TTS Data Preparation

Prepare high-quality TTS training data from raw audio recordings.

## Quick Start

```bash
pip install voxkitchen[audio,enhance,align]
vkit init my-tts-project --template tts
cd my-tts-project
# Put your audio files in ./data/
vkit run pipeline.yaml
```

## What the Pipeline Does

| Stage | Operator | Why |
|-------|----------|-----|
| Format convert | `ffmpeg_convert` | Handle any input format (opus, flac, mp3, etc.) |
| Resample | `resample` → 22.05kHz mono | TTS standard sample rate; mono for single-speaker |
| Denoise | `speech_enhance` (aggressiveness=0.3) | Light denoising — preserve natural speech quality |
| VAD | `silero_vad` (min 1s) | Split into utterances; 1s minimum for meaningful TTS segments |
| SNR | `snr_estimate` | Measure signal-to-noise ratio |
| Filter | `quality_score_filter` | Keep only 2–15s segments with SNR > 15 dB |
| ASR + Align | `qwen3_asr` (timestamps=true) | Transcribe + word-level forced alignment in one pass |
| Pack | `pack_jsonl` | Output manifest with text, timestamps, quality metrics |

## Key Design Decisions

### Why 22.05 kHz?

Most TTS systems (VITS, Tacotron2, FastSpeech2) use 22.05 kHz. Using 16 kHz loses high-frequency detail that matters for speech naturalness. 44.1/48 kHz is unnecessarily large.

### Why light denoising (0.3)?

Aggressive denoising removes subtle speech nuances (breath, lip sounds) that make TTS output sound natural. For TTS, clean-but-natural is better than sterile-but-artificial. If your source audio is very noisy, increase to 0.5.

### Why 2–15 seconds?

- **< 2s**: Too short for TTS models to learn prosody patterns
- **> 15s**: Attention-based TTS models struggle with long sequences
- Sweet spot for most TTS architectures: **5–10 seconds**

### Why SNR > 15 dB?

TTS training is more sensitive to noise than ASR training. The model learns to reproduce whatever is in the audio, including noise. SNR 15 dB means the speech is ~30x louder than noise — clean enough for high-quality synthesis.

## Customization

### For Chinese TTS

Change the ASR language detection to Chinese for better accuracy:

```yaml
  - name: asr
    op: qwen3_asr
    args:
      model: Qwen/Qwen3-ASR-0.6B
      language: Chinese
      return_timestamps: true
```

### For multi-speaker TTS

Add speaker diarization before packing:

```yaml
  - name: diarize
    op: pyannote_diarize
    args:
      model: pyannote/speaker-diarization-3.1
```

### Adjust quality thresholds

```yaml
  # Stricter (fewer but cleaner segments)
  conditions:
    - "duration > 3"
    - "duration < 12"
    - "metrics.snr > 20"

  # Looser (more data, tolerate some noise)
  conditions:
    - "duration > 1"
    - "duration < 20"
    - "metrics.snr > 10"
```

## Output Format

The final `pack_jsonl` stage produces a JSONL manifest where each line is a JSON object:

```json
{
  "id": "segment_001__enhanced",
  "duration": 5.2,
  "text": "这是一段示例语音",
  "word_alignments": [
    {"text": "这", "start": 0.12, "end": 0.35},
    {"text": "是", "start": 0.35, "end": 0.52},
    ...
  ],
  "metrics": {"snr": 22.5}
}
```
