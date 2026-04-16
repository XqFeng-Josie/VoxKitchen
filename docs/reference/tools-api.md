# Python Tools API

For quick tasks without writing a YAML pipeline, use the standalone functions in `voxkitchen.tools`.

```python
from voxkitchen.tools import (
    audio_info,
    estimate_bandwidth,
    transcribe,
    detect_speech,
    classify_gender,
    estimate_snr,
    resample_audio,
    normalize_loudness,
    extract_speaker_embedding,
    enhance_speech,
    align_words,
)
```

## Audio Info

```python
info = audio_info("recording.wav")
# AudioInfo(path='recording.wav', sample_rate=16000, num_channels=1,
#           num_samples=160000, duration=10.0, format='WAV')

# Detect upsampled audio
info = audio_info("suspicious.wav", estimate_real_sr=True)
print(f"Header: {info.sample_rate} Hz, real: {info.real_sample_rate} Hz")
```

## ASR Transcription

```python
segments = transcribe("speech.wav", model="tiny")
# [SpeechSegment(start=0.0, end=3.2, text="Hello world")]

# Different engines
segments = transcribe("speech.wav", engine="sensevoice")     # Chinese/multilingual
segments = transcribe("speech.wav", engine="paraformer")      # Chinese, fast
segments = transcribe("speech.wav", engine="wenet", model="chinese")
```

## Speech Detection (VAD)

```python
segments = detect_speech("recording.wav", method="silero")
# [SpeechSegment(start=0.5, end=2.8), SpeechSegment(start=4.1, end=6.3)]

# Lightweight alternative
segments = detect_speech("recording.wav", method="webrtc")
```

## Quality Estimation

```python
snr = estimate_snr("noisy.wav")
# 18.3 (dB)

bandwidth = estimate_bandwidth("maybe_upsampled.wav")
# 8000.0 (Hz — real content is 8kHz despite 16kHz header)
```

## Gender Classification

```python
result = classify_gender("speaker.wav")
# {"gender": "f", "median_f0": 210.5, "method": "f0"}

result = classify_gender("speaker.wav", method="speechbrain")
# {"gender": "m", "method": "speechbrain", ...}
```

## Audio Processing

```python
resample_audio("input.wav", "output_16k.wav", target_sr=16000)
normalize_loudness("loud.wav", "normalized.wav", target_lufs=-23.0)
```

## Speaker Embedding

```python
emb = extract_speaker_embedding("speaker.wav")
# [0.12, -0.34, 0.56, ...]  (512-d vector)

# SpeechBrain backend
emb = extract_speaker_embedding("speaker.wav", method="speechbrain",
                                 model="speechbrain/spkrec-ecapa-voxceleb")
```

Requires: `pip install voxkitchen[speaker]`

## Speech Enhancement

```python
enhance_speech("noisy.wav", "clean.wav", aggressiveness=0.5)
```

Requires: `pip install voxkitchen[enhance]`

## Forced Alignment

```python
words = align_words("speech.wav", "hello world", language="English")
# [{"text": "hello", "start": 0.12, "end": 0.58},
#  {"text": "world", "start": 0.62, "end": 1.15}]

# Chinese
words = align_words("speech.wav", "你好世界", language="Chinese")
```

Requires: `pip install voxkitchen[align]`
