# Operator Reference

VoxKitchen ships with **51 built-in operators** across 8 categories.

!!! tip
    Run `vkit operators` to see this list in your terminal, or `vkit operators show <name>` for details.

## Categories

- [Audio Processing](#basic) (4 operators)
- [Segmentation](#segment) (4 operators)
- [Data Augmentation](#augment) (4 operators)
- [Annotation](#annotate) (17 operators)
- [Quality & Filtering](#quality) (11 operators)
- [Output / Packing](#pack) (6 operators)
- [Utility](#noop) (1 operators)
- [Synthesize](#synthesize) (4 operators)

## Audio Processing { #basic }

### `channel_merge`

**Merge multi-channel audio into mono or a specified number of channels.**

- **Device:** cpu
- **Install:** `pip install voxkitchen[core]`
- **Produces audio:** Yes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_channels` | int | `1` |  |

```yaml
- name: my_channel_merge
  op: channel_merge
  args:
    target_channels: 1
```

---

### `ffmpeg_convert`

**Convert audio format using ffmpeg (e.g. opus to wav, flac to mp3).**

- **Device:** cpu
- **Install:** `pip install voxkitchen[core]`
- **Produces audio:** Yes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_format` | str | `wav` |  |
| `clean_names` | bool | `True` |  |

```yaml
- name: my_ffmpeg_convert
  op: ffmpeg_convert
  args:
    target_format: wav
    clean_names: True
```

---

### `loudness_normalize`

**Normalize audio loudness to a target LUFS level (EBU R 128).**

- **Device:** cpu
- **Install:** `pip install voxkitchen[core]`
- **Produces audio:** Yes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_lufs` | float | `-23.0` |  |

```yaml
- name: my_loudness_normalize
  op: loudness_normalize
  args:
    target_lufs: -23.0
```

---

### `resample`

**Resample audio to a target sample rate and channel count.**

- **Device:** cpu
- **Install:** `pip install voxkitchen[core]`
- **Produces audio:** Yes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_sr` | int | required |  |
| `target_channels` | int | None | `None` |  |

```yaml
- name: my_resample
  op: resample
  args:
    target_sr: <int>
    target_channels: None
```

---

## Segmentation { #segment }

### `fixed_segment`

**Split each input Cut into fixed-length child Cuts.**

This is a 1-to-many operator: one Cut in, N Cuts out. Each child shares
the parent's ``recording`` and ``recording_id`` — no new audio is written.
The child's ``start`` is offset within the parent's audio file, so playback
of ``child.recording`` from ``child.start`` for ``child.duration`` seconds
yields the correct audio slice.

- **Device:** cpu
- **Install:** `pip install voxkitchen[core]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `segment_duration` | float | `10.0` |  |
| `min_remaining` | float | `0.5` |  |

```yaml
- name: my_fixed_segment
  op: fixed_segment
  args:
    segment_duration: 10.0
    min_remaining: 0.5
```

---

### `silence_split`

**Split each Cut on silent regions using librosa.effects.split.**

Returns one child Cut per non-silent interval.  No new audio is written.

- **Device:** cpu
- **Install:** `pip install voxkitchen[segment]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_db` | int | `30` |  |
| `min_duration` | float | `0.1` |  |

```yaml
- name: my_silence_split
  op: silence_split
  args:
    top_db: 30
    min_duration: 0.1
```

---

### `silero_vad`

**Detect speech regions using Silero VAD and emit one child Cut per region.**

Loads the Silero VAD model via torch.hub (cached after first download).
Works on both GPU and CPU. Requires network on first run to download
the model (~2 MB). Use ``webrtc_vad`` or ``silence_split`` if torch
is not available.

- **Device:** gpu
- **Install:** `pip install voxkitchen[segment]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | `0.5` |  |
| `min_speech_duration_ms` | int | `250` |  |
| `min_silence_duration_ms` | int | `100` |  |
| `speech_pad_ms` | int | `30` |  |

```yaml
- name: my_silero_vad
  op: silero_vad
  args:
    threshold: 0.5
    min_speech_duration_ms: 250
    min_silence_duration_ms: 100
    speech_pad_ms: 30
```

---

### `webrtc_vad`

**Detect speech regions using webrtcvad and emit one child Cut per region.**

Reads audio bytes from the parent Cut, runs frame-by-frame VAD, merges
consecutive speech frames, applies minimum-duration and padding, then
creates child Cuts for each speech region.  No new audio is written.

- **Device:** cpu
- **Install:** `pip install voxkitchen[segment]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `aggressiveness` | int | `2` |  |
| `frame_duration_ms` | int | `30` |  |
| `min_speech_duration_ms` | int | `250` |  |
| `padding_ms` | int | `30` |  |

```yaml
- name: my_webrtc_vad
  op: webrtc_vad
  args:
    aggressiveness: 2
    frame_duration_ms: 30
    min_speech_duration_ms: 250
    padding_ms: 30
```

---

## Data Augmentation { #augment }

### `noise_augment`

**Mix audio with random noise files at a random SNR.**

- **Device:** cpu
- **Install:** `pip install voxkitchen[core]`
- **Produces audio:** Yes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `noise_dir` | str | required |  |
| `snr_range` | list[float] | `[5.0, 20.0]` |  |

```yaml
- name: my_noise_augment
  op: noise_augment
  args:
    noise_dir: <str>
    snr_range: [5.0, 20.0]
```

---

### `reverb_augment`

**Add room reverb by convolving with Room Impulse Response files.**

- **Device:** cpu
- **Install:** `pip install voxkitchen[core]`
- **Produces audio:** Yes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rir_dir` | str | required |  |
| `normalize` | bool | `True` |  |

```yaml
- name: my_reverb_augment
  op: reverb_augment
  args:
    rir_dir: <str>
    normalize: True
```

---

### `speed_perturb`

**Apply speed perturbation (tempo + pitch change) via resampling.**

- **Device:** cpu
- **Install:** `pip install voxkitchen[audio]`
- **Produces audio:** Yes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `factors` | list[float] | `[0.9, 1.0, 1.1]` |  |

```yaml
- name: my_speed_perturb
  op: speed_perturb
  args:
    factors: [0.9, 1.0, 1.1]
```

---

### `volume_perturb`

**Apply random volume gain within a dB range.**

- **Device:** cpu
- **Install:** `pip install voxkitchen[core]`
- **Produces audio:** Yes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_gain_db` | float | `-6.0` |  |
| `max_gain_db` | float | `6.0` |  |

```yaml
- name: my_volume_perturb
  op: volume_perturb
  args:
    min_gain_db: -6.0
    max_gain_db: 6.0
```

---

## Annotation { #annotate }

### `codec_tokenize`

**Encode audio into discrete codec tokens (EnCodec / DAC).**

- **Device:** gpu
- **Install:** `pip install voxkitchen[codec]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | str | `encodec` |  |
| `bandwidth` | float | `6.0` |  |
| `model` | str | `encodec_24khz` |  |

```yaml
- name: my_codec_tokenize
  op: codec_tokenize
  args:
    backend: encodec
    bandwidth: 6.0
    model: encodec_24khz
```

---

### `emotion_recognize`

**Recognize speech emotions using emotion2vec (9 classes: angry, happy, sad, ...).**

- **Device:** gpu
- **Install:** `pip install voxkitchen[funasr]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `iic/emotion2vec_plus_large` |  |
| `granularity` | str | `utterance` |  |

```yaml
- name: my_emotion_recognize
  op: emotion_recognize
  args:
    model: iic/emotion2vec_plus_large
    granularity: utterance
```

---

### `faster_whisper_asr`

**Transcribe audio using faster-whisper and add Supervisions with text + language.**

Uses CTranslate2 for inference. Has GPU and CPU, but on CPU the
compute_type is coerced to "int8" because float16 is not supported.

.. warning::
CTranslate2 may deadlock on macOS ARM64. Use ``whisper_openai_asr``
on macOS instead.

- **Device:** gpu
- **Install:** `pip install voxkitchen[asr]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `tiny` |  |
| `language` | str | None | `None` |  |
| `beam_size` | int | `5` |  |
| `compute_type` | str | `int8` |  |
| `cpu_threads` | int | `4` |  |

```yaml
- name: my_faster_whisper_asr
  op: faster_whisper_asr
  args:
    model: tiny
    language: None
    beam_size: 5
    compute_type: int8
    cpu_threads: 4
```

---

### `forced_align`

**Align text to audio at word level using Qwen3-ForcedAligner (11 languages).**

- **Device:** gpu
- **Install:** `pip install voxkitchen[align]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `Qwen/Qwen3-ForcedAligner-0.6B` |  |
| `language` | str | `Chinese` |  |

```yaml
- name: my_forced_align
  op: forced_align
  args:
    model: Qwen/Qwen3-ForcedAligner-0.6B
    language: Chinese
```

---

### `gender_classify`

**Classify speaker gender using one of several methods.**

Methods:

- ``f0``: Extract fundamental frequency via librosa's pyin. Male if
median F0 < threshold (default 165 Hz), else female. Fast, no
model download, but only ~80-85% accurate on clean adult speech.
Fails on children, elderly, or noisy audio.

- ``speechbrain``: Use a SpeechBrain EncoderClassifier. More accurate
(~95%+) but requires model download. Default model is a speaker
recognition model (placeholder) — override ``speechbrain_model``
with a true gender classifier for best results.

- ``inaspeechsegmenter``: Use INA's speech segmenter which jointly
detects speech/music/noise and classifies gender. Well-tested in
broadcast media analysis (~90-95%). Requires ``pip install
inaSpeechSegmenter`` (uses TensorFlow).

- **Device:** cpu
- **Install:** `pip install voxkitchen[core]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | `f0` |  |
| `f0_threshold` | float | `165.0` |  |
| `speechbrain_model` | str | `speechbrain/spkrec-ecapa-voxceleb` |  |

```yaml
- name: my_gender_classify
  op: gender_classify
  args:
    method: f0
    f0_threshold: 165.0
    speechbrain_model: speechbrain/spkrec-ecapa-voxceleb
```

---

### `mel_extract`

**Extract mel spectrogram and save as .npy file per cut.**

- **Device:** cpu
- **Install:** `pip install voxkitchen[core]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_fft` | int | `1024` |  |
| `hop_length` | int | `256` |  |
| `n_mels` | int | `80` |  |
| `fmin` | float | `0.0` |  |
| `fmax` | float | None | `8000.0` |  |
| `ref_db` | float | `20.0` |  |
| `output_dir` | str | None | `None` |  |

```yaml
- name: my_mel_extract
  op: mel_extract
  args:
    n_fft: 1024
    hop_length: 256
    n_mels: 80
    fmin: 0.0
    fmax: 8000.0
    ref_db: 20.0
    output_dir: None
```

---

### `paraformer_asr`

**Transcribe audio using Paraformer (FunASR).**

The default model includes built-in VAD and punctuation restoration,
making it suitable for long-form audio without pre-segmentation.
Much faster than Whisper for Chinese.

- **Device:** gpu
- **Install:** `pip install voxkitchen[funasr]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch` |  |
| `language` | str | `zh` |  |

```yaml
- name: my_paraformer_asr
  op: paraformer_asr
  args:
    model: iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch
    language: zh
```

---

### `pyannote_diarize`

**Add speaker-label Supervisions to each Cut using pyannote.audio.**

Requires accepting the pyannote model user agreement on HuggingFace and
setting ``HF_TOKEN`` (or passing ``hf_token`` in the config).

- **Device:** gpu
- **Install:** `pip install voxkitchen[diarize]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `pyannote/speaker-diarization-3.1` |  |
| `min_speakers` | int | None | `None` |  |
| `max_speakers` | int | None | `None` |  |
| `hf_token` | str | None | `None` |  |

```yaml
- name: my_pyannote_diarize
  op: pyannote_diarize
  args:
    model: pyannote/speaker-diarization-3.1
    min_speakers: None
    max_speakers: None
    hf_token: None
```

---

### `qwen3_asr`

**Transcribe audio using Qwen3-ASR.**

30 languages + 22 Chinese dialects. Set ``return_timestamps=True``
to also get word-level timestamps (uses ForcedAligner internally).

- **Device:** gpu
- **Install:** `pip install voxkitchen[align]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `Qwen/Qwen3-ASR-0.6B` |  |
| `language` | str | None | `None` |  |
| `return_timestamps` | bool | `False` |  |
| `aligner_model` | str | `Qwen/Qwen3-ForcedAligner-0.6B` |  |
| `max_new_tokens` | int | `512` |  |

```yaml
- name: my_qwen3_asr
  op: qwen3_asr
  args:
    model: Qwen/Qwen3-ASR-0.6B
    language: None
    return_timestamps: False
    aligner_model: Qwen/Qwen3-ForcedAligner-0.6B
    max_new_tokens: 512
```

---

### `sensevoice_asr`

**Transcribe audio using SenseVoice (FunASR).**

SenseVoice supports Chinese, English, Japanese, Korean, and Cantonese.
The ``SenseVoiceSmall`` model is fast and accurate for these languages.

- **Device:** gpu
- **Install:** `pip install voxkitchen[funasr]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `iic/SenseVoiceSmall` |  |
| `language` | str | `auto` |  |

```yaml
- name: my_sensevoice_asr
  op: sensevoice_asr
  args:
    model: iic/SenseVoiceSmall
    language: auto
```

---

### `speaker_embed`

**Extract speaker embedding vectors using WeSpeaker or SpeechBrain.**

- **Device:** gpu
- **Install:** `pip install voxkitchen[speaker]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | `wespeaker` |  |
| `wespeaker_model` | str | `english` |  |
| `speechbrain_model` | str | `speechbrain/spkrec-ecapa-voxceleb` |  |

```yaml
- name: my_speaker_embed
  op: speaker_embed
  args:
    method: wespeaker
    wespeaker_model: english
    speechbrain_model: speechbrain/spkrec-ecapa-voxceleb
```

---

### `speech_enhance`

**Remove background noise using DeepFilterNet neural denoiser.**

- **Device:** cpu
- **Install:** `pip install voxkitchen[enhance]`
- **Produces audio:** Yes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | `deepfilternet` |  |
| `aggressiveness` | float | `0.5` |  |

```yaml
- name: my_speech_enhance
  op: speech_enhance
  args:
    method: deepfilternet
    aggressiveness: 0.5
```

---

### `speechbrain_langid`

**Add a language-identification Supervision to each Cut using SpeechBrain.**

Uses the VoxLingua107 ECAPA-TDNN model by default. Runs on CPU with
automatic fallback from CUDA.

- **Device:** gpu
- **Install:** `pip install voxkitchen[classify]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `speechbrain/lang-id-voxlingua107-ecapa` |  |

```yaml
- name: my_speechbrain_langid
  op: speechbrain_langid
  args:
    model: speechbrain/lang-id-voxlingua107-ecapa
```

---

### `wenet_asr`

**Transcribe audio using WeNet.**

WeNet supports streaming and non-streaming decoding. This operator
uses non-streaming (offline) mode for best accuracy.

- **Device:** gpu
- **Install:** `pip install voxkitchen[wenet]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `chinese` |  |
| `language` | str | `zh` |  |

```yaml
- name: my_wenet_asr
  op: wenet_asr
  args:
    model: chinese
    language: zh
```

---

### `whisper_langid`

**Detect the spoken language of each cut using Whisper.**

Adds a Supervision with the detected ``language`` code (e.g., "en",
"zh", "ja"). Uses only the first 30 seconds for detection — fast
even on long recordings.

Backend selection (``backend`` config):
- ``auto``: prefer faster-whisper, fall back to openai-whisper
- ``openai``: use openai-whisper (macOS-safe)
- ``faster-whisper``: use faster-whisper (faster on GPU)

- **Device:** gpu
- **Install:** `pip install voxkitchen[whisper]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `tiny` |  |
| `backend` | str | `auto` |  |

```yaml
- name: my_whisper_langid
  op: whisper_langid
  args:
    model: tiny
    backend: auto
```

---

### `whisper_openai_asr`

**Transcribe audio using OpenAI's official whisper (pure PyTorch).**

Works on both CPU and GPU. On CPU, set ``fp16: false``. Auto-detects
CUDA and falls back to CPU transparently.

This is the recommended ASR operator for macOS where CTranslate2-based
operators (faster_whisper_asr) may deadlock.

- **Device:** gpu
- **Install:** `pip install voxkitchen[whisper]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `tiny` |  |
| `language` | str | None | `None` |  |
| `beam_size` | int | `5` |  |
| `fp16` | bool | `True` |  |

```yaml
- name: my_whisper_openai_asr
  op: whisper_openai_asr
  args:
    model: tiny
    language: None
    beam_size: 5
    fp16: True
```

---

### `whisperx_asr`

**Transcribe audio with word-level alignment using whisperx.**

If whisperx is not installed, falls back to faster-whisper at segment level
(no word alignment). Either path requires the ``asr`` extras group.

- **Device:** gpu
- **Install:** `pip install voxkitchen[asr]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `tiny` |  |
| `language` | str | None | `None` |  |
| `batch_size` | int | `8` |  |
| `compute_type` | str | `int8` |  |

```yaml
- name: my_whisperx_asr
  op: whisperx_asr
  args:
    model: tiny
    language: None
    batch_size: 8
    compute_type: int8
```

---

## Quality & Filtering { #quality }

### `audio_fingerprint_dedup`

**Remove near-duplicate cuts using MFCC mean features + simhash.**

For each cut, a 13-coefficient MFCC mean vector is computed and hashed
with simhash.  Cuts whose hash is within ``similarity_threshold`` bits
(hamming distance) of any previously seen hash are dropped as duplicates.

- **Device:** cpu
- **Install:** `pip install voxkitchen[segment,quality]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `similarity_threshold` | int | `3` |  |

```yaml
- name: my_audio_fingerprint_dedup
  op: audio_fingerprint_dedup
  args:
    similarity_threshold: 3
```

---

### `bandwidth_estimate`

**Estimate effective audio bandwidth and store in metrics.**

Detects files that were upsampled from a lower sample rate — e.g., an
8 kHz telephone recording saved as 48 kHz WAV will show
``bandwidth_khz ≈ 4.0``.

Computes STFT, measures mean power per frequency bin, then finds the
frequency where energy drops sharply (ratio method). Writes:
- ``metrics["bandwidth_khz"]``: effective bandwidth in kHz

- **Device:** cpu
- **Install:** `pip install voxkitchen[core]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nfft` | int | `512` |  |
| `hop` | int | `256` |  |

```yaml
- name: my_bandwidth_estimate
  op: bandwidth_estimate
  args:
    nfft: 512
    hop: 256
```

---

### `cer_wer`

**Compute CER and WER between ASR output and reference text.**

- **Device:** cpu
- **Install:** `pip install voxkitchen[core]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hypothesis_field` | str | `text` |  |
| `reference_field` | str | `reference_text` |  |

```yaml
- name: my_cer_wer
  op: cer_wer
  args:
    hypothesis_field: text
    reference_field: reference_text
```

---

### `clipping_detect`

**Detect audio clipping and store the ratio of clipped samples.**

Clipping occurs when recording levels are too high, causing the
waveform to be truncated at the maximum amplitude ceiling. This
produces harsh distortion that degrades ASR and TTS training.

Writes ``metrics["clipping_ratio"]`` — fraction of samples whose
absolute value exceeds ``ceiling`` (default 0.99). A ratio > 0.01
indicates significant clipping.

- **Device:** cpu
- **Install:** `pip install voxkitchen[core]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ceiling` | float | `0.99` |  |

```yaml
- name: my_clipping_detect
  op: clipping_detect
  args:
    ceiling: 0.99
```

---

### `dnsmos_score`

**Score audio quality using Microsoft DNSMOS (no reference needed).**

Writes four metrics:
- ``dnsmos_ovrl`` — P.835 overall quality (1-5)
- ``dnsmos_sig`` — P.835 speech signal quality (1-5)
- ``dnsmos_bak`` — P.835 background noise quality (1-5)
- ``dnsmos_p808`` — P.808 overall MOS (1-5)

Higher is better. Typically ``dnsmos_ovrl > 3.0`` is considered
acceptable for training data.

- **Device:** cpu
- **Install:** `pip install voxkitchen[dnsmos]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_gpu` | bool | `False` |  |

```yaml
- name: my_dnsmos_score
  op: dnsmos_score
  args:
    use_gpu: False
```

---

### `duration_filter`

**Drop Cuts whose duration falls outside [min_duration, max_duration].**

This is an N-to-fewer operator: no audio is read or written.

- **Device:** cpu
- **Install:** `pip install voxkitchen[core]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_duration` | float | `0.0` |  |
| `max_duration` | float | None | `None` |  |

```yaml
- name: my_duration_filter
  op: duration_filter
  args:
    min_duration: 0.0
    max_duration: None
```

---

### `pitch_stats`

**Compute pitch (F0) statistics using PyWorld (dio + stonemask).**

More accurate than librosa.pyin for speech. Writes:
- ``metrics["pitch_mean"]`` — mean F0 in Hz (voiced frames only)
- ``metrics["pitch_std"]`` — normalized std (0-1 range, pitch-independent)

A ``pitch_mean`` of 0 means no voiced frames were detected.

- **Device:** cpu
- **Install:** `pip install voxkitchen[pitch]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `f0_min` | float | `50.0` |  |
| `f0_max` | float | `2400.0` |  |
| `frame_period_ms` | float | `5.0` |  |

```yaml
- name: my_pitch_stats
  op: pitch_stats
  args:
    f0_min: 50.0
    f0_max: 2400.0
    frame_period_ms: 5.0
```

---

### `quality_score_filter`

**Drop Cuts that do not satisfy all conditions.**

Each condition is a whitespace-separated triple ``field.path op value``
where ``op`` is one of ``>``, ``>=``, ``<``, ``<=``, ``==``, ``!=``.
All conditions are AND-ed together.

- **Device:** cpu
- **Install:** `pip install voxkitchen[core]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conditions` | list[str] | required |  |

```yaml
- name: my_quality_score_filter
  op: quality_score_filter
  args:
    conditions: <list[str]>
```

---

### `snr_estimate`

**Estimate SNR via a peak-to-RMS ratio and store it in cut.metrics["snr"].**

This is a rough proxy (not WADA-SNR or model-based) sufficient for v0.1.
No audio is written; only the metrics dict is updated.

- **Device:** cpu
- **Install:** `pip install voxkitchen[core]`
- **Produces audio:** No

```yaml
- name: my_snr_estimate
  op: snr_estimate
```

---

### `speaker_similarity`

**Score speaker similarity against a reference embedding (cosine).**

- **Device:** cpu
- **Install:** `pip install voxkitchen[core]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference_path` | str | required |  |
| `embedding_key` | str | `speaker_embedding` |  |

```yaml
- name: my_speaker_similarity
  op: speaker_similarity
  args:
    reference_path: <str>
    embedding_key: speaker_embedding
```

---

### `utmos_score`

**Predict speech naturalness MOS using UTMOS (no reference needed).**

Writes ``metrics["utmos"]`` — predicted MOS score (1-5).
Higher is better. Scores > 4.0 indicate natural-sounding speech.

Useful for filtering synthetic/degraded audio from training data.

- **Device:** cpu
- **Install:** `pip install voxkitchen[dnsmos]`
- **Produces audio:** No

```yaml
- name: my_utmos_score
  op: utmos_score
```

---

## Output / Packing { #pack }

### `pack_huggingface`

**Export CutSet as a HuggingFace Dataset with audio column.**

- **Device:** cpu
- **Install:** `pip install voxkitchen[pack]`
- **Produces audio:** Yes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | None | `None` |  |

```yaml
- name: my_pack_huggingface
  op: pack_huggingface
  args:
    output_dir: None
```

---

### `pack_jsonl`

**Write a flat JSONL manifest — one JSON object per line.**

Fields: id, origin_id, start, end, duration, sample_rate,
text, snr, gender (male/female/unknown), speaker, language.

``start``/``end`` are the VAD segment boundaries in the original
recording.  ``origin_id`` traces back to the source filename.

- **Device:** cpu
- **Install:** `pip install voxkitchen[core]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_path` | str | None | `None` |  |

```yaml
- name: my_pack_jsonl
  op: pack_jsonl
  args:
    output_path: None
```

---

### `pack_kaldi`

**Export CutSet in Kaldi format (wav.scp, text, utt2spk, spk2utt).**

- **Device:** cpu
- **Install:** `pip install voxkitchen[core]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | None | `None` |  |

```yaml
- name: my_pack_kaldi
  op: pack_kaldi
  args:
    output_dir: None
```

---

### `pack_manifest`

**Write a flat manifest (cuts.jsonl.gz) with no audio export.**

- **Device:** cpu
- **Install:** `pip install voxkitchen[core]`
- **Produces audio:** No

```yaml
- name: my_pack_manifest
  op: pack_manifest
```

---

### `pack_parquet`

**Export CutSet as Apache Parquet with audio file references.**

- **Device:** cpu
- **Install:** `pip install voxkitchen[pack]`
- **Produces audio:** No

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | None | `None` |  |

```yaml
- name: my_pack_parquet
  op: pack_parquet
  args:
    output_dir: None
```

---

### `pack_webdataset`

**Export CutSet as WebDataset tar shards with embedded audio.**

- **Device:** cpu
- **Install:** `pip install voxkitchen[pack]`
- **Produces audio:** Yes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | None | `None` |  |
| `shard_size` | int | `1000` |  |

```yaml
- name: my_pack_webdataset
  op: pack_webdataset
  args:
    output_dir: None
    shard_size: 1000
```

---

## Utility { #noop }

### `identity`

**Pass cuts through unchanged (no-op, useful for testing).**

- **Device:** cpu
- **Install:** `pip install voxkitchen[core]`
- **Produces audio:** No

```yaml
- name: my_identity
  op: identity
```

---

## Synthesize { #synthesize }

### `tts_chattts`

**Synthesize conversational speech using ChatTTS.**

- **Device:** gpu
- **Install:** `pip install voxkitchen[tts-chattts]`
- **Produces audio:** Yes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | None | `None` |  |
| `temperature` | float | `0.3` |  |
| `top_p` | float | `0.7` |  |
| `top_k` | int | `20` |  |

```yaml
- name: my_tts_chattts
  op: tts_chattts
  args:
    seed: None
    temperature: 0.3
    top_p: 0.7
    top_k: 20
```

---

### `tts_cosyvoice`

**Synthesize speech using CosyVoice2 with optional voice cloning.**

- **Device:** gpu
- **Install:** `pip install voxkitchen[tts-cosyvoice]`
- **Produces audio:** Yes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | str | `FunAudioLLM/CosyVoice2-0.5B` |  |
| `mode` | str | `sft` |  |
| `spk_id` | str | `default` |  |
| `reference_audio` | str | None | `None` |  |
| `reference_text` | str | None | `None` |  |

```yaml
- name: my_tts_cosyvoice
  op: tts_cosyvoice
  args:
    model_id: FunAudioLLM/CosyVoice2-0.5B
    mode: sft
    spk_id: default
    reference_audio: None
    reference_text: None
```

---

### `tts_fish_speech`

**Synthesize speech using Fish-Speech codec language model.**

- **Device:** gpu
- **Install:** `pip install voxkitchen[tts-fish-speech]`
- **Produces audio:** Yes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | str | `fishaudio/fish-speech-1.5` |  |
| `reference_audio` | str | None | `None` |  |
| `reference_text` | str | None | `None` |  |
| `max_new_tokens` | int | `1024` |  |
| `top_p` | float | `0.7` |  |
| `temperature` | float | `0.7` |  |
| `repetition_penalty` | float | `1.2` |  |

```yaml
- name: my_tts_fish_speech
  op: tts_fish_speech
  args:
    model_id: fishaudio/fish-speech-1.5
    reference_audio: None
    reference_text: None
    max_new_tokens: 1024
    top_p: 0.7
    temperature: 0.7
    repetition_penalty: 1.2
```

---

### `tts_kokoro`

**Synthesize speech from text using Kokoro TTS.**

- **Device:** cpu
- **Install:** `pip install voxkitchen[tts-kokoro]`
- **Produces audio:** Yes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `voice` | str | `af_heart` |  |
| `lang_code` | str | `a` |  |
| `speed` | float | `1.0` |  |

```yaml
- name: my_tts_kokoro
  op: tts_kokoro
  args:
    voice: af_heart
    lang_code: a
    speed: 1.0
```

---
