# VoxKitchen Operator Reference

All 51 operators across 7 categories. Each entry shows: name, device, produces_audio,
required pip extras, and key config fields. Use `vkit operators show <name>` to see
the full Pydantic config schema and a YAML example.

---

## Table of Contents
1. [Audio (5)](#audio)
2. [Segmentation (4)](#segmentation)
3. [Augmentation (4)](#augmentation)
4. [Annotation (17)](#annotation)
5. [Quality (11)](#quality)
6. [Synthesize (4)](#synthesize)
7. [Pack (6)](#pack)

---

## Audio

### `resample`
- **Device:** cpu | **Produces audio:** yes | **Extras:** audio
- Resample audio to target sample rate and/or channel count.
- **Args:** `target_sr` (int, required), `target_channels` (1 or 2, default: 1)

### `ffmpeg_convert`
- **Device:** cpu | **Produces audio:** yes | **Extras:** audio
- Convert audio format using ffmpeg.
- **Args:** `target_format` (str, e.g. "wav", "flac", "mp3"), `bitrate` (str, optional, e.g. "128k")

### `channel_merge`
- **Device:** cpu | **Produces audio:** yes | **Extras:** audio
- Merge multi-channel audio to mono by averaging.
- **Args:** none

### `loudness_normalize`
- **Device:** cpu | **Produces audio:** yes | **Extras:** audio
- Normalize loudness to target LUFS.
- **Args:** `target_lufs` (float, default: -23.0), `true_peak` (float, default: -1.0)

### `identity`
- **Device:** cpu | **Produces audio:** no | **Extras:** none
- No-op pass-through. Useful for testing or as a placeholder.
- **Args:** none

---

## Segmentation

### `silero_vad`
- **Device:** cpu | **Produces audio:** yes | **Extras:** segment
- Voice activity detection using Silero VAD. Splits recordings into speech segments.
  Best general-purpose VAD — robust and accurate.
- **Args:**
  - `threshold` (float, default: 0.5) — speech probability threshold
  - `min_speech_duration_ms` (int, default: 250)
  - `min_silence_duration_ms` (int, default: 100)
  - `max_speech_duration_s` (float, default: 30.0) — split long segments
  - `window_size_samples` (int, default: 512)

### `webrtc_vad`
- **Device:** cpu | **Produces audio:** yes | **Extras:** segment
- WebRTC-based VAD. Faster than silero but less accurate.
- **Args:**
  - `aggressiveness` (0–3, default: 2) — higher = more aggressive filtering
  - `min_speech_duration_ms` (int, default: 300)

### `fixed_segment`
- **Device:** cpu | **Produces audio:** yes | **Extras:** audio
- Split recordings into fixed-length chunks (ignores silence).
- **Args:**
  - `segment_duration_s` (float, required) — length of each chunk
  - `min_segment_duration_s` (float, default: 0.5) — skip shorter trailing chunks

### `silence_split`
- **Device:** cpu | **Produces audio:** yes | **Extras:** audio
- Split on silence using amplitude threshold.
- **Args:**
  - `silence_threshold_db` (float, default: -40.0)
  - `min_silence_duration_ms` (int, default: 500)
  - `min_segment_duration_ms` (int, default: 1000)

---

## Augmentation

### `speed_perturb`
- **Device:** cpu | **Produces audio:** yes | **Extras:** audio
- Apply speed perturbation. Multiplies each cut by N factors (increases dataset size).
- **Args:**
  - `factors` (list[float], default: [0.9, 1.0, 1.1])
  - `keep_original` (bool, default: true)

### `volume_perturb`
- **Device:** cpu | **Produces audio:** yes | **Extras:** audio
- Apply random volume gain.
- **Args:**
  - `min_gain_db` (float, default: -6.0)
  - `max_gain_db` (float, default: 6.0)
  - `randomize_per_cut` (bool, default: true)

### `noise_augment`
- **Device:** cpu | **Produces audio:** yes | **Extras:** audio
- Add background noise from a noise dataset.
- **Args:**
  - `noise_dir` (str, required) — path to noise audio files
  - `snr_db_range` ([float, float], default: [10.0, 30.0])

### `reverb_augment`
- **Device:** cpu | **Produces audio:** yes | **Extras:** audio
- Apply room impulse response (RIR) reverb.
- **Args:**
  - `rir_dir` (str, required) — path to RIR audio files
  - `p` (float, default: 0.5) — probability of applying reverb per cut

---

## Annotation

### ASR Operators

#### `faster_whisper_asr`
- **Device:** gpu | **Produces audio:** no | **Extras:** asr
- Whisper ASR via faster-whisper (CTranslate2). Good multilingual baseline.
- **Args:**
  - `model` (str, default: "large-v3") — e.g. tiny, base, small, medium, large-v3
  - `compute_type` (str, default: "float16") — float16, int8, float32
  - `language` (str, optional) — force language code (e.g. "zh", "en")
  - `beam_size` (int, default: 5)
  - `return_timestamps` (bool, default: true)
  - `batch_size` (int, default: 16)

#### `whisper_openai_asr`
- **Device:** gpu | **Produces audio:** no | **Extras:** whisper
- OpenAI's Whisper via the official Python package. Use on macOS (avoids deadlock).
- **Args:** `model` (str), `language` (str, optional), `fp16` (bool, default: true)

#### `whisperx_asr`
- **Device:** gpu | **Produces audio:** no | **Extras:** asr
- WhisperX with word-level timestamps and better batching.
- **Args:** `model` (str, default: "large-v3"), `compute_type` (str), `language` (str, optional), `batch_size` (int, default: 16)

#### `paraformer_asr`
- **Device:** gpu | **Produces audio:** no | **Extras:** funasr
- FunASR Paraformer — streaming-capable, strong Chinese ASR.
- **Args:** `model` (str, default: "paraformer-zh"), `hotwords` (list[str], optional)

#### `sensevoice_asr`
- **Device:** gpu | **Produces audio:** no | **Extras:** funasr
- FunASR SenseVoice — emotion + event detection + ASR in one pass.
- **Args:** `model` (str, default: "SenseVoiceSmall"), `language` (str, optional)

#### `wenet_asr`
- **Device:** gpu | **Produces audio:** no | **Extras:** wenet
- WeNet ASR — industrial-strength Chinese/English ASR.
- **Args:** `model_dir` (str, required), `mode` (str, default: "attention_rescoring")

#### `qwen3_asr`
- **Device:** gpu | **Produces audio:** no | **Extras:** asr
- Qwen3-ASR — Alibaba's latest model, best for Chinese + multilingual.
- **Args:**
  - `model` (str, default: "Qwen/Qwen3-ASR-0.6B") — also 2B, 7B variants
  - `return_timestamps` (bool, default: true)
  - `language` (str, optional)

### Diarization

#### `pyannote_diarize`
- **Device:** gpu | **Produces audio:** no | **Extras:** diarize
- Pyannote speaker diarization 3.1. Requires HF_TOKEN + model agreement.
- **Args:**
  - `num_speakers` (int, optional) — fix speaker count if known
  - `min_speakers` (int, optional)
  - `max_speakers` (int, optional)
  - `hf_token` (str, optional) — reads from HF_TOKEN env var by default

### Language / Gender / Emotion

#### `speechbrain_langid`
- **Device:** cpu | **Produces audio:** no | **Extras:** classify
- Language identification using SpeechBrain.
- **Args:** `threshold` (float, default: 0.7)

#### `whisper_langid`
- **Device:** gpu | **Produces audio:** no | **Extras:** asr
- Language ID using Whisper's language detection head.
- **Args:** `model` (str, default: "small")

#### `gender_classify`
- **Device:** cpu | **Produces audio:** no | **Extras:** classify
- Gender classification (male/female) stored in `supervision.gender`.
- **Args:** `threshold` (float, default: 0.6)

#### `emotion_recognize`
- **Device:** cpu | **Produces audio:** no | **Extras:** classify
- Emotion recognition (happy/sad/angry/neutral/...) stored in `custom.emotion`.
- **Args:** none

### Speaker / Embeddings

#### `speaker_embed`
- **Device:** cpu | **Produces audio:** no | **Extras:** speaker
- Extract speaker embedding vectors (stored in `custom.speaker_embedding`).
- **Args:** `model` (str, default: "wespeaker") — also supports "ecapa", "titanet"

### Processing

#### `speech_enhance`
- **Device:** cpu | **Produces audio:** yes | **Extras:** enhance
- Speech enhancement / noise reduction.
- **Args:** `aggressiveness` (float 0.0–1.0, default: 0.5) — higher = more aggressive denoising

#### `forced_align`
- **Device:** gpu | **Produces audio:** no | **Extras:** align
- Word-level forced alignment. Requires `supervision.text` to be set (run ASR first).
- **Args:** `model` (str, default: "wav2vec2") — language-specific model

#### `codec_tokenize`
- **Device:** gpu | **Produces audio:** no | **Extras:** codec
- Audio codec tokenization (EnCodec / DAC). Stores token IDs in `custom.codec_tokens`.
- **Args:** `codec` (str, default: "encodec") — "encodec" or "dac", `bandwidth` (float, default: 6.0)

#### `mel_extract`
- **Device:** cpu | **Produces audio:** no | **Extras:** audio
- Extract mel spectrogram features stored in `custom.mel`.
- **Args:** `n_mels` (int, default: 80), `hop_length` (int, default: 256), `win_length` (int, default: 1024)

---

## Quality

### `snr_estimate`
- **Device:** cpu | **Produces audio:** no | **Extras:** quality
- Estimate signal-to-noise ratio (dB). Stored in `metrics.snr`.
- **Args:** none

### `dnsmos_score`
- **Device:** cpu | **Produces audio:** no | **Extras:** dnsmos
- Microsoft DNSMOS speech quality score (0–5). Stored in `metrics.dnsmos`.
- **Args:** `personalized` (bool, default: false)

### `utmos_score`
- **Device:** gpu | **Produces audio:** no | **Extras:** quality
- UTMOS naturalness score (0–5, higher is better). Stored in `metrics.utmos`.
- **Args:** none

### `pitch_stats`
- **Device:** cpu | **Produces audio:** no | **Extras:** pitch
- Pitch analysis (F0, voiced ratio). Stored in `custom.pitch_*`.
- **Args:** `frame_shift_ms` (int, default: 10), `voiced_threshold` (float, default: 0.0)

### `clipping_detect`
- **Device:** cpu | **Produces audio:** no | **Extras:** quality
- Detect audio clipping. Stored in `metrics.clipping_ratio`.
- **Args:** `threshold` (float, default: 0.99) — amplitude threshold for clipping

### `bandwidth_estimate`
- **Device:** cpu | **Produces audio:** no | **Extras:** quality
- Estimate effective bandwidth (useful for detecting telephone-quality audio).
- Stored in `metrics.bandwidth_hz`.
- **Args:** none

### `duration_filter`
- **Device:** cpu | **Produces audio:** no | **Extras:** none
- Simple duration-based filter. Faster than `quality_score_filter` for pure duration filtering.
- **Args:** `min_duration` (float, optional), `max_duration` (float, optional)

### `audio_fingerprint_dedup`
- **Device:** cpu | **Produces audio:** no | **Extras:** quality
- Remove near-duplicate audio using fingerprinting.
- **Args:** `threshold` (float, default: 0.95) — similarity threshold

### `quality_score_filter`
- **Device:** cpu | **Produces audio:** no | **Extras:** none
- Filter cuts using Python-like condition strings. Most flexible quality gate.
- **Args:**
  - `conditions` (list[str], required) — ALL conditions must be true to keep the cut
  - Example conditions:
    - `"duration > 2"`
    - `"duration < 30"`
    - `"metrics.snr > 15"`
    - `"metrics.utmos > 3.5"`
    - `"metrics.dnsmos > 3.0"`
    - `"metrics.clipping_ratio < 0.01"`
    - `"supervision.text is not None"`

### `speaker_similarity`
- **Device:** cpu | **Produces audio:** no | **Extras:** speaker
- Compute cosine similarity between cut's speaker embedding and a reference.
  Stored in `metrics.speaker_similarity`. Requires `speaker_embed` to have run.
- **Args:** `reference_audio` (str, required)

### `cer_wer`
- **Device:** cpu | **Produces audio:** no | **Extras:** none
- Compute CER/WER between `supervision.text` and a reference transcript file.
  Stored in `metrics.cer` and `metrics.wer`.
- **Args:** `reference_manifest` (str, required)

---

## Synthesize

### `tts_kokoro`
- **Device:** cpu | **Produces audio:** yes | **Extras:** tts-kokoro
- Kokoro TTS — high-quality English TTS.
- **Args:**
  - `voice` (str, default: "af_heart") — see Kokoro voice list
  - `speed` (float, default: 1.0)
  - `text_field` (str, default: "supervision.text") — where to read text from

### `tts_chattts`
- **Device:** gpu | **Produces audio:** yes | **Extras:** tts-chattts
- ChatTTS — conversational-style Chinese/English TTS.
- **Args:** `temperature` (float, default: 0.3), `top_p` (float, default: 0.7)

### `tts_cosyvoice`
- **Device:** gpu | **Produces audio:** yes | **Extras:** tts-cosyvoice
- CosyVoice — Chinese TTS with voice cloning support.
- **Args:**
  - `mode` (str, default: "zero_shot") — "sft", "zero_shot", "cross_lingual", "instruct"
  - `reference_audio` (str, optional) — for zero_shot voice cloning
  - `reference_text` (str, optional)
  - `instruct_text` (str, optional)

### `tts_fish_speech`
- **Device:** gpu | **Produces audio:** yes | **Extras:** tts-fish-speech
- Fish-Speech — high-quality multilingual TTS. Uses torch 2.8 (isolated cluster).
  **Cannot be mixed with other extras.** Use Docker `:fish-speech` tag.
- **Args:** `reference_audio` (str, optional), `reference_text` (str, optional)

---

## Pack

### `pack_manifest`
- **Device:** cpu | **Produces audio:** no | **Extras:** pack
- Write the final CutSet as a `.jsonl.gz` manifest. Simple, lossless.
- **Args:** `output_path` (str, optional — defaults to stage dir)

### `pack_jsonl`
- **Device:** cpu | **Produces audio:** no | **Extras:** pack
- Same as pack_manifest but with explicit JSONL naming. Good for downstream tools.
- **Args:** `output_path` (str, optional)

### `pack_huggingface`
- **Device:** cpu | **Produces audio:** no | **Extras:** pack
- Write as HuggingFace `datasets` format. Loads directly with `load_from_disk()`.
- **Args:**
  - `output_dir` (str, optional)
  - `audio_column` (str, default: "audio")
  - `include_audio_bytes` (bool, default: true)

### `pack_webdataset`
- **Device:** cpu | **Produces audio:** no | **Extras:** pack
- Write as WebDataset tar shards. Good for large-scale streaming training.
- **Args:**
  - `output_dir` (str, optional)
  - `shard_size` (int, default: 1000) — cuts per shard
  - `audio_format` (str, default: "wav")

### `pack_parquet`
- **Device:** cpu | **Produces audio:** no | **Extras:** pack
- Write as Parquet files (audio bytes embedded as binary column).
- **Args:**
  - `output_dir` (str, optional)
  - `row_group_size` (int, default: 128)

### `pack_kaldi`
- **Device:** cpu | **Produces audio:** no | **Extras:** pack
- Write Kaldi-format data directory (wav.scp, text, utt2spk, segments).
- **Args:** `output_dir` (str, optional)
