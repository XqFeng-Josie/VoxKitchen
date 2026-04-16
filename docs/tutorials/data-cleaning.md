# Data Cleaning

Clean up raw audio data: measure quality, remove duplicates, filter out bad files.

## Quick Start

```bash
pip install voxkitchen[audio,segment,quality]
vkit init my-cleaning-project --template cleaning
cd my-cleaning-project
# Put your raw audio files in ./data/
vkit run pipeline.yaml
vkit inspect cuts work/*/05_filter/cuts.jsonl.gz
```

## What the Pipeline Does

| Stage | Operator | What it measures/does |
|-------|----------|-----------------------|
| Resample | `resample` → 16kHz | Normalize all audio to the same format |
| SNR | `snr_estimate` | Signal-to-noise ratio (dB) |
| Clipping | `clipping_detect` | Ratio of clipped samples (0.0 = no clipping) |
| Bandwidth | `bandwidth_estimate` | Effective bandwidth in kHz (detects upsampled audio) |
| Dedup | `audio_fingerprint_dedup` | Remove near-duplicate audio via MFCC + SimHash |
| Filter | `quality_score_filter` | Drop files that fail quality thresholds |
| Pack | `pack_jsonl` | Output manifest with all quality metrics |

## Understanding the Metrics

### SNR (Signal-to-Noise Ratio)

| SNR | Quality | Typical source |
|:---:|---------|----------------|
| < 5 dB | Unusable | Street recording, heavy background |
| 5–15 dB | Noisy | Meeting room, casual recording |
| 15–25 dB | Clean | Studio with some ambient noise |
| > 25 dB | Very clean | Professional studio |

### Clipping Ratio

| Ratio | Meaning |
|:---:|---------|
| 0.0 | No clipping — good |
| < 0.01 | Minimal clipping — acceptable |
| > 0.01 | Significant clipping — likely audible distortion |

### Bandwidth

| kHz | Meaning |
|:---:|---------|
| < 4 | Telephone quality (8kHz sample rate equivalent) |
| 4–7 | Wideband telephony |
| > 7 | Full-band — genuine high-quality recording |

A file saved as 16kHz WAV but with bandwidth 3.5 kHz was **upsampled from 8kHz** — the extra samples contain no real information.

## Customization

### Add DNSMOS quality scoring

For a perceptual quality score (1–5 MOS scale):

```yaml
  - name: dnsmos
    op: dnsmos_score

  - name: filter
    op: quality_score_filter
    args:
      conditions:
        - "metrics.snr > 10"
        - "metrics.dnsmos_ovrl > 3.0"   # perceptual quality > 3.0/5.0
```

### Keep only long-form audio

```yaml
  - name: filter
    op: quality_score_filter
    args:
      conditions:
        - "duration > 10"     # minimum 10 seconds
        - "duration < 300"    # maximum 5 minutes
```

### Stricter deduplication

Lower threshold = stricter (fewer false negatives, more false positives):

```yaml
  - name: dedup
    op: audio_fingerprint_dedup
    args:
      similarity_threshold: 2   # default is 3
```
