---
name: voxkitchen
description: >
  Agent guide for VoxKitchen, a Docker-first declarative speech data processing
  toolkit. Use this skill when the user asks to design, validate, run, debug,
  or explain VoxKitchen YAML pipelines; choose operators for ASR, TTS, speaker
  analysis, data cleaning, augmentation, quality filtering, or dataset packing;
  use the vkit CLI; download supported speech datasets; inspect pipeline output;
  or decide which VoxKitchen Docker runtime image to pull. Trigger on vkit,
  voxkitchen, speech pipeline, audio dataset prep, TTS data, ASR training data,
  VAD, diarization, audio YAML recipe, CutSet, or large-scale audio processing.
---

# VoxKitchen Skill

VoxKitchen users write a YAML pipeline and run it through the lightweight
`vkit` launcher inside prebuilt Docker runtimes. The default user path is:

```bash
pipx install voxkitchen      # or: pip install voxkitchen
vkit validate pipeline.yaml
vkit docker pull --tag <recommended-tag>
vkit docker run --tag <recommended-tag> pipeline.yaml
vkit inspect run work/
```

`vkit validate <yaml>` and `vkit docker run <yaml> --dry-run` print the
recommended Docker image for that pipeline. Prefer those commands over asking
users to choose an image manually.

No repository clone is required for the quick start; the published Docker
images include the demo pipeline and demo audio. Recommend cloning the repo only
when users want to inspect or modify examples, use the bundled `skill/`, build
images locally, or contribute code.

## Product Boundaries

- Host `vkit` is a launcher and inspection tool. User pipeline execution should
  use `vkit docker run`, not local `vkit run`.
- Dataset downloads should use `vkit docker download`, which defaults to the
  `slim` image and writes under `./data`.
- Do not recommend `pip install voxkitchen[...]`, pip extras, raw `docker run`,
  or custom image builds for normal users.
- Mention local installs, `vkit run`, `vkit docker build`, or dependency groups
  only when the user is explicitly doing contributor or image-maintainer work.
- Use `./data`, `./work`, and `./output` paths in examples. Avoid `/data/...`
  unless the user is intentionally discussing container internals.

## How To Help

### Designing A Pipeline

Ask only for missing information that changes the YAML:

- Goal: cleaning, ASR training data, TTS data preparation, speaker analysis,
  augmentation, or a custom workflow.
- Source: local audio directory, existing manifest, or supported recipe.
- Output format: `pack_jsonl`, `pack_huggingface`, `pack_kaldi`,
  `pack_webdataset`, `pack_parquet`, or just `pack_manifest`.
- Quality constraints: duration range, SNR, clipping, DNSMOS/UTMOS, language,
  speaker labels, or text availability.

Start from a template when possible:

```bash
vkit init my-project --template cleaning
vkit init my-project --template asr
vkit init my-project --template speaker
vkit init my-project --template tts
```

Then edit `pipeline.yaml`, validate it, and run it in Docker.

### Choosing Images

Use automatic recommendation first:

```bash
vkit validate pipeline.yaml
```

Fallback rules when reasoning from operators:

| Image tag | Use when |
|---|---|
| `slim` | CPU-friendly cleaning, VAD, quality metrics, packing, enhancement |
| `asr` | Faster-Whisper, FunASR, Qwen3-ASR, WeNet, forced alignment |
| `diarize` | Pyannote speaker diarization |
| `tts` | Kokoro, ChatTTS, CosyVoice |
| `fish-speech` | Fish-Speech isolated runtime |
| `latest` | A pipeline mixes multiple specialized runtime families |

For supported dataset recipes, use:

```bash
vkit docker download librispeech --root ./data/librispeech --subsets dev-clean
vkit docker download libritts --root ./data/libritts --subsets dev-clean
vkit docker download ljspeech --root ./data/ljspeech
vkit docker download aishell --root ./data/aishell
vkit docker download aishell3 --root ./data/aishell3
vkit docker download fleurs --root ./data/fleurs --subsets en_us,zh_cn
```

### Running And Debugging

Recommended workflow:

```bash
vkit validate pipeline.yaml
vkit docker run --tag <recommended-tag> pipeline.yaml --dry-run
vkit docker run --tag <recommended-tag> pipeline.yaml
vkit inspect run work/
vkit inspect errors work/
```

Resume or stop at stages through the Docker wrapper:

```bash
vkit docker run --tag asr pipeline.yaml --resume-from vad
vkit docker run --tag asr pipeline.yaml --stop-at asr
```

Common checks:

- `pyannote_diarize` needs `HF_TOKEN` in `./.env` and accepted HuggingFace model
  terms.
- Missing or empty `ingest.args.root` should point users to put audio under
  `./data` or update the pipeline.
- If a pipeline spans ASR plus diarization or TTS, use `latest` unless
  `vkit validate` recommends a smaller tag.
- For macOS ASR issues with CTranslate2-based operators, suggest
  `whisper_openai_asr` or Docker validation before changing the pipeline.

## YAML Skeleton

```yaml
version: "0.1"
name: my-pipeline
work_dir: ./work/${name}-${run_id}

ingest:
  source: dir
  args:
    root: ./data
    recursive: true

stages:
  - name: resample
    op: resample
    args:
      target_sr: 16000
      target_channels: 1

  - name: pack
    op: pack_jsonl
```

Rules for generated YAML:

- Always include `version`, `name`, `work_dir`, `ingest`, and `stages`.
- Keep stage names short, unique, and stable.
- End user-facing pipelines with a `pack_*` stage.
- Put expensive stages after cheap quality gates when that does not change the
  intended data semantics.
- Tell the user to run `vkit validate pipeline.yaml` before execution.

## References

Load these only when needed:

- `references/cli-reference.md`: Docker-first CLI commands and image guidance.
- `references/operators.md`: operator selection guide and common YAML snippets.

When working inside the VoxKitchen repository, prefer the live project docs if
they differ from bundled references:

- `README.md`
- `docs/getting-started.md`
- `docs/reference/cli.md`
- `docs/reference/operators.md`
- `docs/reference/pipeline-yaml.md`

## Agent Compatibility

This directory is the canonical, agent-neutral VoxKitchen skill package:

```text
skill/
├── SKILL.md
├── openai.yaml
└── references/
```

Claude, Codex, and other `SKILL.md`-compatible agents can copy, symlink, or
import this directory into their own skill search path. `openai.yaml` provides
optional OpenAI/Codex-style UI metadata without making the skill Claude-specific.
