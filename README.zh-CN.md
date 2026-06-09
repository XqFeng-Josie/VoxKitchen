<p align="center">
  <img src="https://raw.githubusercontent.com/XqFeng-Josie/VoxKitchen/main/voxkitchen_logo.svg" width="360" alt="VoxKitchen logo">
</p>

<h1 align="center">VoxKitchen</h1>

<p align="center">
  <strong>把原始语音录音变成干净、可检视的训练数据集。</strong>
</p>

<p align="center">
  VoxKitchen 负责 ASR、TTS、说话人分析与数据清洗周边那些重复的音频预处理工作：
  通过一份 Docker 驱动的 YAML 流水线完成转换、切分、标注、过滤与导出。
</p>

<p align="center">
  <a href="https://github.com/XqFeng-Josie/VoxKitchen/actions/workflows/ci.yml"><img src="https://github.com/XqFeng-Josie/VoxKitchen/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/voxkitchen/"><img src="https://img.shields.io/pypi/v/voxkitchen.svg" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python"></a>
  <img src="https://img.shields.io/badge/runtime-Docker--first-2496ED" alt="Docker-first">
  <img src="https://img.shields.io/badge/operators-52-brightgreen" alt="52 operators">
  <a href="https://github.com/XqFeng-Josie/VoxKitchen/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
</p>

<p align="center">
  <a href="https://github.com/XqFeng-Josie/VoxKitchen/blob/main/README.md">English</a>
</p>

在以下场景使用 VoxKitchen：

- 把长录音切成 ASR 训练数据；
- 准备并检视 TTS 数据集；
- 做说话人分离（diarization）、语种标注，或运行语音质量检查；
- 在不维护一堆一次性脚本的前提下清洗、过滤、打包音频。

## 为什么选择 VoxKitchen

语音数据准备通常是一连串脆弱的脚本：转换音频、切分语音、降噪、转写、说话人分离、
过滤、导出。VoxKitchen 让这条链路变得显式且可复现：

- **Docker 优先执行**：预构建的运行时镜像，避免本地依赖冲突。
- **一份 YAML 流水线**：在同一个文件里定义 ingest、stages、filters 和输出打包。
- **52 个内置算子**：音频预处理、VAD、ASR、说话人分离、TTS、质量指标与打包。
- **天生可恢复**：每个 stage 都在 `./work` 下落盘 checkpoint。
- **可检视的产出**：报告、切分统计、来源追溯（provenance）和按 stage 的错误记录。

## 快速开始

环境要求：

- Docker
- Python 3.10+（用于运行轻量的 `vkit` 启动器）

从 PyPI 安装 `vkit` 启动器：

```bash
pipx install voxkitchen      # 推荐 —— 隔离启动器环境
# 或
pip install voxkitchen
```

这只会安装轻量的启动器和检视命令（几 MB，不含 torch / ASR / TTS 依赖）。
所有流水线运行时依赖都待在预构建的 Docker 镜像里。

用最小的运行时镜像跑内置 demo。无需 clone 仓库；发布的镜像已经包含 demo
流水线和 demo 音频。

```bash
vkit docker pull --tag slim
vkit docker run --tag slim examples/pipelines/demo-no-asr.yaml
vkit inspect run ./work/demo-no-asr
```

<details>
<summary>示例输出</summary>

```text
$ vkit docker run --tag slim examples/pipelines/demo-no-asr.yaml
06:20:54  stage [1/9] to_wav  (ffmpeg_convert, 1 cuts in, env=core)
06:20:55  stage [1/9] to_wav  done → 1 cuts out (0.3s)
06:20:55  stage [2/9] vad  (silero_vad, 1 cuts in, env=core)
06:21:08  stage [2/9] vad  done → 7 cuts out (13.2s)
06:21:08  stage [3/9] extract  (ffmpeg_convert, 7 cuts in, env=core)
06:21:08  stage [3/9] extract  done → 7 cuts out (0.6s)
06:21:08  stage [4/9] snr  (snr_estimate, 7 cuts in, env=core)
06:21:08  stage [4/9] snr  done → 7 cuts out (0.0s)
06:21:08  stage [5/9] pitch  (pitch_stats, 7 cuts in, env=core)
06:21:11  stage [5/9] pitch  done → 7 cuts out (2.3s)
06:21:11  stage [6/9] clipping  (clipping_detect, 7 cuts in, env=core)
06:21:11  stage [6/9] clipping  done → 7 cuts out (0.0s)
06:21:11  stage [7/9] gender  (gender_classify, 7 cuts in, env=core)
06:21:44  stage [7/9] gender  done → 7 cuts out (33.2s)
06:21:44  stage [8/9] filter  (quality_score_filter, 7 cuts in, env=core)
06:21:44  stage [8/9] filter  done → 7 cuts out (0.0s)
06:21:44  stage [9/9] pack  (pack_jsonl, 7 cuts in, env=core)
06:21:44  stage [9/9] pack  done → 7 cuts out (0.0s)
06:21:44  report generated: work/demo-no-asr/report.html
pipeline complete
  work_dir: work/demo-no-asr
  final cuts: work/demo-no-asr/08_pack/cuts.jsonl.gz
  report: work/demo-no-asr/report.html

$ vkit inspect run ./work/demo-no-asr
Pipeline run: demo-no-asr
  00_to_wav: OK (1 cuts)  0.3s, 4 cuts/s
  01_vad: OK (7 cuts)  13.2s, 1 cuts/s
  02_extract: OK (7 cuts)  0.6s, 11 cuts/s
  03_snr: OK (7 cuts)  0.0s, 310 cuts/s
  04_pitch: OK (7 cuts)  2.3s, 3 cuts/s
  05_clipping: OK (7 cuts)  0.0s, 381 cuts/s
  06_gender: OK (7 cuts)  33.2s, 0 cuts/s
  07_filter: OK (7 cuts)  0.0s, 8124 cuts/s
  08_pack: OK (7 cuts)  0.0s, 6638 cuts/s
```

</details>

`vkit docker run` 会把运行产物写到 `./work` 下、导出的数据集写到 `./output` 下，
并使用你的宿主机用户 ID。当 `./data` 目录存在时也会自动挂载它。

## 你能用它做什么

| 目标 | 起步命令 | 运行时镜像 |
|---|---|---|
| 清洗并过滤原始语音音频 | `vkit init my-cleaning --template cleaning` | `slim` |
| 构建 ASR 训练 manifest | `vkit init my-asr --template asr` | `asr` |
| 分析说话人与语种 | `vkit init my-speakers --template speaker` | `latest` |
| 准备 TTS 训练数据（质量门控） | `vkit init my-tts --template tts` | `asr` |
| 用内置音色合成语音 | 见 [Speaker TTS 教程](https://github.com/XqFeng-Josie/VoxKitchen/blob/main/docs/tutorials/tts-speaker.md) | `tts` |
| 用 3–10 秒参考音克隆音色 | 见 [声音克隆 & TTS 教程](https://github.com/XqFeng-Josie/VoxKitchen/blob/main/docs/tutorials/tts-voice-cloning.md) | `tts` 或 `fish-speech` |

## 工作原理

![VoxKitchen pipeline overview](https://raw.githubusercontent.com/XqFeng-Josie/VoxKitchen/main/pipeline.png)

一条流水线就是一份 YAML 文件。每个 stage 读取一个 `CutSet`、写一个 checkpoint，
再把结果传给下一个 stage。

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
      conditions: ["duration > 1", "duration < 30", "metrics.snr > 10"]

  - name: pack
    op: pack_jsonl
```

被中断的运行会从已完成的 checkpoint 处恢复。

## 创建一个项目

```bash
vkit init my-project --template asr
cd my-project

# 先把你的音频文件放进 ./data。
vkit validate pipeline.yaml
vkit docker run --tag asr pipeline.yaml --dry-run
vkit docker run --tag asr pipeline.yaml
vkit inspect run work/
```

列出所有模板：

```bash
vkit init --list-templates
```

不确定某条流水线需要哪个镜像？运行：

```bash
vkit validate pipeline.yaml
```

它会针对该 YAML 打印推荐的 `vkit docker pull --tag ...` 和
`vkit docker run --tag ...` 命令。

## 运行时镜像

每个 `vkit docker` 命令都接受 `--tag <name>`：

| 标签 | 适用场景 | GPU | 大致体积 |
|---|---|---|---|
| `slim` | CPU 友好的清洗、VAD、质量、打包、增强 | 否 | ~22 GB |
| `asr` | Faster-Whisper、FunASR、Qwen3-ASR、强制对齐 | 是 | ~36 GB |
| `diarize` | Pyannote 说话人分离 | 是 | ~20 GB |
| `tts` | Kokoro、ChatTTS、CosyVoice | 是 | ~46 GB |
| `fish-speech` | Fish-Speech 独立运行时 | 是 | ~58 GB |
| `latest` | 跨 ASR、说话人分离、TTS 或 Fish-Speech 的混合流水线 | 是 | ~88 GB |

当一条流水线混用多个运行时家族（例如 ASR 加说话人分离、或 ASR 加 TTS）时使用
`latest`。否则，优先选择包含你所需算子的最小镜像。

实用检查：

```bash
vkit docker pull --tag asr
vkit docker doctor --tag asr --expect asr
vkit docker doctor --tag latest
```

## 配置

部分算子需要 API token。创建 `./.env`；`vkit docker run` 会自动把它传进容器。

```bash
cp .env.example .env
```

| 变量 | 被谁需要 | 说明 |
|---|---|---|
| `HF_TOKEN` | `pyannote_diarize` | 需先在 HuggingFace 上接受 pyannote 模型协议。 |

## 常用命令

```bash
vkit init <path> --template asr           # 脚手架生成一个项目
vkit validate pipeline.yaml               # 校验 YAML 并推荐镜像
vkit docker run --tag asr pipeline.yaml --dry-run
vkit docker run --tag asr pipeline.yaml
vkit inspect run work/                    # stage 概览
vkit inspect cuts <cuts.jsonl.gz>          # CutSet 统计
vkit inspect errors work/                  # 按 stage 列出失败的 cut
vkit operators search <keyword>            # 按名称或摘要查找算子
vkit operators --category quality          # 列出某一类别的算子
vkit schema export --out pipeline.schema.json  # 为 YAML 提供编辑器自动补全
vkit recipes                               # 列出数据集 recipe
vkit docker download --tag slim librispeech --root ./data/librispeech --subsets dev-clean
vkit docker doctor --tag latest            # 检查镜像健康状况
```

## 文档

- [入门指南](https://github.com/XqFeng-Josie/VoxKitchen/blob/main/docs/getting-started.md)
- [示例与用例](https://github.com/XqFeng-Josie/VoxKitchen/blob/main/docs/examples.md)
- [流水线 YAML](https://github.com/XqFeng-Josie/VoxKitchen/blob/main/docs/reference/pipeline-yaml.md)
- [数据集目录](https://github.com/XqFeng-Josie/VoxKitchen/blob/main/docs/datasets/index.md)
- [CLI 参考](https://github.com/XqFeng-Josie/VoxKitchen/blob/main/docs/reference/cli.md)
- [算子参考](https://github.com/XqFeng-Josie/VoxKitchen/blob/main/docs/reference/operators.md)
- [Docker 构建指南](https://github.com/XqFeng-Josie/VoxKitchen/blob/main/docs/docker-build.md)
- [贡献指南](https://github.com/XqFeng-Josie/VoxKitchen/blob/main/CONTRIBUTING.md)

## Agent Skill

本仓库包含一个与 agent 无关的 VoxKitchen skill，位于 [skill/](https://github.com/XqFeng-Josie/VoxKitchen/tree/main/skill)。
Claude、Codex 以及其他兼容 `SKILL.md` 的 agent 可以把该文件夹复制、软链接或导入到自己的
skill 搜索路径中。该 skill 遵循本 README 中 Docker 优先的 `vkit` 工作流。

## 引用

如果你在研究中使用了 VoxKitchen，请引用本项目。仓库内附
[`CITATION.cff`](https://github.com/XqFeng-Josie/VoxKitchen/blob/main/CITATION.cff)，
GitHub 会在仓库页的「Cite this repository」处渲染它。

## 许可证

Apache 2.0。见 [LICENSE](https://github.com/XqFeng-Josie/VoxKitchen/blob/main/LICENSE)。
