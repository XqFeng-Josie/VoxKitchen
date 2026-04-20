# VoxKitchen 下一阶段开发路线图

> **Historical document.** This was the plan leading up to v0.1.0 (templates,
> docs site, public release, more recipes). For what actually shipped, see
> [CHANGELOG.md](https://github.com/XqFeng-Josie/VoxKitchen/blob/main/CHANGELOG.md);
> for what's next, see project issues. Kept for historical reference — do not
> treat as current guidance. Not linked from the docs nav for the same reason.

## 阶段总览

| 顺序 | 方向 | 目标 | 预估工作量 |
|:---:|------|------|:---:|
| 1 | 场景化 Pipeline 模板 | 让用户 5 分钟内跑通完整流程 | 中 |
| 2 | 文档站 + 教程 | 降低认知门槛，让用户自助 | 中 |
| 3 | 公开发布准备 | GitHub 社区基础设施 + 首次正式发布 | 小 |
| 4 | 更多数据源 Recipes | 扩大数据覆盖面 | 中 |

---

## 阶段 1：场景化 Pipeline 模板

### 目标

从"工具集合"变成"解决方案"。用户不需要自己组合算子，直接 `vkit init --template tts` 就能得到一个可运行的 TTS 数据制备 pipeline。

### 1.1 TTS 数据制备模板

**场景**：从粗糙的音频制备高质量 TTS 训练数据

```yaml
# templates/tts-data-prep.yaml
version: "0.1"
name: tts-data-prep
description: "Raw audio → TTS-ready dataset with word-level alignment"
work_dir: ./work/${name}

ingest:
  source: dir
  args:
    root: ./data/raw

stages:
  # 1. 基础处理
  - name: to_wav
    op: ffmpeg_convert
    args: { target_format: wav }

  - name: resample
    op: resample
    args: { target_sr: 22050, target_channels: 1 }  # TTS 常用 22.05kHz

  # 2. 降噪
  - name: enhance
    op: speech_enhance
    args: { aggressiveness: 0.3 }  # TTS 用轻度降噪，保留自然感

  # 3. 切分
  - name: vad
    op: silero_vad
    args: { threshold: 0.5, min_speech_duration_ms: 1000 }  # TTS 要较长段

  # 4. 质量筛选
  - name: snr
    op: snr_estimate
  - name: dnsmos
    op: dnsmos_score
  - name: filter
    op: quality_score_filter
    args:
      conditions:
        - "duration > 2"
        - "duration < 15"
        - "metrics.snr > 15"
        - "metrics.dnsmos_ovrl > 3.0"

  # 5. ASR + 强制对齐
  - name: asr
    op: qwen3_asr
    args:
      model: Qwen/Qwen3-ASR-0.6B
      return_timestamps: true

  # 6. 打包
  - name: pack
    op: pack_jsonl
```

**配套文档**：`docs/tutorials/tts-data-prep.md`
- 为什么 TTS 要 22.05kHz
- 降噪激进度如何选择
- 质量阈值的含义（DNSMOS 3.0 = 什么水平）
- 对齐结果怎么用于 TTS 训练

### 1.2 ASR 训练数据模板

**场景**：大规模音频 → 增强后的 ASR 训练数据

```yaml
# templates/asr-training-data.yaml
stages:
  - name: resample
    op: resample
    args: { target_sr: 16000, target_channels: 1 }

  - name: vad
    op: silero_vad
    args: { threshold: 0.5 }

  # 数据增强 3 倍
  - name: speed
    op: speed_perturb
    args: { factors: [0.9, 1.0, 1.1] }

  - name: volume
    op: volume_perturb
    args: { min_gain_db: -3, max_gain_db: 3 }

  - name: asr
    op: faster_whisper_asr
    args: { model: large-v3, compute_type: float16 }

  - name: filter
    op: quality_score_filter
    args:
      conditions:
        - "duration > 1"
        - "duration < 30"

  - name: pack
    op: pack_huggingface
    args: { output_dir: ./output/hf_dataset }
```

### 1.3 数据清洗模板

**场景**：清理质量差、有重复的原始数据

```yaml
# templates/data-cleaning.yaml
stages:
  - name: resample
    op: resample
    args: { target_sr: 16000, target_channels: 1 }

  - name: snr
    op: snr_estimate

  - name: clipping
    op: clipping_detect

  - name: bandwidth
    op: bandwidth_estimate

  - name: dedup
    op: audio_fingerprint_dedup
    args: { similarity_threshold: 3 }

  - name: filter
    op: quality_score_filter
    args:
      conditions:
        - "duration > 0.5"
        - "duration < 60"
        - "metrics.snr > 10"
        - "metrics.clipping_ratio < 0.01"
        - "metrics.bandwidth_khz > 4"   # 排除电话线质量

  - name: pack
    op: pack_jsonl
```

### 1.4 说话人分析模板

**场景**：分析数据集中的说话人分布

```yaml
# templates/speaker-analysis.yaml
stages:
  - name: resample
    op: resample
    args: { target_sr: 16000, target_channels: 1 }

  - name: vad
    op: silero_vad
    args: { threshold: 0.5 }

  - name: diarize
    op: pyannote_diarize

  - name: embed
    op: speaker_embed
    args: { method: wespeaker }

  - name: gender
    op: gender_classify
    args: { method: f0 }

  - name: langid
    op: whisper_langid
    args: { model: tiny }

  - name: pack
    op: pack_jsonl
```

### 1.5 `vkit init --template` 命令增强

当前 `vkit init` 只创建一个空模板。增强为：

```bash
vkit init my-project                         # 空模板（现有行为）
vkit init my-project --template tts          # TTS 数据制备
vkit init my-project --template asr          # ASR 训练数据
vkit init my-project --template cleaning     # 数据清洗
vkit init my-project --template speaker      # 说话人分析

vkit init --list-templates                   # 列出所有可用模板
```

### 实现清单

- [ ] 创建 `voxkitchen/templates/` 目录，存放模板 YAML
- [ ] 修改 `vkit init` 支持 `--template` 参数
- [ ] 创建 4 个模板 YAML
- [ ] 每个模板配套一个教程文档
- [ ] 创建 `vkit init --list-templates`
- [ ] 测试每个模板的 `vkit validate`

---

## 阶段 2：文档站 + 教程

### 目标

把 mkdocs 从 3 页扩展到完整的文档站，覆盖：入门 → 教程 → 参考 → 贡献指南。

### 2.1 文档结构

```
docs/
├── index.md                          # 首页（已有）
├── getting-started.md                # 快速开始（已有，需更新）
├── concepts/
│   └── data-protocol.md              # 数据模型（已有）
│
├── tutorials/                        # 新增
│   ├── tts-data-prep.md              # TTS 数据制备教程
│   ├── asr-training-data.md          # ASR 训练数据教程
│   ├── data-cleaning.md              # 数据清洗教程
│   ├── custom-operator.md            # 如何写自定义算子
│   └── download-datasets.md          # 数据下载教程
│
├── reference/                        # 新增
│   ├── operators.md                  # 算子参考（自动生成）
│   ├── recipes.md                    # 数据集 recipe 参考
│   ├── cli.md                        # CLI 命令参考
│   ├── tools-api.md                  # Python tools API 参考
│   └── pipeline-yaml.md             # YAML 规范参考
│
└── contributing.md                   # 贡献指南（阶段 3）
```

### 2.2 算子参考自动生成

写一个脚本 `scripts/gen_operator_docs.py`：
- 遍历所有注册算子
- 提取 docstring、config 字段、device、required_extras
- 生成 Markdown 表格
- 输出到 `docs/reference/operators.md`

这样每次加新算子后跑一次脚本就能更新文档。

### 2.3 mkdocs.yml 导航更新

```yaml
nav:
  - Home: index.md
  - Getting Started: getting-started.md
  - Concepts:
    - Data Protocol: concepts/data-protocol.md
  - Tutorials:
    - TTS Data Preparation: tutorials/tts-data-prep.md
    - ASR Training Data: tutorials/asr-training-data.md
    - Data Cleaning: tutorials/data-cleaning.md
    - Download Datasets: tutorials/download-datasets.md
    - Custom Operators: tutorials/custom-operator.md
  - Reference:
    - Operators: reference/operators.md
    - Recipes & Download: reference/recipes.md
    - CLI Commands: reference/cli.md
    - Python Tools API: reference/tools-api.md
    - Pipeline YAML: reference/pipeline-yaml.md
  - Contributing: contributing.md
```

### 实现清单

- [ ] 创建 `scripts/gen_operator_docs.py` 自动生成算子文档
- [ ] 编写 4 个教程文档（配合阶段 1 的模板）
- [ ] 编写参考文档（CLI、tools API、YAML 规范、recipes）
- [ ] 更新 getting-started.md（反映新功能）
- [ ] 更新 mkdocs.yml 导航
- [ ] 部署到 GitHub Pages

---

## 阶段 3：公开发布准备

### 目标

让项目从 private develop 变成可以公开的开源项目。

### 3.1 GitHub 社区基础设施

| 文件 | 内容 |
|------|------|
| `CONTRIBUTING.md` | 开发环境搭建、commit 规范、如何加算子、PR 流程 |
| `CHANGELOG.md` | v0.1.0 和 v0.2.0 的变更记录 |
| `SECURITY.md` | 漏洞报告流程 |
| `CODE_OF_CONDUCT.md` | 社区行为规范（用 Contributor Covenant） |
| `.github/ISSUE_TEMPLATE/bug_report.yml` | Bug 报告模板 |
| `.github/ISSUE_TEMPLATE/feature_request.yml` | 功能请求模板 |
| `.github/ISSUE_TEMPLATE/new_dataset.yml` | 新数据集 recipe 请求 |
| `.github/PULL_REQUEST_TEMPLATE.md` | PR 模板 |

### 3.2 README 优化

- 添加 badges（CI、PyPI、Python 版本、License）
- 添加项目 logo/banner（可选）
- 添加 "Who is this for" 段落
- 添加 "Comparison with other tools" 段落（vs Lhotse、NeMo、SpeechBrain）

### 3.3 修复 Release Workflow

```yaml
# .github/workflows/release.yml 修复步骤
1. 在 GitHub Settings → Environments 创建 "release" 环境
2. 在 PyPI 注册 voxkitchen 包名（或用 pending publisher）
3. 配置 Trusted Publisher（GitHub → PyPI OIDC）
4. 打 v0.2.0 tag 触发发布
```

### 3.4 v0.2.0 Release Notes 大纲

```markdown
## v0.2.0 — Feature Expansion Release

### New Operators (+9)
- **Augmentation**: speed_perturb, volume_perturb, noise_augment, reverb_augment
- **Annotation**: speaker_embed, speech_enhance, forced_align, emotion_recognize, qwen3_asr

### New Features
- Lazy CutSet for large-scale manifest processing
- Per-stage execution statistics (_stats.json)
- `vkit download` command for dataset downloads
- FLEURS recipe (102-language HuggingFace dataset)
- Operator categories in `vkit operators` output
- Python tools API: extract_speaker_embedding, enhance_speech, align_words

### Infrastructure
- LibriSpeech and AISHELL download support
- Scene-based pipeline templates (TTS, ASR, cleaning)
- Documentation site with tutorials
```

### 实现清单

- [ ] 创建 CONTRIBUTING.md
- [ ] 创建 CHANGELOG.md
- [ ] 创建 SECURITY.md + CODE_OF_CONDUCT.md
- [ ] 创建 Issue/PR templates
- [ ] README 添加 badges
- [ ] 配置 GitHub release environment + PyPI trusted publisher
- [ ] 打 v0.2.0 tag 并发布

---

## 阶段 4：更多数据源 Recipes

### 目标

覆盖主流中英文 + 多语种数据集，让"找数据"这个价值主张成立。

### 4.1 新增 Recipes

| 数据集 | 语言 | 规模 | 下载方式 | 优先级 |
|--------|------|------|---------|:---:|
| **MUSAN** | - | 噪声/音乐 | openslr | P0（增强算子配套） |
| **GigaSpeech** | EN | 10k h | HuggingFace | P0 |
| **WenetSpeech** | ZH | 10k h | HuggingFace | P0 |
| **MLS** | 8 lang | 50k h | openslr | P1 |
| **VoxCeleb 1/2** | Multi | 2k h | 手动下载，只做 recipe | P1 |
| **People's Speech** | EN | 30k h | HuggingFace | P2 |
| **KeSpeech** | ZH 方言 | 1.5k h | 需申请 | P2 |

### 4.2 MUSAN 噪声数据 Recipe（最高优先级）

MUSAN 是增强算子（noise_augment、reverb_augment）的配套数据。有了它，用户可以：

```bash
vkit download musan --root /data/musan
```

然后在 pipeline 里直接用：

```yaml
- name: add_noise
  op: noise_augment
  args:
    noise_dir: /data/musan/noise
    snr_range: [5, 20]

- name: add_reverb
  op: reverb_augment
  args:
    rir_dir: /data/musan/rir     # 需要额外 RIR 数据
```

### 4.3 `vkit recipes` 命令

新增 CLI 命令，列出所有可用 recipe 及其下载状态：

```bash
$ vkit recipes
┌─────────────┬──────────┬──────────────────┬──────────┐
│ Recipe      │ Language │ Download Support │ Status   │
├─────────────┼──────────┼──────────────────┼──────────┤
│ librispeech │ EN       │ ✓ openslr        │ ready    │
│ aishell     │ ZH       │ ✓ openslr        │ ready    │
│ fleurs      │ 102 lang │ ✓ HuggingFace    │ ready    │
│ commonvoice │ Multi    │ ✗ manual         │ ready    │
│ musan       │ -        │ ✓ openslr        │ ready    │
│ gigaspeech  │ EN       │ ✓ HuggingFace    │ ready    │
│ wenetspeech │ ZH       │ ✓ HuggingFace    │ ready    │
└─────────────┴──────────┴──────────────────┴──────────┘
```

### 实现清单

- [ ] MUSAN recipe + download
- [ ] GigaSpeech recipe + download
- [ ] WenetSpeech recipe + download
- [ ] MLS recipe + download
- [ ] VoxCeleb recipe（仅解析，无下载）
- [ ] `vkit recipes` CLI 命令
- [ ] 每个 recipe 配套 example pipeline

---

## 执行节奏建议

| 周次 | 工作内容 |
|:---:|---------|
| Week 1 | 阶段 1：4 个场景模板 + `vkit init --template` |
| Week 2 | 阶段 2：自动生成算子文档 + 4 个教程 |
| Week 3 | 阶段 2：参考文档 + mkdocs 部署 |
| Week 4 | 阶段 3：CONTRIBUTING + Issue templates + Release |
| Week 5-6 | 阶段 4：MUSAN + GigaSpeech + WenetSpeech recipes |

---

## 验收标准

项目可以公开发布时，应满足：

1. **新用户 5 分钟跑通**：`pip install voxkitchen[all]` → `vkit init --template tts` → `vkit run pipeline.yaml` → 得到结果
2. **文档自助**：用户遇到问题能在文档站找到答案
3. **贡献者友好**：有人想加新算子，看 CONTRIBUTING.md + custom-operator 教程就能做
4. **数据覆盖**：中英文各有 2+ 个可直接下载的大规模数据集
5. **CI 绿色**：所有 test pass，release workflow 能发 PyPI
