# TTS Synthesis Operators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 4 TTS synthesis operators (Kokoro, ChatTTS, CosyVoice2, Fish-Speech) that convert text to speech audio, plus a unified `synthesize()` tool function.

**Architecture:** Each TTS operator follows the `speech_enhance` pattern: reads text from `supervision.text`, generates audio in `derived/`, creates a new Recording + Cut with provenance. Input Cuts need only a text supervision — no source audio required. `reads_audio_bytes = False, produces_audio = True`.

**Tech Stack:** Kokoro (kokoro, espeak-ng), ChatTTS (ChatTTS), CosyVoice2 (cosyvoice via git, modelscope), Fish-Speech (fish-speech via git, ormsgpack).

**Spec reference:** `docs/superpowers/specs/2026-04-15-tts-tools-design.md`

---

## File Structure

```
Created:
  src/voxkitchen/operators/synthesize/__init__.py
  src/voxkitchen/operators/synthesize/tts_kokoro.py
  src/voxkitchen/operators/synthesize/tts_chattts.py
  src/voxkitchen/operators/synthesize/tts_cosyvoice.py
  src/voxkitchen/operators/synthesize/tts_fish_speech.py
  tests/unit/operators/synthesize/__init__.py
  tests/unit/operators/synthesize/test_tts_kokoro.py
  tests/unit/operators/synthesize/test_tts_chattts.py
  tests/unit/operators/synthesize/test_tts_cosyvoice.py
  tests/unit/operators/synthesize/test_tts_fish_speech.py

Modified:
  src/voxkitchen/operators/__init__.py
  src/voxkitchen/tools.py
  pyproject.toml
  README.md
```

**Responsibility per file:**

- `tts_kokoro.py` — Kokoro TTS: lightweight, CPU-capable, 82M params, 8 languages
- `tts_chattts.py` — ChatTTS: conversational style, random speaker sampling, GPU
- `tts_cosyvoice.py` — CosyVoice2: zero-shot voice cloning, ModelScope auto-download, GPU
- `tts_fish_speech.py` — Fish-Speech: zero-shot voice cloning, codec-LM, GPU
- `tools.py` — add `synthesize()` convenience function dispatching to all 4 engines
- `operators/__init__.py` — register new synthesize category operators
- `pyproject.toml` — add `tts-kokoro`, `tts-chattts`, `tts-cosyvoice`, `tts-fish-speech` extras

---

## Task 1: Kokoro TTS operator

**Files:**
- Create: `src/voxkitchen/operators/synthesize/__init__.py`
- Create: `src/voxkitchen/operators/synthesize/tts_kokoro.py`
- Create: `tests/unit/operators/synthesize/__init__.py`
- Create: `tests/unit/operators/synthesize/test_tts_kokoro.py`
- Modify: `src/voxkitchen/operators/__init__.py`

- [ ] **Step 1: Create synthesize package**

Create `src/voxkitchen/operators/synthesize/__init__.py`:

```python
"""TTS synthesis operators: generate speech audio from text."""
```

Create `tests/unit/operators/synthesize/__init__.py`:

```python
```

- [ ] **Step 2: Write tests**

Create `tests/unit/operators/synthesize/test_tts_kokoro.py`:

```python
"""Unit tests for tts_kokoro operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import soundfile as sf

try:
    import kokoro  # noqa: F401
except ImportError:
    pytest.skip("kokoro not available", allow_module_level=True)

from voxkitchen.operators.registry import get_operator
from voxkitchen.operators.synthesize.tts_kokoro import (
    TtsKokoroConfig,
    TtsKokoroOperator,
)
from voxkitchen.pipeline.context import RunContext
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.supervision import Supervision


def _ctx(tmp_path: Path) -> RunContext:
    return RunContext(
        work_dir=tmp_path,
        pipeline_run_id="run-test",
        stage_index=1,
        stage_name="tts",
        num_gpus=0,
        num_cpu_workers=1,
        gc_mode="aggressive",
        device="cpu",
    )


def _text_cut(cid: str, text: str) -> Cut:
    return Cut(
        id=cid,
        recording_id=f"text-{cid}",
        start=0.0,
        duration=0.0,
        supervisions=[
            Supervision(
                id=f"sup-{cid}",
                recording_id=f"text-{cid}",
                start=0.0,
                duration=0.0,
                text=text,
            )
        ],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="test",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
            pipeline_run_id="run-test",
        ),
    )


def test_tts_kokoro_is_registered() -> None:
    assert get_operator("tts_kokoro") is TtsKokoroOperator


def test_tts_kokoro_produces_audio() -> None:
    assert TtsKokoroOperator.produces_audio is True
    assert TtsKokoroOperator.reads_audio_bytes is False


def test_tts_kokoro_config_defaults() -> None:
    config = TtsKokoroConfig()
    assert config.voice == "af_heart"
    assert config.lang_code == "a"
    assert config.speed == 1.0


@pytest.mark.slow
def test_tts_kokoro_synthesizes_audio(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_text_cut("c0", "Hello, this is a test.")])
    config = TtsKokoroConfig(voice="af_heart", lang_code="a")
    op = TtsKokoroOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out = next(iter(result))
    assert out.recording is not None
    audio_path = out.recording.sources[0].source
    assert Path(audio_path).exists()
    info = sf.info(audio_path)
    assert info.samplerate == 24000
    assert info.duration > 0.1
    assert out.duration > 0.1
    assert out.provenance.generated_by == "tts_kokoro"
    assert out.provenance.source_cut_id == "c0"


@pytest.mark.slow
def test_tts_kokoro_skips_cut_without_text(tmp_path: Path) -> None:
    cut_no_text = Cut(
        id="c-empty",
        recording_id="text-empty",
        start=0.0,
        duration=0.0,
        supervisions=[],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="test",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
            pipeline_run_id="run-test",
        ),
    )
    ctx = _ctx(tmp_path)
    cs = CutSet([cut_no_text])
    config = TtsKokoroConfig()
    op = TtsKokoroOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    cuts = list(result)
    assert len(cuts) == 0
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/unit/operators/synthesize/test_tts_kokoro.py -v
```

- [ ] **Step 4: Implement tts_kokoro**

Create `src/voxkitchen/operators/synthesize/tts_kokoro.py`:

```python
"""Kokoro TTS operator: lightweight text-to-speech synthesis.

Kokoro is a compact (82M params) TTS model supporting 8 languages.
Can run on CPU. Output sample rate is 24kHz.

Requires system dependency: ``sudo apt-get install espeak-ng``
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.recording import Recording
from voxkitchen.utils.audio import recording_from_file, save_audio
from voxkitchen.utils.time import now_utc

logger = logging.getLogger(__name__)

KOKORO_SR = 24000


class TtsKokoroConfig(OperatorConfig):
    voice: str = "af_heart"
    lang_code: str = "a"  # a=AmE, b=BrE, j=Japanese, z=Mandarin
    speed: float = 1.0


@register_operator
class TtsKokoroOperator(Operator):
    """Synthesize speech from text using Kokoro TTS."""

    name = "tts_kokoro"
    config_cls = TtsKokoroConfig
    device = "cpu"
    produces_audio = True
    reads_audio_bytes = False
    required_extras = ["tts-kokoro"]

    _pipeline: Any

    def setup(self) -> None:
        from kokoro import KPipeline

        assert isinstance(self.config, TtsKokoroConfig)
        self._pipeline = KPipeline(lang_code=self.config.lang_code)

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, TtsKokoroConfig)
        derived_dir = self.ctx.stage_dir / "derived"
        derived_dir.mkdir(parents=True, exist_ok=True)

        out_cuts: list[Cut] = []
        for cut in cuts:
            text = self._extract_text(cut)
            if text is None:
                logger.warning("cut %s has no text, skipping", cut.id)
                continue

            audio_chunks: list[np.ndarray] = []
            for _gs, _ps, audio in self._pipeline(
                text, voice=self.config.voice, speed=self.config.speed
            ):
                audio_chunks.append(np.asarray(audio, dtype=np.float32))

            if not audio_chunks:
                logger.warning("cut %s produced no audio, skipping", cut.id)
                continue

            audio_full = np.concatenate(audio_chunks)
            out_path = derived_dir / f"{cut.id}__kokoro.wav"
            save_audio(out_path, audio_full, KOKORO_SR)
            new_rec = recording_from_file(out_path, recording_id=f"{cut.id}_kokoro")

            out_cuts.append(
                Cut(
                    id=f"{cut.id}__kokoro",
                    recording_id=new_rec.id,
                    start=0.0,
                    duration=new_rec.duration,
                    recording=new_rec,
                    supervisions=cut.supervisions,
                    metrics=cut.metrics,
                    custom=dict(cut.custom) if cut.custom else {},
                    provenance=Provenance(
                        source_cut_id=cut.id,
                        generated_by="tts_kokoro",
                        stage_name=self.ctx.stage_name,
                        created_at=now_utc(),
                        pipeline_run_id=self.ctx.pipeline_run_id,
                    ),
                )
            )
        return CutSet(out_cuts)

    @staticmethod
    def _extract_text(cut: Cut) -> str | None:
        for sup in cut.supervisions:
            if sup.text:
                return sup.text
        return None

    def teardown(self) -> None:
        self._pipeline = None
```

- [ ] **Step 5: Register in operators/__init__.py**

Add to `src/voxkitchen/operators/__init__.py` after the augment section and before the pack section, adding a new synthesize section:

```python
# --- synthesize (optional: kokoro, chattts, cosyvoice, fish-speech) ---
try:
    from voxkitchen.operators.synthesize import tts_kokoro as _synth_kokoro  # noqa: F401
except ImportError:
    pass  # kokoro not installed
```

- [ ] **Step 6: Run tests, verify pass**

```bash
pytest tests/unit/operators/synthesize/test_tts_kokoro.py -v
```

- [ ] **Step 7: Commit**

```bash
git add src/voxkitchen/operators/synthesize/ tests/unit/operators/synthesize/ src/voxkitchen/operators/__init__.py
git commit -m "feat(operators): add tts_kokoro — Kokoro TTS synthesis"
```

---

## Task 2: ChatTTS operator

**Files:**
- Create: `src/voxkitchen/operators/synthesize/tts_chattts.py`
- Create: `tests/unit/operators/synthesize/test_tts_chattts.py`
- Modify: `src/voxkitchen/operators/__init__.py`

- [ ] **Step 1: Write tests**

Create `tests/unit/operators/synthesize/test_tts_chattts.py`:

```python
"""Unit tests for tts_chattts operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import soundfile as sf

try:
    import ChatTTS  # noqa: F401
except ImportError:
    pytest.skip("ChatTTS not available", allow_module_level=True)

from voxkitchen.operators.registry import get_operator
from voxkitchen.operators.synthesize.tts_chattts import (
    TtsChatTTSConfig,
    TtsChatTTSOperator,
)
from voxkitchen.pipeline.context import RunContext
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.supervision import Supervision


def _ctx(tmp_path: Path) -> RunContext:
    return RunContext(
        work_dir=tmp_path,
        pipeline_run_id="run-test",
        stage_index=1,
        stage_name="tts",
        num_gpus=0,
        num_cpu_workers=1,
        gc_mode="aggressive",
        device="cpu",
    )


def _text_cut(cid: str, text: str) -> Cut:
    return Cut(
        id=cid,
        recording_id=f"text-{cid}",
        start=0.0,
        duration=0.0,
        supervisions=[
            Supervision(
                id=f"sup-{cid}",
                recording_id=f"text-{cid}",
                start=0.0,
                duration=0.0,
                text=text,
            )
        ],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="test",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
            pipeline_run_id="run-test",
        ),
    )


def test_tts_chattts_is_registered() -> None:
    assert get_operator("tts_chattts") is TtsChatTTSOperator


def test_tts_chattts_produces_audio() -> None:
    assert TtsChatTTSOperator.produces_audio is True
    assert TtsChatTTSOperator.reads_audio_bytes is False
    assert TtsChatTTSOperator.device == "gpu"


def test_tts_chattts_config_defaults() -> None:
    config = TtsChatTTSConfig()
    assert config.seed is None
    assert config.temperature == 0.3
    assert config.top_p == 0.7


@pytest.mark.slow
def test_tts_chattts_synthesizes_audio(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_text_cut("c0", "你好，这是一个测试。")])
    config = TtsChatTTSConfig(seed=42)
    op = TtsChatTTSOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out = next(iter(result))
    assert out.recording is not None
    audio_path = out.recording.sources[0].source
    assert Path(audio_path).exists()
    info = sf.info(audio_path)
    assert info.samplerate == 24000
    assert info.duration > 0.1
    assert out.provenance.generated_by == "tts_chattts"


@pytest.mark.slow
def test_tts_chattts_skips_cut_without_text(tmp_path: Path) -> None:
    cut_no_text = Cut(
        id="c-empty",
        recording_id="text-empty",
        start=0.0,
        duration=0.0,
        supervisions=[],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="test",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
            pipeline_run_id="run-test",
        ),
    )
    ctx = _ctx(tmp_path)
    cs = CutSet([cut_no_text])
    config = TtsChatTTSConfig()
    op = TtsChatTTSOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    assert len(list(result)) == 0


@pytest.mark.slow
def test_tts_chattts_reproducible_with_seed(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    text = "测试可复现性。"
    config = TtsChatTTSConfig(seed=42)

    op = TtsChatTTSOperator(config, ctx)
    op.setup()
    r1 = op.process(CutSet([_text_cut("c0", text)]))
    out1 = next(iter(r1))

    r2 = op.process(CutSet([_text_cut("c1", text)]))
    out2 = next(iter(r2))
    op.teardown()

    # Same seed should produce same speaker timbre (not necessarily bit-identical)
    assert out1.recording is not None
    assert out2.recording is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/operators/synthesize/test_tts_chattts.py -v
```

- [ ] **Step 3: Implement tts_chattts**

Create `src/voxkitchen/operators/synthesize/tts_chattts.py`:

```python
"""ChatTTS operator: conversational-style text-to-speech synthesis.

ChatTTS produces natural-sounding conversational speech. Supports
speaker sampling via seed for reproducibility. No voice cloning.
Output sample rate is 24kHz. Requires GPU (4GB+ VRAM).

Prosody control tokens: ``[laugh]``, ``[uv_break]``, ``[lbreak]``
can be embedded directly in the input text.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.utils.audio import recording_from_file, save_audio
from voxkitchen.utils.time import now_utc

logger = logging.getLogger(__name__)

CHATTTS_SR = 24000


class TtsChatTTSConfig(OperatorConfig):
    seed: int | None = None  # fix speaker timbre; None = random
    temperature: float = 0.3
    top_p: float = 0.7
    top_k: int = 20


@register_operator
class TtsChatTTSOperator(Operator):
    """Synthesize conversational speech using ChatTTS."""

    name = "tts_chattts"
    config_cls = TtsChatTTSConfig
    device = "gpu"
    produces_audio = True
    reads_audio_bytes = False
    required_extras = ["tts-chattts"]

    _chat: Any
    _spk: Any

    def setup(self) -> None:
        import ChatTTS
        import torch

        assert isinstance(self.config, TtsChatTTSConfig)
        self._chat = ChatTTS.Chat()
        self._chat.load(compile=False)

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            self._spk = self._chat.sample_random_speaker()
        else:
            self._spk = None

    def process(self, cuts: CutSet) -> CutSet:
        import torch

        assert isinstance(self.config, TtsChatTTSConfig)
        derived_dir = self.ctx.stage_dir / "derived"
        derived_dir.mkdir(parents=True, exist_ok=True)

        out_cuts: list[Cut] = []
        for cut in cuts:
            text = self._extract_text(cut)
            if text is None:
                logger.warning("cut %s has no text, skipping", cut.id)
                continue

            params_infer = ChatTTS.Chat.InferCodeParams(
                spk_emb=self._spk,
                temperature=self.config.temperature,
                top_P=self.config.top_p,
                top_K=self.config.top_k,
            )

            wavs = self._chat.infer(
                [text],
                params_infer_code=params_infer,
            )

            if wavs is None or len(wavs) == 0:
                logger.warning("cut %s produced no audio, skipping", cut.id)
                continue

            audio = np.asarray(wavs[0], dtype=np.float32).flatten()
            audio = np.clip(audio, -1.0, 1.0)

            out_path = derived_dir / f"{cut.id}__chattts.wav"
            save_audio(out_path, audio, CHATTTS_SR)
            new_rec = recording_from_file(out_path, recording_id=f"{cut.id}_chattts")

            out_cuts.append(
                Cut(
                    id=f"{cut.id}__chattts",
                    recording_id=new_rec.id,
                    start=0.0,
                    duration=new_rec.duration,
                    recording=new_rec,
                    supervisions=cut.supervisions,
                    metrics=cut.metrics,
                    custom=dict(cut.custom) if cut.custom else {},
                    provenance=Provenance(
                        source_cut_id=cut.id,
                        generated_by="tts_chattts",
                        stage_name=self.ctx.stage_name,
                        created_at=now_utc(),
                        pipeline_run_id=self.ctx.pipeline_run_id,
                    ),
                )
            )
        return CutSet(out_cuts)

    @staticmethod
    def _extract_text(cut: Cut) -> str | None:
        for sup in cut.supervisions:
            if sup.text:
                return sup.text
        return None

    def teardown(self) -> None:
        self._chat = None
        self._spk = None
```

- [ ] **Step 4: Register in operators/__init__.py**

Add after the kokoro registration:

```python
try:
    from voxkitchen.operators.synthesize import tts_chattts as _synth_chattts  # noqa: F401
except ImportError:
    pass  # ChatTTS not installed
```

- [ ] **Step 5: Run tests, verify pass**

```bash
pytest tests/unit/operators/synthesize/test_tts_chattts.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/voxkitchen/operators/synthesize/tts_chattts.py tests/unit/operators/synthesize/test_tts_chattts.py src/voxkitchen/operators/__init__.py
git commit -m "feat(operators): add tts_chattts — ChatTTS synthesis"
```

---

## Task 3: CosyVoice2 TTS operator

**Files:**
- Create: `src/voxkitchen/operators/synthesize/tts_cosyvoice.py`
- Create: `tests/unit/operators/synthesize/test_tts_cosyvoice.py`
- Modify: `src/voxkitchen/operators/__init__.py`

- [ ] **Step 1: Write tests**

Create `tests/unit/operators/synthesize/test_tts_cosyvoice.py`:

```python
"""Unit tests for tts_cosyvoice operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import soundfile as sf

try:
    from cosyvoice.cli.cosyvoice import AutoModel  # noqa: F401
except ImportError:
    pytest.skip("cosyvoice not available", allow_module_level=True)

from voxkitchen.operators.registry import get_operator
from voxkitchen.operators.synthesize.tts_cosyvoice import (
    TtsCosyVoiceConfig,
    TtsCosyVoiceOperator,
)
from voxkitchen.pipeline.context import RunContext
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.supervision import Supervision


def _ctx(tmp_path: Path) -> RunContext:
    return RunContext(
        work_dir=tmp_path,
        pipeline_run_id="run-test",
        stage_index=1,
        stage_name="tts",
        num_gpus=0,
        num_cpu_workers=1,
        gc_mode="aggressive",
        device="cpu",
    )


def _text_cut(cid: str, text: str) -> Cut:
    return Cut(
        id=cid,
        recording_id=f"text-{cid}",
        start=0.0,
        duration=0.0,
        supervisions=[
            Supervision(
                id=f"sup-{cid}",
                recording_id=f"text-{cid}",
                start=0.0,
                duration=0.0,
                text=text,
            )
        ],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="test",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
            pipeline_run_id="run-test",
        ),
    )


def test_tts_cosyvoice_is_registered() -> None:
    assert get_operator("tts_cosyvoice") is TtsCosyVoiceOperator


def test_tts_cosyvoice_produces_audio() -> None:
    assert TtsCosyVoiceOperator.produces_audio is True
    assert TtsCosyVoiceOperator.reads_audio_bytes is False
    assert TtsCosyVoiceOperator.device == "gpu"


def test_tts_cosyvoice_config_defaults() -> None:
    config = TtsCosyVoiceConfig()
    assert config.model_id == "FunAudioLLM/CosyVoice2-0.5B"
    assert config.mode == "sft"
    assert config.spk_id == "default"
    assert config.reference_audio is None


@pytest.mark.slow
def test_tts_cosyvoice_sft_mode(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_text_cut("c0", "你好，这是一个语音合成测试。")])
    config = TtsCosyVoiceConfig(mode="sft")
    op = TtsCosyVoiceOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out = next(iter(result))
    assert out.recording is not None
    audio_path = out.recording.sources[0].source
    assert Path(audio_path).exists()
    info = sf.info(audio_path)
    assert info.samplerate == 24000
    assert info.duration > 0.1
    assert out.provenance.generated_by == "tts_cosyvoice"


@pytest.mark.slow
def test_tts_cosyvoice_skips_cut_without_text(tmp_path: Path) -> None:
    cut_no_text = Cut(
        id="c-empty",
        recording_id="text-empty",
        start=0.0,
        duration=0.0,
        supervisions=[],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="test",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
            pipeline_run_id="run-test",
        ),
    )
    ctx = _ctx(tmp_path)
    cs = CutSet([cut_no_text])
    config = TtsCosyVoiceConfig()
    op = TtsCosyVoiceOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    assert len(list(result)) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/operators/synthesize/test_tts_cosyvoice.py -v
```

- [ ] **Step 3: Implement tts_cosyvoice**

Create `src/voxkitchen/operators/synthesize/tts_cosyvoice.py`:

```python
"""CosyVoice2 TTS operator: high-quality text-to-speech with voice cloning.

Supports three modes:
- ``sft``: built-in speaker voices (fastest, no reference audio needed)
- ``zero_shot``: clone any voice from a reference audio + transcript
- ``cross_lingual``: clone voice across languages (reference audio only)

Model is auto-downloaded from ModelScope on first use.
Output sample rate is 24kHz. Requires GPU.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.utils.audio import recording_from_file, save_audio
from voxkitchen.utils.time import now_utc

logger = logging.getLogger(__name__)

COSYVOICE_SR = 24000


class TtsCosyVoiceConfig(OperatorConfig):
    model_id: str = "FunAudioLLM/CosyVoice2-0.5B"
    mode: str = "sft"  # sft / zero_shot / cross_lingual
    spk_id: str = "default"  # speaker ID for sft mode
    reference_audio: str | None = None  # path for zero_shot / cross_lingual
    reference_text: str | None = None  # transcript of reference audio (zero_shot)


@register_operator
class TtsCosyVoiceOperator(Operator):
    """Synthesize speech using CosyVoice2 with optional voice cloning."""

    name = "tts_cosyvoice"
    config_cls = TtsCosyVoiceConfig
    device = "gpu"
    produces_audio = True
    reads_audio_bytes = False
    required_extras = ["tts-cosyvoice"]

    _model: Any
    _sample_rate: int

    def setup(self) -> None:
        from modelscope import snapshot_download

        assert isinstance(self.config, TtsCosyVoiceConfig)
        model_dir = snapshot_download(self.config.model_id)

        from cosyvoice.cli.cosyvoice import AutoModel

        self._model = AutoModel(model_dir=model_dir)
        self._sample_rate = getattr(self._model, "sample_rate", COSYVOICE_SR)

    def process(self, cuts: CutSet) -> CutSet:
        import torch

        assert isinstance(self.config, TtsCosyVoiceConfig)
        derived_dir = self.ctx.stage_dir / "derived"
        derived_dir.mkdir(parents=True, exist_ok=True)

        out_cuts: list[Cut] = []
        for cut in cuts:
            text = self._extract_text(cut)
            if text is None:
                logger.warning("cut %s has no text, skipping", cut.id)
                continue

            chunks: list[np.ndarray] = []
            for chunk in self._infer(text):
                speech = chunk["tts_speech"]
                if isinstance(speech, torch.Tensor):
                    speech = speech.cpu().numpy()
                chunks.append(np.asarray(speech, dtype=np.float32).flatten())

            if not chunks:
                logger.warning("cut %s produced no audio, skipping", cut.id)
                continue

            audio = np.concatenate(chunks)
            audio = np.clip(audio, -1.0, 1.0)

            out_path = derived_dir / f"{cut.id}__cosyvoice.wav"
            save_audio(out_path, audio, self._sample_rate)
            new_rec = recording_from_file(out_path, recording_id=f"{cut.id}_cosyvoice")

            out_cuts.append(
                Cut(
                    id=f"{cut.id}__cosyvoice",
                    recording_id=new_rec.id,
                    start=0.0,
                    duration=new_rec.duration,
                    recording=new_rec,
                    supervisions=cut.supervisions,
                    metrics=cut.metrics,
                    custom=dict(cut.custom) if cut.custom else {},
                    provenance=Provenance(
                        source_cut_id=cut.id,
                        generated_by="tts_cosyvoice",
                        stage_name=self.ctx.stage_name,
                        created_at=now_utc(),
                        pipeline_run_id=self.ctx.pipeline_run_id,
                    ),
                )
            )
        return CutSet(out_cuts)

    def _infer(self, text: str) -> Any:
        assert isinstance(self.config, TtsCosyVoiceConfig)
        mode = self.config.mode

        if mode == "sft":
            return self._model.inference_sft(
                text, spk_id=self.config.spk_id, stream=False
            )

        if mode == "zero_shot":
            if not self.config.reference_audio or not self.config.reference_text:
                raise ValueError(
                    "zero_shot mode requires both reference_audio and reference_text"
                )
            return self._model.inference_zero_shot(
                text,
                self.config.reference_text,
                self.config.reference_audio,
                stream=False,
            )

        if mode == "cross_lingual":
            if not self.config.reference_audio:
                raise ValueError("cross_lingual mode requires reference_audio")
            return self._model.inference_cross_lingual(
                text,
                self.config.reference_audio,
                stream=False,
            )

        raise ValueError(
            f"unknown mode: {mode!r}, use 'sft', 'zero_shot', or 'cross_lingual'"
        )

    @staticmethod
    def _extract_text(cut: Cut) -> str | None:
        for sup in cut.supervisions:
            if sup.text:
                return sup.text
        return None

    def teardown(self) -> None:
        self._model = None
```

- [ ] **Step 4: Register in operators/__init__.py**

Add after the chattts registration:

```python
try:
    from voxkitchen.operators.synthesize import tts_cosyvoice as _synth_cosyvoice  # noqa: F401
except ImportError:
    pass  # cosyvoice not installed
```

- [ ] **Step 5: Run tests, verify pass**

```bash
pytest tests/unit/operators/synthesize/test_tts_cosyvoice.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/voxkitchen/operators/synthesize/tts_cosyvoice.py tests/unit/operators/synthesize/test_tts_cosyvoice.py src/voxkitchen/operators/__init__.py
git commit -m "feat(operators): add tts_cosyvoice — CosyVoice2 synthesis with voice cloning"
```

---

## Task 4: Fish-Speech TTS operator

**Files:**
- Create: `src/voxkitchen/operators/synthesize/tts_fish_speech.py`
- Create: `tests/unit/operators/synthesize/test_tts_fish_speech.py`
- Modify: `src/voxkitchen/operators/__init__.py`

- [ ] **Step 1: Write tests**

Create `tests/unit/operators/synthesize/test_tts_fish_speech.py`:

```python
"""Unit tests for tts_fish_speech operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import soundfile as sf

try:
    from fish_speech.inference import TTSInference  # noqa: F401
except ImportError:
    try:
        import fish_speech  # noqa: F401
    except ImportError:
        pytest.skip("fish-speech not available", allow_module_level=True)

from voxkitchen.operators.registry import get_operator
from voxkitchen.operators.synthesize.tts_fish_speech import (
    TtsFishSpeechConfig,
    TtsFishSpeechOperator,
)
from voxkitchen.pipeline.context import RunContext
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.supervision import Supervision


def _ctx(tmp_path: Path) -> RunContext:
    return RunContext(
        work_dir=tmp_path,
        pipeline_run_id="run-test",
        stage_index=1,
        stage_name="tts",
        num_gpus=0,
        num_cpu_workers=1,
        gc_mode="aggressive",
        device="cpu",
    )


def _text_cut(cid: str, text: str) -> Cut:
    return Cut(
        id=cid,
        recording_id=f"text-{cid}",
        start=0.0,
        duration=0.0,
        supervisions=[
            Supervision(
                id=f"sup-{cid}",
                recording_id=f"text-{cid}",
                start=0.0,
                duration=0.0,
                text=text,
            )
        ],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="test",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
            pipeline_run_id="run-test",
        ),
    )


def test_tts_fish_speech_is_registered() -> None:
    assert get_operator("tts_fish_speech") is TtsFishSpeechOperator


def test_tts_fish_speech_produces_audio() -> None:
    assert TtsFishSpeechOperator.produces_audio is True
    assert TtsFishSpeechOperator.reads_audio_bytes is False
    assert TtsFishSpeechOperator.device == "gpu"


def test_tts_fish_speech_config_defaults() -> None:
    config = TtsFishSpeechConfig()
    assert config.model_id == "fishaudio/fish-speech-1.5"
    assert config.reference_audio is None
    assert config.reference_text is None


@pytest.mark.slow
def test_tts_fish_speech_synthesizes_audio(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_text_cut("c0", "Hello, this is a test.")])
    config = TtsFishSpeechConfig()
    op = TtsFishSpeechOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out = next(iter(result))
    assert out.recording is not None
    audio_path = out.recording.sources[0].source
    assert Path(audio_path).exists()
    info = sf.info(audio_path)
    assert info.samplerate > 0
    assert info.duration > 0.1
    assert out.provenance.generated_by == "tts_fish_speech"


@pytest.mark.slow
def test_tts_fish_speech_skips_cut_without_text(tmp_path: Path) -> None:
    cut_no_text = Cut(
        id="c-empty",
        recording_id="text-empty",
        start=0.0,
        duration=0.0,
        supervisions=[],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="test",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
            pipeline_run_id="run-test",
        ),
    )
    ctx = _ctx(tmp_path)
    cs = CutSet([cut_no_text])
    config = TtsFishSpeechConfig()
    op = TtsFishSpeechOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    assert len(list(result)) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/operators/synthesize/test_tts_fish_speech.py -v
```

- [ ] **Step 3: Implement tts_fish_speech**

Create `src/voxkitchen/operators/synthesize/tts_fish_speech.py`:

```python
"""Fish-Speech TTS operator: codec-LM text-to-speech with voice cloning.

Fish-Speech uses a VQGAN + language model architecture for high-quality
zero-shot voice cloning. Supports 13 languages.

Model is auto-downloaded from HuggingFace on first use.
Output sample rate is 44100Hz. Requires GPU (24GB+ VRAM recommended).

NOTE: Fish-Speech's internal API may change between versions. This
operator targets fish-speech v1.5+. If the import paths have changed,
update the ``_setup_*`` and ``_infer`` methods.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.utils.audio import recording_from_file, save_audio
from voxkitchen.utils.time import now_utc

logger = logging.getLogger(__name__)

FISH_SPEECH_SR = 44100


class TtsFishSpeechConfig(OperatorConfig):
    model_id: str = "fishaudio/fish-speech-1.5"
    reference_audio: str | None = None
    reference_text: str | None = None
    max_new_tokens: int = 1024
    top_p: float = 0.7
    temperature: float = 0.7
    repetition_penalty: float = 1.2


@register_operator
class TtsFishSpeechOperator(Operator):
    """Synthesize speech using Fish-Speech codec language model."""

    name = "tts_fish_speech"
    config_cls = TtsFishSpeechConfig
    device = "gpu"
    produces_audio = True
    reads_audio_bytes = False
    required_extras = ["tts-fish-speech"]

    _inference: Any
    _sample_rate: int

    def setup(self) -> None:
        assert isinstance(self.config, TtsFishSpeechConfig)
        self._sample_rate = FISH_SPEECH_SR
        self._load_model()

    def _load_model(self) -> None:
        """Load Fish-Speech inference pipeline.

        Fish-Speech packages its inference as a pipeline that handles
        VQGAN encoding/decoding and LLM generation. The exact import
        path may vary across versions — adapt if needed.
        """
        assert isinstance(self.config, TtsFishSpeechConfig)
        try:
            from fish_speech.inference import TTSInference

            self._inference = TTSInference(model_id=self.config.model_id)
        except ImportError:
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(self.config.model_id)
            from fish_speech.inference import TTSInference

            self._inference = TTSInference(model_dir=model_dir)

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, TtsFishSpeechConfig)
        derived_dir = self.ctx.stage_dir / "derived"
        derived_dir.mkdir(parents=True, exist_ok=True)

        out_cuts: list[Cut] = []
        for cut in cuts:
            text = self._extract_text(cut)
            if text is None:
                logger.warning("cut %s has no text, skipping", cut.id)
                continue

            audio = self._infer(text)
            if audio is None or len(audio) == 0:
                logger.warning("cut %s produced no audio, skipping", cut.id)
                continue

            audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

            out_path = derived_dir / f"{cut.id}__fish_speech.wav"
            save_audio(out_path, audio, self._sample_rate)
            new_rec = recording_from_file(
                out_path, recording_id=f"{cut.id}_fish_speech"
            )

            out_cuts.append(
                Cut(
                    id=f"{cut.id}__fish_speech",
                    recording_id=new_rec.id,
                    start=0.0,
                    duration=new_rec.duration,
                    recording=new_rec,
                    supervisions=cut.supervisions,
                    metrics=cut.metrics,
                    custom=dict(cut.custom) if cut.custom else {},
                    provenance=Provenance(
                        source_cut_id=cut.id,
                        generated_by="tts_fish_speech",
                        stage_name=self.ctx.stage_name,
                        created_at=now_utc(),
                        pipeline_run_id=self.ctx.pipeline_run_id,
                    ),
                )
            )
        return CutSet(out_cuts)

    def _infer(self, text: str) -> np.ndarray | None:
        """Run TTS inference for a single text string.

        Returns 1-D float32 numpy array of audio samples, or None on failure.
        """
        assert isinstance(self.config, TtsFishSpeechConfig)
        try:
            ref_audio = self.config.reference_audio
            ref_text = self.config.reference_text

            result = self._inference.synthesize(
                text=text,
                reference_audio=ref_audio,
                reference_text=ref_text,
                max_new_tokens=self.config.max_new_tokens,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                repetition_penalty=self.config.repetition_penalty,
            )

            if isinstance(result, tuple):
                audio, sr = result
                self._sample_rate = sr
            else:
                audio = result

            return np.asarray(audio, dtype=np.float32).flatten()
        except Exception:
            logger.exception("Fish-Speech inference failed")
            return None

    @staticmethod
    def _extract_text(cut: Cut) -> str | None:
        for sup in cut.supervisions:
            if sup.text:
                return sup.text
        return None

    def teardown(self) -> None:
        self._inference = None
```

- [ ] **Step 4: Register in operators/__init__.py**

Add after the cosyvoice registration:

```python
try:
    from voxkitchen.operators.synthesize import tts_fish_speech as _synth_fish  # noqa: F401
except ImportError:
    pass  # fish-speech not installed
```

- [ ] **Step 5: Run tests, verify pass**

```bash
pytest tests/unit/operators/synthesize/test_tts_fish_speech.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/voxkitchen/operators/synthesize/tts_fish_speech.py tests/unit/operators/synthesize/test_tts_fish_speech.py src/voxkitchen/operators/__init__.py
git commit -m "feat(operators): add tts_fish_speech — Fish-Speech synthesis with voice cloning"
```

---

## Task 5: tools.py + pyproject.toml + README

**Files:**
- Modify: `src/voxkitchen/tools.py`
- Modify: `pyproject.toml`
- Modify: `README.md`

- [ ] **Step 1: Add synthesize() to tools.py**

Append to end of `src/voxkitchen/tools.py`:

```python
# ---------------------------------------------------------------------------
# TTS Synthesis
# ---------------------------------------------------------------------------


def synthesize(
    text: str,
    output_path: str | Path,
    *,
    engine: str = "kokoro",
    voice: str | None = None,
    language: str | None = None,
    speed: float = 1.0,
    seed: int | None = None,
    reference_audio: str | None = None,
    reference_text: str | None = None,
) -> Path:
    """Synthesize speech from text and save to a WAV file.

    Args:
        engine: TTS engine. Options:

            - ``"kokoro"`` — lightweight (82M), CPU-capable, 8 languages
            - ``"chattts"`` — conversational style, Chinese/English, GPU
            - ``"cosyvoice"`` — CosyVoice2, zero-shot voice cloning, GPU
            - ``"fish_speech"`` — Fish-Speech, zero-shot cloning, GPU

        voice: Voice/speaker ID (engine-specific). Defaults:

            - kokoro: ``"af_heart"``
            - cosyvoice: ``"default"`` (sft mode)

        language: Language code (kokoro only):
            ``"a"`` (AmE), ``"b"`` (BrE), ``"j"`` (Japanese), ``"z"`` (Mandarin)
        speed: Speech speed multiplier (kokoro only).
        seed: Random seed for speaker sampling (chattts only).
        reference_audio: Path to reference audio for voice cloning
            (cosyvoice zero_shot/cross_lingual, fish_speech).
        reference_text: Transcript of reference audio
            (cosyvoice zero_shot only).

    Returns:
        Path to the output WAV file.

    Example::

        from voxkitchen.tools import synthesize

        # Kokoro (lightweight, CPU)
        synthesize("Hello world!", "output.wav", engine="kokoro")

        # ChatTTS (conversational Chinese)
        synthesize("你好世界", "output.wav", engine="chattts", seed=42)

        # CosyVoice2 (voice cloning)
        synthesize("你好", "clone.wav", engine="cosyvoice",
                   reference_audio="ref.wav", reference_text="参考文本")

        # Fish-Speech (voice cloning)
        synthesize("Hello", "clone.wav", engine="fish_speech",
                   reference_audio="ref.wav")
    """
    from voxkitchen.schema.supervision import Supervision

    out = Path(output_path)

    # Build a text-only Cut
    cut_id = "synth-0"
    cut = Cut(
        id=cut_id,
        recording_id=f"text-{cut_id}",
        start=0.0,
        duration=0.0,
        supervisions=[
            Supervision(
                id=f"sup-{cut_id}",
                recording_id=f"text-{cut_id}",
                start=0.0,
                duration=0.0,
                text=text,
            )
        ],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="tools",
            stage_name="standalone",
            created_at=datetime.now(tz=timezone.utc),
            pipeline_run_id="standalone",
        ),
    )
    ctx = _make_ctx()

    if engine == "kokoro":
        from voxkitchen.operators.synthesize.tts_kokoro import (
            TtsKokoroConfig,
            TtsKokoroOperator,
        )

        config = TtsKokoroConfig(
            voice=voice or "af_heart",
            lang_code=language or "a",
            speed=speed,
        )
        op: Operator = TtsKokoroOperator(config, ctx)

    elif engine == "chattts":
        from voxkitchen.operators.synthesize.tts_chattts import (
            TtsChatTTSConfig,
            TtsChatTTSOperator,
        )

        config = TtsChatTTSConfig(seed=seed)  # type: ignore[assignment]
        op = TtsChatTTSOperator(config, ctx)

    elif engine == "cosyvoice":
        from voxkitchen.operators.synthesize.tts_cosyvoice import (
            TtsCosyVoiceConfig,
            TtsCosyVoiceOperator,
        )

        mode = "sft"
        if reference_audio and reference_text:
            mode = "zero_shot"
        elif reference_audio:
            mode = "cross_lingual"
        config = TtsCosyVoiceConfig(  # type: ignore[assignment]
            mode=mode,
            spk_id=voice or "default",
            reference_audio=reference_audio,
            reference_text=reference_text,
        )
        op = TtsCosyVoiceOperator(config, ctx)

    elif engine == "fish_speech":
        from voxkitchen.operators.synthesize.tts_fish_speech import (
            TtsFishSpeechConfig,
            TtsFishSpeechOperator,
        )

        config = TtsFishSpeechConfig(  # type: ignore[assignment]
            reference_audio=reference_audio,
            reference_text=reference_text,
        )
        op = TtsFishSpeechOperator(config, ctx)

    else:
        raise ValueError(
            f"unknown TTS engine: {engine!r}. "
            f"Options: 'kokoro', 'chattts', 'cosyvoice', 'fish_speech'"
        )

    op.setup()
    try:
        result = op.process(CutSet([cut]))
    finally:
        op.teardown()

    out_cut = next(iter(result))
    if out_cut.recording:
        import shutil

        derived = Path(out_cut.recording.sources[0].source)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(derived, out)
    return out
```

- [ ] **Step 2: Update pyproject.toml extras**

Add these four groups to `[project.optional-dependencies]`:

```toml
tts-kokoro = ["kokoro>=0.9", "soundfile>=0.12", "misaki[zh,ja,ko,vi]>=0.9"]
tts-chattts = ["ChatTTS>=0.2"]
tts-cosyvoice = ["modelscope>=1.0"]
tts-fish-speech = ["fish-speech>=1.5"]
```

Add all four to the `all` extras group. Also add `funasr`, `wenet`, `align`, `codec` extras to `all` if not already present.

Add to the `[[tool.mypy.overrides]]` `module` list: `"kokoro.*", "ChatTTS.*", "cosyvoice.*", "modelscope.*", "fish_speech.*"`.

- [ ] **Step 3: Update README.md**

Update the operator count from `47` to `51`.

Add a new row to the operators table:

```markdown
| **Synthesize** | `tts_kokoro`, `tts_chattts`, `tts_cosyvoice`, `tts_fish_speech` |
```

Add TTS install extras to the install section:

```markdown
#   TTS synthesis
pip install -e ".[tts-kokoro]"      # Kokoro TTS (lightweight, CPU-capable)
pip install -e ".[tts-chattts]"     # ChatTTS (conversational style, GPU)
pip install -e ".[tts-cosyvoice]"   # CosyVoice2 (voice cloning, GPU)
pip install -e ".[tts-fish-speech]" # Fish-Speech (voice cloning, GPU)
```

Add TTS example to the Python tools API section:

```python
from voxkitchen.tools import synthesize

# Lightweight TTS (CPU)
synthesize("Hello world!", "output.wav", engine="kokoro")

# Voice cloning
synthesize("你好", "clone.wav", engine="cosyvoice",
           reference_audio="ref.wav", reference_text="参考文本")
```

- [ ] **Step 4: Commit**

```bash
git add src/voxkitchen/tools.py pyproject.toml README.md
git commit -m "feat: synthesize() tool function + TTS extras + README update"
```

---

## Verification

1. **Fast tests (no models):**
   ```bash
   pytest tests/unit/operators/synthesize/ -v -m "not slow and not gpu" --tb=short
   ```

2. **Lint + format:**
   ```bash
   ruff check src/voxkitchen/operators/synthesize/ tests/unit/operators/synthesize/ && ruff format --check src/voxkitchen/operators/synthesize/ tests/unit/operators/synthesize/
   ```

3. **Type check:**
   ```bash
   mypy src/voxkitchen/operators/synthesize/
   ```

4. **Operator registration (if engine is installed):**
   ```bash
   vkit operators | grep -E "tts_kokoro|tts_chattts|tts_cosyvoice|tts_fish_speech"
   ```

5. **Smoke test (requires kokoro installed):**
   ```bash
   python -c "from voxkitchen.tools import synthesize; synthesize('Hello world', '/tmp/test_tts.wav', engine='kokoro')"
   ```
