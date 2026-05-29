"""LibriTTS-R recipe: parse a local LibriTTS-R directory into a CutSet.

LibriTTS-R is a sound-quality-restored version of LibriTTS — same 585 hours,
same 2456 speakers, same per-utterance splits and texts, but enhanced via
Google's Miipher speech-restoration model. It is the current default
large-scale multi-speaker English TTS corpus when audio fidelity matters
(zero-shot TTS, neural codec training, voice cloning).

The on-disk layout is identical to LibriTTS — same ``<subset>/<spk>/<chapter>``
hierarchy, same ``.normalized.txt`` / ``.original.txt`` files, same
``speakers.tsv`` — except the top-level directory is ``LibriTTS_R/``. This
recipe therefore reuses every parsing rule from :class:`LibriTTSRecipe`
unchanged and overrides only the download targets and the top-dir hook.

Subset compressed sizes (HEAD-probed 2026-05, OpenSLR resource 141)::

    dev-clean         1.3 GB   dev-other         930 MB
    test-clean        1.2 GB   test-other        965 MB
    train-clean-100   7.6 GB   train-clean-360   27 GB
    train-other-500   44 GB
"""

from __future__ import annotations

from voxkitchen.ingest.recipes import register_recipe
from voxkitchen.ingest.recipes.libritts import LibriTTSRecipe


class LibriTTSRRecipe(LibriTTSRecipe):
    """Parse LibriTTS-R into a CutSet (LibriTTS layout, restored audio)."""

    name = "libritts_r"
    _top_dir = "LibriTTS_R"
    _provenance_tag = "libritts_r_recipe@1"

    # OpenSLR resource 141. Subset *names* match LibriTTS (``train-clean-100``,
    # etc.) but the tarballs use underscores in their filenames (``train_clean_100.tar.gz``).
    download_urls = {
        "dev-clean": ["https://www.openslr.org/resources/141/dev_clean.tar.gz"],
        "dev-other": ["https://www.openslr.org/resources/141/dev_other.tar.gz"],
        "test-clean": ["https://www.openslr.org/resources/141/test_clean.tar.gz"],
        "test-other": ["https://www.openslr.org/resources/141/test_other.tar.gz"],
        "train-clean-100": ["https://www.openslr.org/resources/141/train_clean_100.tar.gz"],
        "train-clean-360": ["https://www.openslr.org/resources/141/train_clean_360.tar.gz"],
        "train-other-500": ["https://www.openslr.org/resources/141/train_other_500.tar.gz"],
    }
    # HEAD-probed Content-Length values (2026-05) on www.openslr.org/141.
    download_sizes = {
        "dev-clean": 1_357_352_348,
        "dev-other": 975_450_334,
        "test-clean": 1_294_768_623,
        "test-other": 1_011_954_678,
        "train-clean-100": 8_129_817_509,
        "train-clean-360": 28_953_983_271,
        "train-other-500": 46_843_185_428,
    }


register_recipe(LibriTTSRRecipe())
