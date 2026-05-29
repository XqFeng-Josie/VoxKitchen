<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# Shrutilipi

A 6400+ hour labelled ASR corpus across 12 Indian languages mined from All India Radio news bulletins by AI4Bharat, with document-level audio-text alignment.

- **Task:** asr, multilingual
- **Languages:** multi
- **Hours:** 6400
- **Domain:** broadcast/news
- **License:** CC BY 4.0
- **Homepage:** [https://ai4bharat.iitm.ac.in/datasets/shrutilipi](https://ai4bharat.iitm.ac.in/datasets/shrutilipi)
- **Paper:** [https://arxiv.org/abs/2208.12666](https://arxiv.org/abs/2208.12666)

## Recommendation

Best-in-class scale for Indic ASR pretraining and low-resource fine-tuning — Bengali, Hindi, Tamil, Telugu, and other Indian languages underserved by Western corpora. Broadcast-news domain skews formal/read-aloud register, so complement with conversational data for spoken-dialogue use cases.

## Getting the data

Obtain from the [dataset homepage](https://ai4bharat.iitm.ac.in/datasets/shrutilipi).

12 languages — bn, gu, hi, kn, ml, mr, or, pa, sa, ta, te, ur. Also mirrored on Hugging Face as ai4bharat/Shrutilipi.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `examples/pipelines/fleurs-multilingual.yaml` — run it with `vkit docker run`.
