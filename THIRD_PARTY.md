# Third-party data and model attribution

The Apache-2.0 license in `LICENSE` applies to the repository's original code.
It does not replace the licenses or access conditions of third-party datasets,
model weights, or artifacts derived from them.

## WordProject / BhasaAnuvaad benchmark data

The benchmark manifests, English/Hindi reference excerpts, and benchmark audio
archives under `teacher_a_100_artifacts/` and `teacher_b_100_artifacts/` originate
from AI4Bharat's WordProject subset of BhasaAnuvaad.

- Attribution: AI4Bharat and the BhasaAnuvaad contributors, Jain et al. (2024);
  original source material is identified by the dataset as WordProject.
- Dataset: https://huggingface.co/datasets/ai4bharat/WordProject
- Paper and contributor list: https://arxiv.org/abs/2411.04699
- Original source: https://www.wordproject.org/
- Dataset-declared license: Creative Commons Attribution 4.0 International,
  https://creativecommons.org/licenses/by/4.0/
- Dataset revision used: `82398593980643a685816a77236adaa05591b12d`.
- Changes: a deterministic 100-example validation subset was selected; audio was
  decoded/resampled to mono 16 kHz PCM WAV and packaged for benchmarking. Human
  reference text is retained for evaluation. Teacher-generated predictions are
  separate from human references.

The two benchmark archives contain the same shared audio selection. Their
manifests, IDs, and hashes record its provenance. These audio files are dataset
material, not audio recordings contributed by this project's users. Retain source
attribution and change notices when reusing them. Access to the upstream dataset
requires following its published access process.

## Teacher model references

Weights are not included in this repository. Existing experiment metadata records
the source revisions and generation configurations.

| Model | Role | Source-declared license |
| --- | --- | --- |
| [Whisper Large-v3 Turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | Teacher A speech recognition | MIT |
| [IndicTrans2 English–Indic 1B](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B) | Teacher A English-to-Hindi text translation | MIT |
| [SeamlessM4T-v2 Large](https://huggingface.co/facebook/seamless-m4t-v2-large) | Teacher B speech-to-Hindi-text translation | CC BY-NC 4.0 |
| [Whisper Tiny](https://huggingface.co/openai/whisper-tiny) | Separate frozen-encoder experiment | Apache-2.0 |

Read each source model's license and access terms before downloading or reusing
weights. Publishing experiment code and scores does not establish permission to
commercialize or redistribute teacher outputs or distilled checkpoints. The
repository does not distribute student weights or make that downstream claim.

Other dataset preparations, including Samanantar and IndicVoices-R, retain their
own provenance in `dataset/catalog.json` and `docs/dataset-audit.md`.

## Explanatory images

The two images in `docs/images/` were generated for this project using an image
generation tool. Their prompts are retained locally in the ignored
`docs/images/prompts.md` file. They
illustrate the code's architecture and default experiment workflow; they are not
measured attention maps, model-generated translations, or benchmark results.
