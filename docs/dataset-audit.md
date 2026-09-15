# English-to-Hindi dataset audit

Date: 2026-09-15

## Conclusion

No large, clean, fully verified corpus of real English source speech paired with
independently recorded Hindi target speech was found. That is not a blocker for the
selected UnitY-style architecture because its two trainable mappings can use different
sources:

```text
English speech → Hindi semantic tokens
    BhasaAnuvaad/SeamlessAlign + Indic-TEDST

Hindi semantic tokens → Hindi speech units → waveform
    IndicVoices-R + Kathbath
```

Synthetic Hindi target speech may supplement training, but every such row must retain
its TTS model ID, license, checkpoint hash, and `audio_origin=synthetic`. It cannot be
presented as human-recorded parallel speech.

## Accepted shortlist

### BhasaAnuvaad / SeamlessAlign

- Role: primary English-speech→Hindi-text training source.
- Published scale: more than 44,400 aggregate speech hours and 17M aligned text segments
  across supported Indic languages; exact accepted Hindi hours must be computed from the
  downloaded manifests.
- Origin: mixture of curated, web-mined and synthetic/aligned material.
- Published dataset license: CC BY 4.0.
- Required controls: retain source category and alignment score, begin with curated and
  high-confidence Hindi rows, and deduplicate against every evaluation set.
- Sources: [official repository](https://github.com/AI4Bharat/BhasaAnuvaad) and
  [paper](https://arxiv.org/abs/2411.04699).

### Indic-TEDST

- Role: clean bootstrap, development and held-out speech-translation evaluation.
- Hindi split described by the paper: approximately 45.8K train segments / 76.46 hours,
  7.6K validation / 13.52 hours, and 7.2K test / 13.52 hours.
- Origin: real English TED speech with English transcripts and Hindi translations.
- Terms: form/task governed; the paper is not a blanket redistribution license.
- Required controls: preserve official splits and record the accepted access terms.
- Sources: [paper](https://aclanthology.org/2024.lrec-main.790/) and
  [IWSLT Indic task](https://iwslt.org/2025/indic).

### Samanantar

- Role: English→Hindi semantic-decoder pretraining and teacher data.
- Published scale: 49.6M English–Indic pairs overall, with a multi-million-pair Hindi
  portion.
- Origin: existing corpora plus automatically mined web text.
- Licensing caveat: AI4Bharat's packaging is CC0; that does not automatically clear all
  rights in every underlying web sentence.
- Required controls: filter by source, language ID, alignment score, length ratio,
  duplication and Devanagari coverage.
- Sources: [official description](https://github.com/AI4Bharat/indicnlp.ai4bharat.org/blob/master/content/pages/samanantar.md)
  and [paper](https://arxiv.org/abs/2104.05596).

### IndicVoices-R

- Role: primary Hindi speech-unit extraction and multi-speaker vocoder training.
- Published scale: 1,704 hours and 10,496 speakers across 22 languages; Hindi-only hours
  must be calculated from its manifest.
- Origin: real speech enhanced and ASR-filtered for TTS use.
- Published license: CC BY 4.0.
- Required controls: start with clean Hindi speakers/scenarios and retain enhancement
  and confidence metadata.
- Source: [official repository](https://github.com/AI4Bharat/IndicVoices-R).

### Kathbath / IndicSUPERB

- Role: supplementary Hindi unit discovery, pronunciation coverage and robustness.
- Published Hindi scale: approximately 150.2 hours across 121 speakers.
- Origin: real read speech with clean/noisy partitions.
- Licensing caveat: packaging is CC0, while AI4Bharat notes it does not own source text.
- Source: [official repository](https://github.com/AI4Bharat/indicSUPERB).

## Evaluation-only sources

- FLORES-200 `eng_Latn`/`hin_Deva`: professionally translated text dev/devtest,
  CC BY-SA 4.0. Never use it for training or teacher-target generation.
  [Dataset card](https://huggingface.co/datasets/facebook/flores)
- FLEURS English and Hindi: roughly 12 hours / 2,000 utterances per language,
  CC BY 4.0. Useful for speech and pronunciation evaluation, but it is read speech rather
  than conversational S2S data.
  [Dataset card](https://huggingface.co/datasets/google/fleurs)
- Indic-TEDST validation/test: preserve the official boundary and exclude it from all
  pseudo-labelling.

A project-owned, consented conversational set of 200–500 English utterances with
professional Hindi translations and multiple Hindi speakers is recommended for final
product evaluation. It must remain held out permanently.

## Rejected or conditional sources

- CoVoST 2: reject for this direction; it does not provide English→Hindi.
  [Official repository](https://github.com/facebookresearch/covost)
- IIT Bombay English–Hindi: useful for non-commercial learning, but its compiled release
  is CC BY-NC 4.0 and contains mixed-source material. Keep outside any unrestricted
  commercial training set until rights are resolved.
  [Official corpus](https://www.cfilt.iitb.ac.in/~parallelcorp/iitb_en_hi_parallel/)
- IndicTTS: potentially valuable clean single-speaker Hindi TTS data, but access uses
  custom terms that must be recorded and reviewed before adding a downloader.
  [Official database](https://www.iitm.ac.in/donlab/indictts/database)
- GramVaani SLR118: useful for robustness, not first-pass vocoding; academic use is free
  and commercial use requires permission.
  [OpenSLR 118](https://www.openslr.org/118/)

## Unit tokenizer and vocoder recommendation

Use IndicWav2Vec representations with a project-trained k-means tokenizer as the first
Hindi-unit candidate. Benchmark multiple representation layers and initially 500–1,000
clusters. The tokenizer is used to create training targets and need not ship in the
runtime. [IndicWav2Vec](https://github.com/AI4Bharat/IndicWav2Vec)

Train project-owned unit-conditioned HiFi-GAN weights on the selected licensed Hindi
speech. HiFi-GAN code is MIT; that license does not transfer to arbitrary third-party
checkpoints or data. [HiFi-GAN](https://github.com/jik876/hifi-gan)

SeamlessM4T weights are CC BY-NC 4.0. Use them only as a non-commercial reference/teacher
until the intended downstream use and distillation rights receive explicit legal review.
[Seamless Communication](https://github.com/facebookresearch/seamless_communication)

## Mandatory data controls

Every manifest row must retain dataset/subset IDs, source URL and license, real/synthetic
origin, human/mined/synthetic translation origin, alignment score, speaker, duration,
split and content hash. Deduplicate by normalized text and audio fingerprint before
training. Evaluation sets must be excluded from teacher-target generation as well as
student training.
