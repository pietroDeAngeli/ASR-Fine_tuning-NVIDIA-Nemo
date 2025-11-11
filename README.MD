# Fine-tuning ASR Models for Low-Resource Languages

This project explores how to fine-tune Automatic Speech Recognition (ASR) models for new languages using limited data, particularly Italian, with NVIDIA NeMo and open-source tools.

---

## Topics Covered

- Transfer learning and fine-tuning ASR decoders
- Adapting QuartzNet (English) to Italian audio
- Preprocessing Mozilla Common Voice datasets
- Custom dataset preparation pipeline
- Character set adaptation and augmentation
- Integration of N-gram rescoring with KenLM

---

## Dataset

- **Source**: Mozilla Common Voice 22.0
- **Audio Format**: .mp3 (converted to .wav at 16 kHz mono)
- **Split**: 
- **Link**: [https://datacollective.mozillafoundation.org/datasets/cmflnuzw55y8stkav91ul91ix](https://datacollective.mozillafoundation.org/datasets/cmflnuzw55y8stkav91ul91ix)

---

## Model Used

- `stt_en_quartznet15x5` (CNN-based)
- Pre-trained on 7000h of English audio
- Modified decoder vocabulary to include Italian accented characters

---

## Preprocessing Pipeline

Implemented via `preprocess_dataset.py`:

- `convert_mp3_to_wav()`: Convert and resample audio
- `process_tsv()`: Convert TSV to CSV for manifests
- `delete_invalidated()`: Remove broken samples
- `keep_first_n_lines()`: Trim dataset size for quick testing
- `MVCtoNEMO()`: Convert CSV to NeMo manifest with durations

---

## Fine-Tuning

- Encoder frozen to reuse language-agnostic acoustic features
- Decoder retrained with Italian transcriptions
- Used `Trainer.fit()` from PyTorch Lightning
- Results saved in `.nemo` format

---

## Results

```text
REF: appese le scarpette al chiodo assume il ruolo di allenatore dei portieri
PRE: a pese le scampeta ciodo assume rolo di allenatar di portiri
```

---

## Project Structure

```
dataset/
├── train/validation/test/
├── clips/            # audio files
models/
├── stt_en_quartznet15x5.nemo
utils/
├── preprocess_dataset.py
├── train.py
```

---

## N-gram Rescoring with KenLM

- 70% weight: general Italian dataset
- 30% weight: domain-specific (TV transcripts)
- Language model built using 4-grams, converted to binary format

---

## Results

| Model              | WER (%) | Insertions | Deletions | Substitutions |
|-------------------|---------|------------|-----------|---------------|
| Original model    | 45.62   | 455        | 2070      | 5658          |
| N-gram rescoring  | 40.73   | 264        | 2426      | 4869          |

---

## Notes

- The improvements are limited but consistent
- More data and epochs would yield better WER
- The methods are modular and generalizable to other languages

