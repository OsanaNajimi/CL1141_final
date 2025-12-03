# English-Chinese Name Phonetic Distance

This tool computes the phonetic distance between an English name and a Traditional Chinese name to evaluate transliteration quality.

## Algorithm

1.  **English Processing**:

    - **IPA Transcription**: Converts English names to IPA using `phonemizer`, preserving stress.
    - **Aspiration**: Adds aspiration (`ʰ`) to voiceless stops (`p`, `t`, `k`) at word beginnings or stressed syllables.
    - **Syllabization**: Segments IPA into Chinese-like syllables (Onset-Nucleus-Coda structure) using a custom greedy tokenizer.

2.  **Chinese Processing**:

    - Converts Traditional Chinese characters to Pinyin (`pypinyin`) and then to IPA (`epitran`).

3.  **Distance Computation**:
    - **Syllable Distance**: Decomposes each syllable into **Onset**, **Nucleus**, and **Coda**.
      - Calculates the sum of feature edit distances (`panphon`) for corresponding parts.
      - **Penalty**: If a part is missing in one syllable but present in the other, a penalty of `0.25` is applied. If both are missing, the cost is `0`.
      - **Special Case**: Maps `/ɚ/` to `/ə/` for feature vector retrieval.
    - **Word Distance**: Uses **Dynamic Programming** to find the minimum weighted edit distance between the two syllable sequences.
    - **Alignment**: Performs backtracking to produce the optimal alignment path and individual syllable costs.

## Requirements

- Python 3.x
- **System Dependencies**:
  - `espeak` (required by `phonemizer`)
  - C++ Build Tools (required for `panphon`/`epitran` installation on Windows)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the script directly:

```bash
python IPA.py
```

**Example Output:**

```
English: alice -> ['æ', 'lɪ', 's']
Chinese: 愛莉絲 -> ['ai̯', 'li', 'sz̩']
Phonetic distance: 0.75
Alignment:
('æ', 'ai̯') -> 0.67
('lɪ', 'li') -> 0.08
('s', 'sz̩') -> 0.00
```
