# book-translate

A personal, repeatable pipeline for translating full-length books between
**English and Hindi**, chapter by chapter, on **local GPU models** — with a
built-in back-translation QA step so meaning and tone survive translation, not
just individual words.

The code is **book-agnostic**: every script takes `--book <name>` and works
purely off the folder layout. Adding a new book = drop chapter `.txt` files into
`books/<name>/source/` and run the same commands. Model choice for each stage is
a **config change** (`config.yaml`), never a code change.

## What it does

| Stage | Script | What it does | Automated? |
|------:|--------|--------------|------------|
| 0 | `extract_glossary.py` | Proposes candidate named-entity glossary entries into `pending` | You approve → `confirmed` |
| 1 | `translate.py` | English → Hindi, using style rules + confirmed glossary + translation-memory examples | auto |
| 2 | `backtranslate.py` | Hindi → English with a **different** model (independent check) | auto |
| 3 | `score_qa.py` | Compares source vs back-translation **paragraph-by-paragraph** (cosine similarity); a judge model rules on flagged paragraphs; emits a confidence band | auto |
| 4 | `approve.py` | You approve a chapter → copies draft to `final/`, ingests it into translation memory so later chapters improve | You trigger |

`run_pipeline.py` chains stages **1–3** for you and skips work already done
(resumable). Stages 0 and 4 are deliberate human gates.

## Setup

Requires **Python 3.10–3.13** (the pinned torch/numpy have no wheels for 3.14
yet) and an NVIDIA GPU (RTX 2070 Super / Turing works out of the box).

> **Windows note:** use your Python-launcher interpreter, e.g. `py -3.12`, to
> build the venv. Do **not** use the MSYS2/MinGW `python` that Git Bash may
> default to (often 3.14) — the ML wheels won't install there. Check with
> `py -0p`.

```powershell
cd book-translate
# if running for the first start - create new venv
py -3.12 -m venv .venv
# activate venv -  this can be done on subsequent runs
.\.venv\Scripts\Activate.ps1        # PowerShell
# (bash/git-bash: source .venv/Scripts/activate)

pip install -r requirements.txt
```

`requirements.txt` pins the **CUDA 12.1** build of PyTorch. If your NVIDIA driver
is older, install torch first from the cu118 index, then the rest:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Verify the GPU is visible:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Local model server

Run **one** of these and pull the models named in `config.yaml`:

- **Ollama** (`http://localhost:11434`):
  ```bash
  ollama pull mistral-nemo:12b
  ollama pull llama3.1:8b
  ollama pull aya-expanse:8b
  ```
- **LM Studio** — start its local server (`http://localhost:1234`) and set each
  stage's `provider: lmstudio` in `config.yaml`.

Swap any stage's `model`/`provider` in `config.yaml` as you test quality — no
code changes needed.

## Global vs per-book data

- `style_rules.json` (repo root, **global**) — loanword/honorific/register
  conventions applied to every book (e.g. keep "Dear" in English, transliterate
  "Mr." rather than translate to श्री). Hand-curated by you.
- `books/<book>/glossary.json` (**per book**) — proper nouns / recurring terms
  for that book. `pending` (proposed) vs `confirmed` (only these are injected).
- `books/<book>/translation_memory/tm_index.json` (**per book**) — approved
  paragraph pairs used as retrieval examples. Not shared across books.

### Glossary entry `mode` values

| mode | meaning |
|------|---------|
| `transliterate` | render phonetically in Devanagari, don't translate meaning (names, places) |
| `keep_english` | leave the term in Latin script inside the Hindi sentence (Hinglish) |
| `translate_custom` | force one fixed Hindi word/phrase for this term everywhere |
| `translate_default` | tracking only — model translates normally |

## Typical workflow for a book

```bash
# 1. Put chapter files in books/<book>/source/  (ch01.txt, ch02.txt, ...)

# 2. Propose glossary entries, then hand-edit glossary.json: move good ones
#    from "pending" to "confirmed".
python scripts/extract_glossary.py --book pride_and_prejudice --chapter 1

# 3. Translate + back-translate + QA a chapter (or a range).
python scripts/run_pipeline.py --book pride_and_prejudice --chapters 1
#    Outputs: drafts/translated_hi/ch01_hi.txt
#             drafts/backtranslated_en/ch01_back_en.txt
#             qa_reports/ch01_qa.json  (+ book_summary.json)

# 4. Review the Hindi draft. Optionally edit books/<book>/final/translated_hi/ch01_hi.txt
#    directly, then approve — this locks it in and feeds the translation memory
#    so later chapters get more consistent.
python scripts/approve.py --book pride_and_prejudice --chapter 1
```

Chapter files are expected to be named `chNN.txt` (zero-padded, e.g. `ch01.txt`).
Paragraphs are separated by blank lines and preserved 1:1 through the pipeline.

## Individual commands

```bash
python scripts/extract_glossary.py --book <book> [--chapter N | --all]
python scripts/translate.py        --book <book> --chapter N
python scripts/backtranslate.py    --book <book> --chapter N
python scripts/score_qa.py         --book <book> --chapter N
python scripts/update_tm.py        --book <book> --chapter N   # (approve.py calls this for you)
python scripts/approve.py          --book <book> --chapter N
python scripts/run_pipeline.py     --book <book> [--chapters 1-5]
```

## Notes

- Built for **local** inference now. Cloud providers (Groq, Cerebras, Mistral,
  OpenRouter) are pre-wired as commented presets in `config.yaml` and use the
  same OpenAI-compatible client — switching to them is config-only.
- The first run downloads the sentence-transformers embedding model (~470 MB).
