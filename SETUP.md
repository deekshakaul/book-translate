# Setup Guide - Pride & Prejudice Translation Pipeline

## Prerequisites

- Python 3.9+ installed
- 16GB+ RAM (8GB minimum for CPU-only)
- Optional: NVIDIA GPU with CUDA 11.8+ (for faster inference)
- Optional: Ollama (for running local LLM models)

## Quick Start (Automated - Windows)

```powershell
cd path/to/book-translate
.\setup.ps1
```

This detects your GPU/CUDA version, creates the venv, installs everything,
and creates `.env` from the template. Then just fill in your `GROQ_API_KEY`
in `.env` and you're ready to run the pipeline (see below).

If you'd rather do it manually (or you're on Mac/Linux), follow the steps below.

## Quick Start (Manual)

### 1. Clone/Download Project
```bash
cd path/to/book-translate
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. GPU Setup (Optional, NVIDIA only)
```bash
# Check your CUDA version
nvidia-smi

# If CUDA 12.1 (default in requirements.txt): you're done!
# If CUDA 11.8:
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# If CUDA 12.4:
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# If CPU only (no GPU):
pip install torch==2.5.1
```

### 5. Configure API keys and models
```bash
# Copy the template and fill in your values
cp .env.example .env    # Mac/Linux
copy .env.example .env  # Windows
```

Edit `.env` and set:
- `GROQ_API_KEY` — get one free at https://console.groq.com/keys
- `GROQ_LLAMA_70B` / `GROQ_MIXTRAL_70B` — Groq rotates model availability
  over time; check https://console.groq.com/docs/models for current names
  if you hit a "model decommissioned" (HTTP 404) error
- `OLLAMA_LLAMA_8B` / `OLLAMA_NEMO` — must match tags you've pulled with `ollama pull`

### 6. Install Ollama (Optional, for local models)
Download from: https://ollama.ai

Then pull models matching what's in your `.env`:
```bash
ollama pull llama3.1:8b
ollama pull mistral-nemo:12b
```

## Running the Pipeline

### Terminal 1: Start Ollama (if using local models)
```bash
ollama serve
```

### Terminal 2: Run Translation Pipeline
```bash
# Activate venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# Run chapter 3
python scripts/run_pipeline.py --book "Pride and Prejudice - Jane Austen" --chapters 3 --polish

# Run multiple chapters
python scripts/run_pipeline.py --book "Pride and Prejudice - Jane Austen" --chapters 1 2 3 --polish

# Force re-run (ignore cached results)
python scripts/run_pipeline.py --book "Pride and Prejudice - Jane Austen" --chapters 3 --polish --force
```

## Configuration

Edit `config.yaml` to:
- Change models (translate provider, backtranslate model, etc.)
- Set device to `cpu` or `cuda`
- Adjust rate limiting, thresholds, etc.

## Troubleshooting

### "No module named 'yaml'"
```bash
pip install pyyaml
```

### "No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### CUDA not detected (but you have NVIDIA GPU)
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

### Ollama connection error
Make sure Ollama is running:
```bash
ollama serve  # in separate terminal
```

### Translation runs but output files are empty
This usually means `.env` never got loaded, so `GROQ_API_KEY` was `None` and
the API call silently failed. `python-dotenv` loads `.env` — if it's missing,
`common.py` swallows the import error instead of crashing. Fix:
```bash
pip install python-dotenv
```
Then verify the key is actually visible to Python:
```bash
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(bool(os.getenv('GROQ_API_KEY')))"
```
Should print `True`.

### "Groq returned HTTP 404 for model ..."
The model name in `.env` no longer exists — Groq periodically decommissions
models. Check current available models at https://console.groq.com/docs/models
and update `GROQ_LLAMA_70B` / `GROQ_MIXTRAL_70B` in `.env` accordingly.

## File Structure

```
book-translate/
├── scripts/
│   ├── run_pipeline.py      # Main entry point
│   ├── translate.py         # Translation stage
│   ├── polish.py            # Grammar polish stage
│   ├── score_qa.py          # QA scoring stage
│   ├── extract_glossary.py  # Glossary extraction (Stage 0)
│   ├── update_tm.py         # Translation Memory management
│   └── common.py            # Shared utilities
├── config.yaml              # Configuration (models, providers, thresholds)
├── requirements.txt         # Python dependencies
├── setup.ps1                # Automated setup (Windows)
├── .env.example             # Template for API keys / model names
├── books/                   # Book data (created on first run)
└── README.md                # Project documentation
```

## Next Steps

1. Extract glossary: `python scripts/extract_glossary.py --book "Pride and Prejudice - Jane Austen" --chapter 1`
2. Review and approve glossary entries in `books/Pride and Prejudice - Jane Austen/glossary.json`
3. Run pipeline: `python scripts/run_pipeline.py --book "Pride and Prejudice - Jane Austen" --chapters 1 --polish`
4. Review output in `books/Pride and Prejudice - Jane Austen/ch01_final.txt`
5. Approve translations to build Translation Memory for better future results

## Performance Expectations

### On CPU (MacBook Air, typical laptop)
- ~30-40 minutes per chapter
- All models run locally

### On NVIDIA GPU (RTX 3060+)
- ~2-5 minutes per chapter
- Faster inference for backtranslation and QA

### With Groq Cloud (translate only)
- ~5-10 seconds per chapter (translate)
- + Local model time for other stages

## Support

See README.md for architecture details and contribution guidelines.
