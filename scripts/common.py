"""
common.py — shared library for the book-translate pipeline.

Everything book-agnostic lives here: config loading, the OpenAI-compatible API
client (works for any local or cloud provider given a base_url/api_key/model),
per-book path management + skeleton creation, manifest / glossary / style-rules /
translation-memory I/O, sentence-embeddings, and paragraph utilities.

Stage scripts import from this module; they never talk to a provider or touch the
folder layout directly.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# NOTE: heavier third-party deps (yaml, requests, sentence-transformers, torch)
# are imported lazily inside the functions that need them, so offline utilities
# and the pure text/path/JSON helpers work without the full ML stack installed.

# --------------------------------------------------------------------------- #
# Paths / project root
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
STYLE_RULES_PATH = PROJECT_ROOT / "style_rules.json"
ENV_PATH = PROJECT_ROOT / ".env"

try:
    from dotenv import load_dotenv

    # Explicit path: .env must load regardless of the caller's cwd (running
    # the pipeline from a different directory silently skipped it before).
    if not load_dotenv(ENV_PATH):
        sys.stderr.write(f"[warn] no .env file found at {ENV_PATH} — cloud API keys will be missing\n")
except ImportError:
    sys.stderr.write(
        "[warn] python-dotenv not installed — .env will NOT be loaded, "
        "cloud provider API keys (e.g. GROQ_API_KEY) will be missing. "
        "Fix: pip install python-dotenv\n"
    )

VALID_MODES = {"transliterate", "keep_english", "translate_custom", "translate_default"}
STAGES = ("translate", "backtranslate", "score_qa")  # stages run by run_pipeline


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# JSON helpers (UTF-8, human-readable, Devanagari preserved)
# --------------------------------------------------------------------------- #

def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ${ENV_VAR} references inside config strings."""
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            var = m.group(1)
            got = os.environ.get(var)
            if got is None:
                # Leave the literal in place but warn; local stages don't need keys.
                sys.stderr.write(f"[warn] env var {var} not set (referenced in config)\n")
                return m.group(0)
            return got

        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


_config_cache: dict | None = None


def load_config() -> dict:
    global _config_cache
    if _config_cache is None:
        import yaml

        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"config.yaml not found at {CONFIG_PATH}")
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        _config_cache = _expand_env(raw)
    return _config_cache


@dataclass
class StageConfig:
    stage: str
    provider: str
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int


def resolve_stage(stage: str) -> StageConfig:
    """Resolve a pipeline stage name into a concrete provider endpoint + model."""
    cfg = load_config()
    stages = cfg.get("stages", {})
    if stage not in stages:
        raise KeyError(f"stage '{stage}' not found in config.yaml -> stages")
    scfg = stages[stage]
    provider = scfg["provider"]
    providers = cfg.get("providers", {})
    if provider not in providers:
        raise KeyError(f"provider '{provider}' (used by stage '{stage}') not in config.yaml -> providers")
    pcfg = providers[provider]
    return StageConfig(
        stage=stage,
        provider=provider,
        base_url=pcfg["base_url"].rstrip("/"),
        api_key=pcfg.get("api_key", ""),
        model=scfg["model"],
        temperature=float(scfg.get("temperature", 0.3)),
        max_tokens=int(scfg.get("max_tokens", 4096)),
    )


# --------------------------------------------------------------------------- #
# OpenAI-compatible chat client (one client for local + cloud)
# --------------------------------------------------------------------------- #

class APIError(RuntimeError):
    pass


class OpenAICompatClient:
    """Thin /chat/completions client. Works for Ollama, LM Studio, and any
    OpenAI-compatible cloud endpoint given base_url + api_key + model.

    For Groq (free tier), implements rate limit tracking to avoid 429 errors."""

    def __init__(self, stage_cfg: StageConfig, timeout: int = 600, retries: int = 3):
        self.cfg = stage_cfg
        self.timeout = timeout
        self.retries = retries if stage_cfg.provider.lower() != "groq" else 5

        # Rate limit tracking (Groq: ~30 req/min, ~9000 tokens/min)
        self.last_window_reset = time.time()
        self.requests_this_minute = 0
        self.tokens_this_minute = 0
        self.rate_limit_requests = 30 if stage_cfg.provider.lower() == "groq" else 10000
        self.rate_limit_tokens = 9000 if stage_cfg.provider.lower() == "groq" else 1000000

    def _check_rate_limit(self, estimated_tokens: int = 4096) -> None:
        """Check if request would exceed rate limit; sleep if needed."""
        if self.cfg.provider.lower() != "groq":
            return  # Only enforce for Groq

        now = time.time()
        if now - self.last_window_reset > 60:
            self.last_window_reset = now
            self.requests_this_minute = 0
            self.tokens_this_minute = 0

        # Would this request exceed limits?
        if self.requests_this_minute >= self.rate_limit_requests:
            wait_time = 60 - (now - self.last_window_reset)
            sys.stderr.write(
                f"[rate-limit] Requests/min exceeded ({self.requests_this_minute}/{self.rate_limit_requests}). "
                f"Sleeping {wait_time:.1f}s until window resets...\n"
            )
            time.sleep(max(wait_time + 0.5, 0))
            self.last_window_reset = time.time()
            self.requests_this_minute = 0
            self.tokens_this_minute = 0

        if self.tokens_this_minute + estimated_tokens >= self.rate_limit_tokens:
            wait_time = 60 - (now - self.last_window_reset)
            sys.stderr.write(
                f"[rate-limit] Tokens/min would exceed ({self.tokens_this_minute + estimated_tokens}/{self.rate_limit_tokens}). "
                f"Sleeping {wait_time:.1f}s until window resets...\n"
            )
            time.sleep(max(wait_time + 0.5, 0))
            self.last_window_reset = time.time()
            self.requests_this_minute = 0
            self.tokens_this_minute = 0

    def _update_rate_limit(self, response) -> None:
        """Update rate limit tracking from response headers."""
        if self.cfg.provider.lower() != "groq":
            return

        # Parse Groq's rate limit headers if present
        remaining_req = response.headers.get("x-ratelimit-remaining-requests")
        remaining_tok = response.headers.get("x-ratelimit-remaining-tokens")

        if remaining_req:
            self.requests_this_minute = self.rate_limit_requests - int(remaining_req)
        if remaining_tok:
            self.tokens_this_minute = self.rate_limit_tokens - int(remaining_tok)

    def chat(self, messages: list[dict], **overrides) -> tuple[str, dict]:
        import requests
        import random

        url = f"{self.cfg.base_url}/chat/completions"
        max_tokens = overrides.get("max_tokens", self.cfg.max_tokens)
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": overrides.get("temperature", self.cfg.temperature),
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"

        # Check rate limits before requesting
        self._check_rate_limit(estimated_tokens=max_tokens)

        last_err: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            except requests.exceptions.ConnectionError as e:
                raise APIError(
                    f"Could not reach {self.cfg.provider} at {self.cfg.base_url}. "
                    f"Is the local server (Ollama/LM Studio) running? Original: {e}"
                ) from e
            except requests.exceptions.RequestException as e:
                last_err = e
                # Exponential backoff with jitter for retries
                backoff = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(backoff)
                continue

            # Update rate limit tracking from response headers
            self._update_rate_limit(resp)

            if resp.status_code == 200:
                data = resp.json()
                try:
                    text = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError) as e:
                    raise APIError(f"Unexpected response shape from {self.cfg.provider}: {data}") from e
                if not text or not text.strip():
                    raise APIError(
                        f"{self.cfg.provider}/{self.cfg.model} returned an empty response "
                        f"(finish_reason={data['choices'][0].get('finish_reason')!r}). "
                        f"Full response: {data}"
                    )
                usage = data.get("usage", {}) or {}
                # Track tokens used
                tokens_used = total_tokens(usage)
                self.tokens_this_minute += tokens_used
                self.requests_this_minute += 1
                return text, usage

            # Retry on transient server errors / rate limits.
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                if resp.status_code == 429:
                    # Rate limited: exponential backoff + jitter
                    backoff = (3 ** attempt) + random.uniform(0, 2)
                    sys.stderr.write(f"[rate-limit] HTTP 429; attempt {attempt}/{self.retries}, sleeping {backoff:.1f}s\n")
                else:
                    # Server error: milder backoff
                    backoff = (2 ** attempt) + random.uniform(0, 1)
                last_err = APIError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                time.sleep(backoff)
                continue

            raise APIError(
                f"{self.cfg.provider} returned HTTP {resp.status_code} for model "
                f"'{self.cfg.model}': {resp.text[:500]}"
            )

        raise APIError(f"Request to {self.cfg.provider} failed after {self.retries} attempts: {last_err}")


def total_tokens(usage: dict) -> int:
    if not usage:
        return 0
    return int(usage.get("total_tokens") or 0)


# --------------------------------------------------------------------------- #
# Per-book paths + skeleton
# --------------------------------------------------------------------------- #

class BookPaths:
    """Computes every path for a book and creates the folder skeleton on demand."""

    def __init__(self, book: str):
        cfg = load_config()
        books_dir = PROJECT_ROOT / cfg.get("paths", {}).get("books_dir", "books")
        self.book = book
        self.root = books_dir / book
        self.source = self.root / "source"
        self.glossary = self.root / "glossary.json"
        self.tm_dir = self.root / "translation_memory"
        self.tm_index = self.tm_dir / "tm_index.json"
        self.drafts_hi = self.root / "drafts" / "translated_hi"
        self.drafts_back = self.root / "drafts" / "backtranslated_en"
        self.qa_dir = self.root / "qa_reports"
        self.book_summary = self.qa_dir / "book_summary.json"
        self.final_hi = self.root / "final" / "translated_hi"
        self.manifest = self.root / "manifest.json"

    # ----- per-chapter file helpers -----
    def source_file(self, ch: int) -> Path:
        return self.source / f"{chapter_id(ch)}.txt"

    def draft_hi_file(self, ch: int) -> Path:
        return self.drafts_hi / f"{chapter_id(ch)}_hi.txt"

    def draft_back_file(self, ch: int) -> Path:
        return self.drafts_back / f"{chapter_id(ch)}_back_en.txt"

    def qa_file(self, ch: int) -> Path:
        return self.qa_dir / f"{chapter_id(ch)}_qa.json"

    def final_hi_file(self, ch: int) -> Path:
        return self.final_hi / f"{chapter_id(ch)}_hi.txt"

    def ensure_skeleton(self) -> None:
        """Create dirs + empty glossary/tm/manifest for a book on first run."""
        if not self.source.exists():
            self.source.mkdir(parents=True, exist_ok=True)
        for d in (self.tm_dir, self.drafts_hi, self.drafts_back, self.qa_dir, self.final_hi):
            d.mkdir(parents=True, exist_ok=True)
        if not self.glossary.exists():
            write_json(self.glossary, {"confirmed": [], "pending": []})
        if not self.tm_index.exists():
            write_json(self.tm_index, {"pairs": []})
        if not self.manifest.exists():
            write_json(self.manifest, {"book": self.book, "chapters": {}})

    def list_source_chapters(self) -> list[int]:
        """Chapter numbers present in source/, sorted ascending."""
        nums = []
        if self.source.exists():
            for p in self.source.glob("ch*.txt"):
                m = re.match(r"ch(\d+)", p.stem)
                if m:
                    nums.append(int(m.group(1)))
        return sorted(set(nums))


# --------------------------------------------------------------------------- #
# Chapter id / ranges
# --------------------------------------------------------------------------- #

def chapter_id(n: int) -> str:
    return f"ch{int(n):02d}"


def parse_chapter_range(spec: str) -> list[int]:
    """'1' -> [1]; '1-5' -> [1,2,3,4,5]; '1,3,5' -> [1,3,5]."""
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


# --------------------------------------------------------------------------- #
# Paragraph utilities
# --------------------------------------------------------------------------- #

def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; collapse internal whitespace/newlines per paragraph."""
    blocks = re.split(r"\n\s*\n", text.strip())
    paras = []
    for b in blocks:
        cleaned = re.sub(r"[ \t]*\n[ \t]*", " ", b.strip())
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        if cleaned:
            paras.append(cleaned)
    return paras


def join_paragraphs(paras: Iterable[str]) -> str:
    return "\n\n".join(p.strip() for p in paras if p.strip()) + "\n"


# --------------------------------------------------------------------------- #
# Manifest I/O
# --------------------------------------------------------------------------- #

def load_manifest(bp: BookPaths) -> dict:
    return read_json(bp.manifest, {"book": bp.book, "chapters": {}})


def save_manifest(bp: BookPaths, manifest: dict) -> None:
    write_json(bp.manifest, manifest)


def get_stage_status(bp: BookPaths, ch: int, stage: str) -> str | None:
    manifest = load_manifest(bp)
    return (
        manifest.get("chapters", {})
        .get(chapter_id(ch), {})
        .get(stage, {})
        .get("status")
    )


def set_stage(bp: BookPaths, ch: int, stage: str, status: str, **meta) -> None:
    manifest = load_manifest(bp)
    chapters = manifest.setdefault("chapters", {})
    cid = chapter_id(ch)
    entry = chapters.setdefault(cid, {})
    record = {"status": status, "ts": utc_now()}
    record.update(meta)
    entry[stage] = record
    save_manifest(bp, manifest)


def set_approved(bp: BookPaths, ch: int, value: bool = True) -> None:
    manifest = load_manifest(bp)
    entry = manifest.setdefault("chapters", {}).setdefault(chapter_id(ch), {})
    entry["approved"] = value
    save_manifest(bp, manifest)


def is_approved(bp: BookPaths, ch: int) -> bool:
    manifest = load_manifest(bp)
    return bool(manifest.get("chapters", {}).get(chapter_id(ch), {}).get("approved"))


# --------------------------------------------------------------------------- #
# Style rules
# --------------------------------------------------------------------------- #

def load_style_rules() -> dict:
    return read_json(STYLE_RULES_PATH, {}) or {}


def render_style_block(rules: dict | None = None) -> str:
    rules = rules if rules is not None else load_style_rules()
    if not rules:
        return ""
    lines = ["GLOBAL STYLE RULES (apply to every translation):"]
    if rules.get("register"):
        lines.append(f"- Register/tone: {rules['register']}")
    for r in rules.get("general_rules", []):
        lines.append(f"- {r}")
    honor = rules.get("honorifics", {})
    if honor:
        lines.append("- Honorifics/titles:")
        for k, v in honor.items():
            lines.append(f"    * {k} -> {v}")
    keep = rules.get("keep_english", [])
    if keep:
        note = rules.get("keep_english_note", "")
        lines.append(f"- Keep these words in English (Latin script) inside Hindi: {', '.join(keep)}. {note}".rstrip())
    for r in rules.get("transliteration_conventions", []):
        lines.append(f"- {r}")
    if rules.get("numbers_and_dates"):
        lines.append(f"- Numbers/dates: {rules['numbers_and_dates']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Glossary I/O
# --------------------------------------------------------------------------- #

def load_glossary(bp: BookPaths) -> dict:
    return read_json(bp.glossary, {"confirmed": [], "pending": []}) or {"confirmed": [], "pending": []}


def confirmed_entries(bp: BookPaths) -> list[dict]:
    return load_glossary(bp).get("confirmed", [])


def _existing_terms(gloss: dict) -> set[str]:
    terms = set()
    for bucket in ("confirmed", "pending"):
        for e in gloss.get(bucket, []):
            if e.get("term"):
                terms.add(e["term"].strip().lower())
    return terms


def add_pending(bp: BookPaths, entries: list[dict]) -> int:
    """Append new candidate entries to glossary 'pending', deduped by term.
    Returns the number of newly added entries."""
    gloss = load_glossary(bp)
    existing = _existing_terms(gloss)
    added = 0
    for e in entries:
        term = (e.get("term") or "").strip()
        if not term or term.lower() in existing:
            continue
        entry = {
            "term": term,
            "category": e.get("category", "other"),
            "mode": e.get("mode") if e.get("mode") in VALID_MODES else "transliterate",
            "hindi_form": e.get("hindi_form", ""),
            "notes": e.get("notes", ""),
        }
        gloss.setdefault("pending", []).append(entry)
        existing.add(term.lower())
        added += 1
    if added:
        write_json(bp.glossary, gloss)
    return added


def render_glossary_block(entries: list[dict]) -> str:
    """Turn confirmed glossary entries into explicit rendering instructions,
    grouped by mode so the model gets unambiguous directives."""
    if not entries:
        return ""
    by_mode: dict[str, list[dict]] = {}
    for e in entries:
        by_mode.setdefault(e.get("mode", "translate_default"), []).append(e)

    lines = ["BOOK GLOSSARY (render these EXACTLY the same way every time they appear):"]

    for e in by_mode.get("transliterate", []):
        hf = f" -> {e['hindi_form']}" if e.get("hindi_form") else ""
        note = f"  ({e['notes']})" if e.get("notes") else ""
        lines.append(f"- Transliterate (do NOT translate meaning): \"{e['term']}\"{hf}{note}")

    for e in by_mode.get("translate_custom", []):
        hf = f" -> {e['hindi_form']}" if e.get("hindi_form") else ""
        note = f"  ({e['notes']})" if e.get("notes") else ""
        lines.append(f"- Always translate as this fixed form: \"{e['term']}\"{hf}{note}")

    for e in by_mode.get("keep_english", []):
        note = f"  ({e['notes']})" if e.get("notes") else ""
        lines.append(f"- Keep in English (Latin script) inside the Hindi sentence: \"{e['term']}\"{note}")

    # translate_default entries are tracking-only; no directive injected.
    return "\n".join(lines) if len(lines) > 1 else ""


# --------------------------------------------------------------------------- #
# Embeddings (lazy, GPU with CPU fallback)
# --------------------------------------------------------------------------- #

_embedders = {}  # Cache for different embedding models


def get_embedder(model_type: str = "tm"):
    """Lazily construct and cache the SentenceTransformer model.

    Args:
        model_type: "tm" (translation memory) or "qa" (QA measurement)
    """
    global _embedders
    if model_type not in _embedders:
        from sentence_transformers import SentenceTransformer

        cfg = load_config().get("embeddings", {})

        # Map model_type to config key
        if model_type == "tm":
            model_name = cfg.get("tm_model", "sentence-transformers/LaBSE")
        elif model_type == "qa":
            model_name = cfg.get("qa_model", "backtranslate")
            # qa_model="backtranslate" is handled separately in score_qa.py
            if model_name == "backtranslate":
                return None  # Signal that backtranslation QA is being used
            model_name = model_name if model_name.startswith("sentence-transformers/") else f"sentence-transformers/{model_name}"
        else:
            model_name = cfg.get("model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        device = cfg.get("device", "cuda")
        if device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    sys.stderr.write("[warn] CUDA not available; embeddings running on CPU.\n")
                    device = "cpu"
            except Exception:
                device = "cpu"

        sys.stderr.write(f"[embedding] Loading {model_type} model: {model_name}\n")
        _embedders[model_type] = SentenceTransformer(model_name, device=device)
    return _embedders[model_type]


def embed(texts: list[str], model_type: str = "tm"):
    """Return a (n, d) numpy array of L2-normalized embeddings.

    Args:
        texts: List of strings to embed
        model_type: "tm" (translation memory) or "qa" (QA measurement)
    """
    model = get_embedder(model_type)
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def cosine(a, b) -> float:
    """Cosine similarity for two 1-D vectors (assumed already normalized)."""
    import numpy as np

    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


# --------------------------------------------------------------------------- #
# Translation memory I/O
# --------------------------------------------------------------------------- #

def load_tm(bp: BookPaths) -> dict:
    return read_json(bp.tm_index, {"pairs": []}) or {"pairs": []}


def tm_query(bp: BookPaths, paragraph: str, top_k: int) -> list[dict]:
    """Return up to top_k stored pairs most similar to `paragraph`
    (by source-side embedding cosine). Empty list if TM is empty."""
    tm = load_tm(bp)
    pairs = tm.get("pairs", [])
    if not pairs:
        return []
    query_emb = embed([paragraph])[0]
    scored = []
    for pair in pairs:
        emb = pair.get("embedding")
        if not emb:
            continue
        scored.append((cosine(query_emb, emb), pair))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]


def tm_add(bp: BookPaths, source_paras: list[str], target_paras: list[str]) -> int:
    """Embed source paragraphs and append aligned (source, target) pairs to TM.
    Returns number of pairs added. Aligns by index up to the shorter length."""
    n = min(len(source_paras), len(target_paras))
    if n == 0:
        return 0
    src = source_paras[:n]
    tgt = target_paras[:n]
    embs = embed(src)
    tm = load_tm(bp)
    pairs = tm.setdefault("pairs", [])
    for i in range(n):
        pairs.append(
            {
                "source": src[i],
                "target": tgt[i],
                "embedding": [round(float(x), 6) for x in embs[i].tolist()],
            }
        )
    write_json(bp.tm_index, tm)
    return n


# --------------------------------------------------------------------------- #
# Prompt helpers shared by stage scripts
# --------------------------------------------------------------------------- #

def build_translation_messages(
    chapter_text: str,
    style_block: str,
    glossary_block: str,
    tm_examples: list[dict],
    continuity_para: str | None,
) -> list[dict]:
    """Assemble the en->hi translation prompt (translate.py)."""
    system = (
        "You are an expert literary translator translating English fiction into "
        "natural, idiomatic Hindi (Devanagari). Your priority is to preserve tone, "
        "emotion, sarcasm, wit, humor, irony, cultural nuances at all costs and register — not word-for-word literalness. "
        "\n"
        "CRITICAL — GRAMMATICAL GENDER IN HINDI:\n"
        "Hindi nouns have gender (masculine/feminine) affecting adjectives, verbs, and articles:\n"
        "- Masculine nouns: -आ/-ा endings; adjectives -आ/-ा; past tense -आ/-ा\n"
        "- Feminine nouns: -ी/-इ endings; adjectives -ी/-इ; past tense -ी/-इ\n"
        "Ensure ALL adjectives, verbs, articles, and pronouns agree with noun gender throughout.\n"
        "\n"
        "Gender agreement examples:\n"
        "- 'एक सुंदर लड़की' (a beautiful GIRL - feminine)\n"
        "- 'एक सुंदर लड़का' (a beautiful BOY - masculine)\n"
        "- 'लड़की सुंदर थी' (GIRL was beautiful - feminine past tense)\n"
        "- 'लड़का सुंदर था' (BOY was beautiful - masculine past tense)\n"
        "- 'वह सुंदर है' (SHE is beautiful - feminine subject)\n"
        "- 'वह सुंदर है' (HE is beautiful - masculine subject)\n"
        "\n"
        "Output ONLY the Hindi translation, preserving paragraph breaks 1:1 with the "
        "source. Do not add notes, headers, or commentary."
    )
    parts: list[str] = []
    if style_block:
        parts.append(style_block)
    if glossary_block:
        parts.append(glossary_block)
    if tm_examples:
        ex_lines = ["APPROVED TRANSLATION EXAMPLES from earlier in this book "
                    "(match this voice and terminology):"]
        for ex in tm_examples:
            ex_lines.append(f"EN: {ex.get('source', '').strip()}")
            ex_lines.append(f"HI: {ex.get('target', '').strip()}")
            ex_lines.append("")
        parts.append("\n".join(ex_lines).rstrip())
    if continuity_para:
        parts.append(
            "For continuity, the final paragraph of the PREVIOUS chapter's approved "
            "Hindi translation was:\n" + continuity_para.strip()
        )
    parts.append(
        "Translate the following chapter into Hindi. Preserve each paragraph as a "
        "separate paragraph in the same order:\n\n" + chapter_text.strip()
    )
    user = "\n\n".join(parts)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_backtranslation_messages(hindi_text: str) -> list[dict]:
    """Assemble the hi->en back-translation prompt (backtranslate.py).
    Deliberately spare — an independent, faithful rendering back to English so QA
    can detect meaning/tone drift."""
    system = (
        "You are an expert translator translating Hindi fiction into faithful, "
        "natural English. Render the meaning and tone as accurately as possible so "
        "the result can be compared against an original English text. Output ONLY the "
        "English translation, preserving paragraph breaks 1:1. No notes or commentary."
    )
    user = (
        "Translate the following Hindi text into English, preserving each paragraph "
        "as a separate paragraph in the same order:\n\n" + hindi_text.strip()
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
