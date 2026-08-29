"""
Stage 0 — extract_glossary.py.

Scans a chapter and PROPOSES candidate named-entity glossary entries (people,
places, recurring terms) with a suggested category, render `mode`, and Hindi
form. Proposals go into glossary.json's `pending` list — they are NEVER
auto-confirmed. You review them and move approved entries to `confirmed`
yourself; only `confirmed` entries are injected into translation prompts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import common as c


EXTRACT_SYSTEM = (
    "You extract proper nouns and recurring named entities from English literary text "
    "to build a translation glossary for an English->Hindi book translation. "
    "Return ONLY a JSON array. Each element: "
    '{"term": "<exact surface form>", "category": "name"|"place"|"title"|"term", '
    '"mode": "transliterate"|"keep_english"|"translate_custom"|"translate_default", '
    '"hindi_form": "<suggested Devanagari, or empty>", "notes": "<short, or empty>"}. '
    "Guidance: people and places -> transliterate; honorific+name (e.g. 'Mr. Bennet') -> "
    "transliterate keeping the honorific; common English address words that read naturally "
    "in Hinglish -> keep_english. Only include genuinely recurring/proper terms, not common "
    "nouns. No prose, no markdown fences — just the JSON array."
)


def _extract_json_array(text: str) -> list[dict]:
    text = text.strip()
    # Strip ```json fences if present.
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    sys.stderr.write("[warn] could not parse glossary candidates from model reply\n")
    return []


def extract_chapter(book: str, ch: int) -> int:
    bp = c.BookPaths(book)
    bp.ensure_skeleton()

    src_file = bp.source_file(ch)
    if not src_file.exists():
        raise FileNotFoundError(f"Source chapter not found: {src_file}")

    chapter_text = c.read_text(src_file)
    stage_cfg = c.resolve_stage("extract")
    print(f"[extract] {c.chapter_id(ch)} via {stage_cfg.provider}/{stage_cfg.model}")
    client = c.OpenAICompatClient(stage_cfg)
    text, _ = client.chat([
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": chapter_text},
    ])
    candidates = _extract_json_array(text)
    added = c.add_pending(bp, candidates)
    print(f"[ok] {c.chapter_id(ch)}: {len(candidates)} candidate(s) proposed, {added} new -> pending")
    return added


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 0: propose glossary entries (into 'pending').")
    ap.add_argument("--book", required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--chapter", type=int)
    group.add_argument("--all", action="store_true", help="scan every chapter in source/")
    args = ap.parse_args()

    try:
        bp = c.BookPaths(args.book)
        bp.ensure_skeleton()
        if args.all:
            chapters = bp.list_source_chapters()
            if not chapters:
                sys.exit(f"[error] no chapters found in {bp.source}")
            total = 0
            for ch in chapters:
                total += extract_chapter(args.book, ch)
            print(f"[done] {total} new candidate(s) across {len(chapters)} chapter(s).")
        else:
            extract_chapter(args.book, args.chapter)
        print("[review] Edit glossary.json: move approved entries from 'pending' to 'confirmed'. "
              "Only 'confirmed' entries are used in translation.")
    except (FileNotFoundError, c.APIError, KeyError) as e:
        sys.exit(f"[error] {e}")


if __name__ == "__main__":
    main()
