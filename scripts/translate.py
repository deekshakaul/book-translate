"""
Stage 1 — translate.py (English -> Hindi, one chapter at a time).

Builds a literary-translation prompt from: global style rules + this book's
confirmed glossary + top-k translation-memory examples (retrieved by paragraph
similarity) + an optional continuity paragraph from the previous chapter's
approved Hindi. Saves the draft and records status/tokens in the manifest.
"""

from __future__ import annotations

import argparse
import sys

import common as c


def translate_chapter(book: str, ch: int, force: bool = False) -> None:
    bp = c.BookPaths(book)
    bp.ensure_skeleton()

    src_file = bp.source_file(ch)
    if not src_file.exists():
        raise FileNotFoundError(f"Source chapter not found: {src_file}")

    out_file = bp.draft_hi_file(ch)
    if out_file.exists() and not force and c.get_stage_status(bp, ch, "translate") == "done":
        print(f"[skip] {c.chapter_id(ch)} already translated -> {out_file}")
        return

    chapter_text = c.read_text(src_file)
    paragraphs = c.split_paragraphs(chapter_text)

    # Style + glossary blocks.
    style_block = c.render_style_block()
    glossary_block = c.render_glossary_block(c.confirmed_entries(bp))

    # Translation-memory few-shots: gather nearest neighbours across the chapter's
    # paragraphs, dedup, cap at tm_top_k overall.
    cfg = c.load_config()
    top_k = int(cfg.get("embeddings", {}).get("tm_top_k", 3))
    tm_examples: list[dict] = []
    if c.load_tm(bp).get("pairs"):
        seen = set()
        for para in paragraphs:
            for ex in c.tm_query(bp, para, top_k):
                key = ex.get("source", "")
                if key and key not in seen:
                    seen.add(key)
                    tm_examples.append(ex)
        tm_examples = tm_examples[:top_k]

    # Continuity: last paragraph of previous chapter's approved Hindi, if any.
    continuity_para = None
    prev_final = bp.final_hi_file(ch - 1)
    if ch > 1 and prev_final.exists():
        prev_paras = c.split_paragraphs(c.read_text(prev_final))
        if prev_paras:
            continuity_para = prev_paras[-1]

    messages = c.build_translation_messages(
        chapter_text=chapter_text,
        style_block=style_block,
        glossary_block=glossary_block,
        tm_examples=tm_examples,
        continuity_para=continuity_para,
    )

    stage_cfg = c.resolve_stage("translate")
    print(
        f"[translate] {c.chapter_id(ch)} via {stage_cfg.provider}/{stage_cfg.model} "
        f"| {len(paragraphs)} paras | {len(tm_examples)} TM examples"
        + (" | +continuity" if continuity_para else "")
    )
    client = c.OpenAICompatClient(stage_cfg)
    text, usage = client.chat(messages)

    c.write_text(out_file, text.strip() + "\n")
    c.set_stage(
        bp, ch, "translate", "done",
        model=stage_cfg.model, provider=stage_cfg.provider,
        tokens=c.total_tokens(usage), tm_examples=len(tm_examples),
    )
    print(f"[ok] wrote {out_file}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1: translate a chapter English -> Hindi.")
    ap.add_argument("--book", required=True)
    ap.add_argument("--chapter", type=int, required=True)
    ap.add_argument("--force", action="store_true", help="re-translate even if a draft exists")
    args = ap.parse_args()
    try:
        translate_chapter(args.book, args.chapter, force=args.force)
    except (FileNotFoundError, c.APIError, KeyError) as e:
        sys.exit(f"[error] {e}")


if __name__ == "__main__":
    main()
