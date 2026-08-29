"""
Stage 4 (library + CLI) — update_tm.py.

Ingests an APPROVED chapter into the book's translation memory: aligns the
original English (source) with the approved Hindi (final/translated_hi) paragraph
by paragraph, embeds the source side, and appends the pairs to tm_index.json.
These pairs are then retrieved as few-shot examples for LATER chapters, so
terminology and voice get more consistent as the book progresses.

Normally invoked by approve.py, but can be run directly.
"""

from __future__ import annotations

import argparse
import sys

import common as c


def ingest_chapter(book: str, ch: int, require_approved: bool = True) -> int:
    bp = c.BookPaths(book)
    bp.ensure_skeleton()

    if require_approved and not c.is_approved(bp, ch):
        raise RuntimeError(
            f"{c.chapter_id(ch)} is not approved. Run approve.py first "
            f"(or pass --no-require-approved to force)."
        )

    src_file = bp.source_file(ch)
    final_file = bp.final_hi_file(ch)
    if not src_file.exists():
        raise FileNotFoundError(f"Source not found: {src_file}")
    if not final_file.exists():
        raise FileNotFoundError(
            f"Approved Hindi not found: {final_file}. Approve the chapter first."
        )

    src_paras = c.split_paragraphs(c.read_text(src_file))
    tgt_paras = c.split_paragraphs(c.read_text(final_file))
    if len(src_paras) != len(tgt_paras):
        print(f"[warn] paragraph count mismatch (source={len(src_paras)} "
              f"final={len(tgt_paras)}); aligning first {min(len(src_paras), len(tgt_paras))}")

    added = c.tm_add(bp, src_paras, tgt_paras)
    print(f"[ok] added {added} pair(s) from {c.chapter_id(ch)} to translation memory "
          f"({bp.tm_index})")
    return added


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 4: ingest an approved chapter into translation memory.")
    ap.add_argument("--book", required=True)
    ap.add_argument("--chapter", type=int, required=True)
    ap.add_argument("--no-require-approved", action="store_true",
                    help="ingest even if the chapter is not marked approved")
    args = ap.parse_args()
    try:
        ingest_chapter(args.book, args.chapter, require_approved=not args.no_require_approved)
    except (FileNotFoundError, RuntimeError, c.APIError) as e:
        sys.exit(f"[error] {e}")


if __name__ == "__main__":
    main()
