"""
approve.py — the human approval gate (Stage 4 trigger).

Finalizing a chapter:
  1. If books/<book>/final/translated_hi/chNN_hi.txt already exists (because you
     edited/corrected it there), it is kept as-is. Otherwise the Stage-1 draft is
     copied into final/ unchanged.
  2. The chapter is marked approved in the manifest.
  3. The approved chapter is ingested into the translation memory so later
     chapters draw on it.

This is deliberately manual: nothing feeds the TM or locks in until you approve.
"""

from __future__ import annotations

import argparse
import shutil
import sys

import common as c
import update_tm


def approve_chapter(book: str, ch: int, reingest: bool = False) -> None:
    bp = c.BookPaths(book)
    bp.ensure_skeleton()

    final_file = bp.final_hi_file(ch)
    draft_file = bp.draft_hi_file(ch)

    if final_file.exists():
        print(f"[approve] using your edited final: {final_file}")
    else:
        if not draft_file.exists():
            raise FileNotFoundError(
                f"No Hindi draft to approve: {draft_file}. Run translate.py first, "
                f"or place an edited file at {final_file}."
            )
        final_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(draft_file, final_file)
        print(f"[approve] copied draft -> {final_file}")

    already = c.is_approved(bp, ch)
    c.set_approved(bp, ch, True)

    if already and not reingest:
        print(f"[note] {c.chapter_id(ch)} was already approved; skipping TM re-ingest "
              f"(use --reingest to add its pairs again).")
        return

    added = update_tm.ingest_chapter(book, ch, require_approved=True)
    print(f"[done] {c.chapter_id(ch)} approved; {added} pair(s) now in translation memory.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Approve a chapter: finalize + feed translation memory.")
    ap.add_argument("--book", required=True)
    ap.add_argument("--chapter", type=int, required=True)
    ap.add_argument("--reingest", action="store_true",
                    help="re-ingest into TM even if already approved before")
    args = ap.parse_args()
    try:
        approve_chapter(args.book, args.chapter, reingest=args.reingest)
    except (FileNotFoundError, RuntimeError, c.APIError) as e:
        sys.exit(f"[error] {e}")


if __name__ == "__main__":
    main()
