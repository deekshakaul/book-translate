"""
run_pipeline.py — orchestrator for Stages 1-3 (+ optional Stage 1.5 polish).

For each chapter, runs each stage FULLY before moving to the next:
    translate (en->hi)  ->  [polish (optional)]  ->  backtranslate (hi->en)  ->  score_qa (paragraph QA)
then moves to the next chapter. Work already marked `done` in the manifest is
skipped, so the pipeline is resumable.

It deliberately does NOT run:
  - Stage 0 (extract_glossary): you review/confirm glossary entries first.
  - Stage 4 (approve/update_tm): you review and approve finals yourself.
"""

from __future__ import annotations

import argparse
import sys

import common as c
import translate
import backtranslate
import score_qa
import polish


def run(book: str, chapters: list[int], force: bool = False, polish_enabled: bool = False) -> None:
    bp = c.BookPaths(book)
    bp.ensure_skeleton()

    if not chapters:
        chapters = bp.list_source_chapters()
    if not chapters:
        sys.exit(f"[error] no chapters found in {bp.source}. Add chNN.txt files first.")

    if not c.confirmed_entries(bp):
        print("[note] glossary has no CONFIRMED entries yet — translating without a glossary. "
              "Consider running extract_glossary.py and confirming entries for consistency.")

    results = []
    for ch in chapters:
        if not bp.source_file(ch).exists():
            print(f"[skip] {c.chapter_id(ch)}: no source file, skipping")
            continue
        print(f"\n=== {c.chapter_id(ch)} ===")
        translate.translate_chapter(book, ch, force=force)
        if polish_enabled:
            polish.polish_chapter(book, ch, force=force)
        backtranslate.backtranslate_chapter(book, ch, force=force)
        report = score_qa.score_chapter(book, ch, force=force)
        results.append((ch, report.get("confidence"), report.get("mean_sim"), report.get("n_flagged")))

    print("\n=== QA summary ===")
    for ch, conf, sim, flagged in results:
        print(f"  {c.chapter_id(ch)}: {conf:<6} mean_sim={sim} flagged={flagged}")
    print("\nReview drafts, then approve with: "
          "python scripts/approve.py --book %s --chapter <N>" % book)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Stages 1-3 (translate -> [polish] -> backtranslate -> QA) per chapter.")
    ap.add_argument("--book", required=True)
    ap.add_argument("--chapters", help="e.g. '1', '1-5', '1,3,5'. Default: all chapters in source/.")
    ap.add_argument("--force", action="store_true", help="re-run stages even if already done")
    ap.add_argument("--polish", action="store_true", help="run grammar polish after translate (optional Stage 1.5)")
    args = ap.parse_args()

    chapters = c.parse_chapter_range(args.chapters) if args.chapters else []
    try:
        run(args.book, chapters, force=args.force, polish_enabled=args.polish)
    except (FileNotFoundError, c.APIError, KeyError) as e:
        sys.exit(f"[error] {e}")


if __name__ == "__main__":
    main()
