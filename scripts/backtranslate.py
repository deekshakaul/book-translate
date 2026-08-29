"""
Stage 2 — backtranslate.py (Hindi -> English, one chapter at a time).

Takes the Hindi draft from Stage 1 and translates it back to English using a
SEPARATELY configured model (default: a different model than `translate`), so the
back-translation is an independent check. Stage 3 (score_qa) then compares this
against the original English to detect meaning/tone drift.
"""

from __future__ import annotations

import argparse
import sys

import common as c


def backtranslate_chapter(book: str, ch: int, force: bool = False) -> None:
    bp = c.BookPaths(book)
    bp.ensure_skeleton()

    hi_file = bp.draft_hi_file(ch)
    if not hi_file.exists():
        raise FileNotFoundError(
            f"Hindi draft not found: {hi_file}. Run translate.py for this chapter first."
        )

    out_file = bp.draft_back_file(ch)
    if out_file.exists() and not force and c.get_stage_status(bp, ch, "backtranslate") == "done":
        print(f"[skip] {c.chapter_id(ch)} already back-translated -> {out_file}")
        return

    hindi_text = c.read_text(hi_file)
    messages = c.build_backtranslation_messages(hindi_text)

    stage_cfg = c.resolve_stage("backtranslate")
    print(f"[backtranslate] {c.chapter_id(ch)} via {stage_cfg.provider}/{stage_cfg.model}")
    client = c.OpenAICompatClient(stage_cfg)
    text, usage = client.chat(messages)

    c.write_text(out_file, text.strip() + "\n")
    c.set_stage(
        bp, ch, "backtranslate", "done",
        model=stage_cfg.model, provider=stage_cfg.provider,
        tokens=c.total_tokens(usage),
    )
    print(f"[ok] wrote {out_file}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2: back-translate a chapter Hindi -> English.")
    ap.add_argument("--book", required=True)
    ap.add_argument("--chapter", type=int, required=True)
    ap.add_argument("--force", action="store_true", help="re-run even if output exists")
    args = ap.parse_args()
    try:
        backtranslate_chapter(args.book, args.chapter, force=args.force)
    except (FileNotFoundError, c.APIError, KeyError) as e:
        sys.exit(f"[error] {e}")


if __name__ == "__main__":
    main()
