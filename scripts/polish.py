"""
polish_grammar.py — Post-translate grammatical cleanup (Hindi text).

Fixes punctuation, spacing, verb agreement, and common grammatical errors
in the Hindi translation draft. Runs after translate.py, before backtranslate.

Uses the "polish" stage from config.yaml (add to config if not present).
Falls back to qa_judge model if "polish" stage not defined.

For Groq (free tier), adds configurable delays between paragraph requests to avoid
hitting rate limits. Configure groq_request_delay_ms in config.yaml.

Run with: python scripts/polish_grammar.py --book <name> --chapter <N> [--force]
          Reads: books/<book>/drafts/translated_hi/chNN_hi.txt
          Writes: books/<book>/drafts/translated_hi/chNN_hi.txt (overwrite)
"""

from __future__ import annotations

import argparse
import sys
import time

import common as c


def polish_chapter(book: str, ch: int, force: bool = False) -> None:
    """Polish Hindi grammar in a chapter draft."""
    bp = c.BookPaths(book)
    bp.ensure_skeleton()

    draft_file = bp.draft_hi_file(ch)
    if not draft_file.exists():
        raise FileNotFoundError(f"Draft Hindi file not found: {draft_file}")

    # Check if already polished (optional; can force repolish).
    if not force and c.get_stage_status(bp, ch, "polish") == "done":
        print(f"[skip] {c.chapter_id(ch)} already polished")
        return

    hindi_text = c.read_text(draft_file)
    paragraphs = c.split_paragraphs(hindi_text)

    print(
        f"[polish] {c.chapter_id(ch)} | {len(paragraphs)} paragraphs"
    )

    # Resolve the "polish" stage config. If not defined, fall back to qa_judge model.
    cfg = c.load_config()
    stages = cfg.get("stages", {})
    if "polish" in stages:
        stage_cfg = c.resolve_stage("polish")
    else:
        # Fallback to qa_judge if polish not defined
        print("[note] 'polish' stage not in config.yaml; using qa_judge model as fallback")
        stage_cfg = c.resolve_stage("qa_judge")

    client = c.OpenAICompatClient(stage_cfg)

    polished_paras = []
    total_tokens = 0

    # Get request delay from config if using cloud provider
    cfg = c.load_config()
    request_delay_ms = cfg.get("groq_request_delay_ms", 100) if stage_cfg.provider.lower() == "groq" else 0
    request_delay_s = request_delay_ms / 1000.0

    for i, para in enumerate(paragraphs):
        # Build a grammar-focused prompt.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert Hindi language editor. Your task is to fix "
                    "grammatical errors, punctuation, spacing, and verb agreement in the "
                    "following Hindi text. Preserve the meaning, tone, and structure exactly. "
                    "Output ONLY the corrected Hindi text, no explanation or commentary."
                ),
            },
            {
                "role": "user",
                "content": f"Fix grammatical errors in this Hindi paragraph:\n\n{para}",
            },
        ]

        try:
            polished_text, usage = client.chat(messages)
            polished_paras.append(polished_text.strip())
            total_tokens += c.total_tokens(usage)
            if (i + 1) % 5 == 0:
                print(f"  [{i + 1}/{len(paragraphs)}] polished")
            # Add delay between requests if using cloud provider
            if request_delay_s > 0 and i < len(paragraphs) - 1:
                time.sleep(request_delay_s)
        except c.APIError as e:
            # If a single paragraph fails, keep the original and log warning.
            print(f"[warn] failed to polish paragraph {i}; keeping original: {e}")
            polished_paras.append(para)

    # Write polished text back to the draft file (overwrite).
    polished_full = c.join_paragraphs(polished_paras)
    c.write_text(draft_file, polished_full)

    # Mark as done in manifest.
    c.set_stage(
        bp, ch, "polish", "done",
        model=stage_cfg.model,
        provider=stage_cfg.provider,
        tokens=total_tokens,
    )
    print(f"[ok] polished {draft_file} ({total_tokens} tokens)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Post-translate grammar cleanup: fix Hindi text punctuation, spacing, verb agreement."
    )
    ap.add_argument("--book", required=True)
    ap.add_argument("--chapter", type=int, required=True)
    ap.add_argument("--force", action="store_true", help="repolish even if already done")
    args = ap.parse_args()
    try:
        polish_chapter(args.book, args.chapter, force=args.force)
    except (FileNotFoundError, c.APIError, KeyError) as e:
        sys.exit(f"[error] {e}")


if __name__ == "__main__":
    main()
