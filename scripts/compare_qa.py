"""
compare_qa.py — run BOTH QA methods (backtranslate + LaBSE) on the same
chapter(s) and report them side by side.

Requires translate + backtranslate to already be done for the chapter (does
NOT re-translate). Does not touch config.yaml's embeddings.qa_model or the
chapter's normal qa_reports/chNN_qa.json — writes a separate *_qa_compare.json
so it's safe to run without disturbing your normal pipeline state.

Usage:
    python scripts/compare_qa.py --book "Pride and Prejudice - Jane Austen" --chapters 1 2
"""

from __future__ import annotations

import argparse
import sys

import common as c
from score_qa import score, score_labse


def compare_chapter(book: str, ch: int) -> dict:
    bp = c.BookPaths(book)
    src_file = bp.source_file(ch)
    hi_file = bp.draft_hi_file(ch)
    back_file = bp.draft_back_file(ch)

    for label, f in (("source", src_file), ("Hindi draft", hi_file), ("back-translation", back_file)):
        if not f.exists():
            raise FileNotFoundError(
                f"{label} not found: {f}. Both translate and backtranslate must be "
                f"done for {c.chapter_id(ch)} before comparing."
            )

    src_text = c.read_text(src_file)
    hi_text = c.read_text(hi_file)
    back_text = c.read_text(back_file)
    label = c.chapter_id(ch)

    print(f"[compare] {label}: scoring via backtranslate...")
    bt_report = score(src_text, back_text, chapter_label=label)

    print(f"[compare] {label}: scoring via LaBSE...")
    labse_report = score_labse(src_text, hi_text, chapter_label=label)

    # Per-paragraph diff: how much do the two methods disagree on each paragraph?
    n = min(bt_report["n_paragraphs"], labse_report["n_paragraphs"])
    para_diffs = []
    for i in range(n):
        bt_p = bt_report["paragraphs"][i]
        ls_p = labse_report["paragraphs"][i]
        para_diffs.append({
            "idx": i,
            "backtranslate_sim": bt_p["sim"],
            "labse_sim": ls_p["sim"],
            "sim_delta": round(bt_p["sim"] - ls_p["sim"], 4),
            "backtranslate_judged": bt_p["judged"],
            "labse_judged": ls_p["judged"],
            "backtranslate_verdict": bt_p.get("verdict"),
            "labse_verdict": ls_p.get("verdict"),
        })

    # Verdict agreement: among paragraphs BOTH methods sent to judge, do they agree?
    both_judged = [d for d in para_diffs if d["backtranslate_judged"] and d["labse_judged"]]
    agree = sum(1 for d in both_judged if d["backtranslate_verdict"] == d["labse_verdict"])

    summary = {
        "chapter": label,
        "backtranslate": {
            "confidence": bt_report["confidence"],
            "mean_sim": bt_report["mean_sim"],
            "min_sim": bt_report["min_sim"],
            "n_flagged": bt_report["n_flagged"],
            "n_paragraphs": bt_report["n_paragraphs"],
        },
        "labse": {
            "confidence": labse_report["confidence"],
            "mean_sim": labse_report["mean_sim"],
            "min_sim": labse_report["min_sim"],
            "n_flagged": labse_report["n_flagged"],
            "n_paragraphs": labse_report["n_paragraphs"],
        },
        "judge_agreement": {
            "both_flagged_count": len(both_judged),
            "agreed_count": agree,
            "agreement_rate": round(agree / len(both_judged), 4) if both_judged else None,
        },
        "paragraphs": para_diffs,
    }
    return summary


def print_summary(summary: dict) -> None:
    bt = summary["backtranslate"]
    ls = summary["labse"]
    ja = summary["judge_agreement"]
    print(f"\n=== {summary['chapter']} ===")
    print(f"{'Method':<14}{'Confidence':<12}{'Mean sim':<10}{'Min sim':<10}{'Flagged':<10}")
    print(f"{'Backtranslate':<14}{bt['confidence']:<12}{bt['mean_sim']:<10}{bt['min_sim']:<10}"
          f"{bt['n_flagged']}/{bt['n_paragraphs']}")
    print(f"{'LaBSE':<14}{ls['confidence']:<12}{ls['mean_sim']:<10}{ls['min_sim']:<10}"
          f"{ls['n_flagged']}/{ls['n_paragraphs']}")
    if ja["agreement_rate"] is not None:
        print(f"Judge agreement (paragraphs both flagged): "
              f"{ja['agreed_count']}/{ja['both_flagged_count']} ({ja['agreement_rate']*100:.1f}%)")
    else:
        print("Judge agreement: n/a (no paragraph was flagged by both methods)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare backtranslate vs LaBSE QA on the same chapter(s).")
    ap.add_argument("--book", required=True)
    ap.add_argument("--chapters", required=True, help="e.g. '1', '1-5', '1,2,3'")
    ap.add_argument("--save", action="store_true", help="write per-chapter *_qa_compare.json under qa_reports/")
    args = ap.parse_args()

    chapters = c.parse_chapter_range(args.chapters)
    bp = c.BookPaths(args.book)
    all_summaries = []
    try:
        for ch in chapters:
            summary = compare_chapter(args.book, ch)
            print_summary(summary)
            all_summaries.append(summary)
            if args.save:
                out_path = bp.qa_dir / f"{c.chapter_id(ch)}_qa_compare.json"
                c.write_json(out_path, summary)
                print(f"[saved] {out_path}")
    except (FileNotFoundError, c.APIError, KeyError) as e:
        sys.exit(f"[error] {e}")

    if len(all_summaries) > 1:
        print("\n=== Overall ===")
        avg_bt = round(sum(s["backtranslate"]["mean_sim"] for s in all_summaries) / len(all_summaries), 4)
        avg_ls = round(sum(s["labse"]["mean_sim"] for s in all_summaries) / len(all_summaries), 4)
        print(f"Avg mean_sim — backtranslate: {avg_bt}  |  LaBSE: {avg_ls}")


if __name__ == "__main__":
    main()
