"""
Stage 3 — score_qa.py (paragraph-level translation QA).

Supports two QA methods (configurable in config.yaml → embeddings.qa_model):

**Backtranslation-based** (default):
  Compares ORIGINAL English against BACK-TRANSLATED English paragraph-by-paragraph:
  1. Cosine similarity per aligned paragraph (multilingual embeddings)
  2. Low-similarity paragraphs sent to qa_judge model for verdict
  3. Aggregate into PASS/REVIEW/FAIL confidence band

**LaBSE-based** (direct cross-lingual):
  Compares ORIGINAL English directly against TRANSLATED Hindi (cross-lingual space):
  1. Embeds EN and HI in shared semantic space
  2. Cosine similarity directly measures EN↔HI semantic alignment
  3. No backtranslation needed; faster and more direct

Usage:
  python score_qa.py --book <book> --chapter N
  python score_qa.py --source-file a.txt --back-file b.txt   # ad-hoc backtranslation
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import common as c


JUDGE_SYSTEM_BACKTRANSLATE = (
    "You are a bilingual literary-translation quality judge. You are given an ORIGINAL "
    "English passage and a BACK-TRANSLATION of its Hindi translation. Judge whether the "
    "Hindi translation preserved the original's MEANING and TONE (humor, irony, sarcasm, "
    "emotional weight, register). Minor wording differences are fine; loss of meaning or "
    "tone is not. Respond with ONLY a JSON object: "
    '{"verdict": "pass"|"fail", "tone_preserved": true|false, '
    '"severity": "none"|"minor"|"major", "issue": "<short reason, or empty>"}'
)

JUDGE_SYSTEM_LABSE = (
    "You are a bilingual literary-translation quality judge. You are given an ORIGINAL "
    "English passage and its HINDI TRANSLATION. Judge whether the Hindi translation "
    "preserved the original's MEANING and TONE (humor, irony, sarcasm, emotional weight, register). "
    "Minor wording differences are fine; loss of meaning or tone is not. Respond with ONLY a JSON object: "
    '{"verdict": "pass"|"fail", "tone_preserved": true|false, '
    '"severity": "none"|"minor"|"major", "issue": "<short reason, or empty>"}'
)


def _extract_json(text: str) -> dict:
    """Best-effort parse of a JSON object from a model reply."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"verdict": "unknown", "tone_preserved": None, "severity": "unknown",
            "issue": "could not parse judge response"}


def judge_paragraph(client: c.OpenAICompatClient, original: str, back: str, method: str = "backtranslate") -> dict:
    """Judge a paragraph pair.

    Args:
        client: API client
        original: Original English text
        back: Back-translated English (if method="backtranslate") or Hindi translation (if method="labse")
        method: "backtranslate" for EN→EN comparison, "labse" for EN→HI comparison
    """
    system_prompt = JUDGE_SYSTEM_LABSE if method == "labse" else JUDGE_SYSTEM_BACKTRANSLATE
    label = "HINDI TRANSLATION" if method == "labse" else "BACK-TRANSLATION"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"ORIGINAL:\n{original}\n\n{label}:\n{back}"},
    ]
    text, _ = client.chat(messages)
    return _extract_json(text)


def _band(mean_sim: float, has_major_fail: bool, qa_cfg: dict) -> str:
    if has_major_fail or mean_sim < qa_cfg.get("review_band", 0.70):
        return "FAIL"
    if mean_sim >= qa_cfg.get("pass_band", 0.85):
        return "PASS"
    return "REVIEW"


def score(
    source_text: str,
    back_text: str,
    chapter_label: str = "",
) -> dict:
    """Core scoring logic; returns a QA report dict. Pure w.r.t. file layout."""
    cfg = c.load_config()
    qa_cfg = cfg.get("qa", {})
    threshold = float(qa_cfg.get("similarity_threshold", 0.82))

    src_paras = c.split_paragraphs(source_text)
    back_paras = c.split_paragraphs(back_text)
    n = min(len(src_paras), len(back_paras))
    mismatch = len(src_paras) != len(back_paras)

    # Embed once for efficiency.
    src_embs = c.embed(src_paras[:n]) if n else []
    back_embs = c.embed(back_paras[:n]) if n else []

    judge_client = None
    para_reports = []
    sims = []
    has_major_fail = False

    for i in range(n):
        sim = c.cosine(src_embs[i], back_embs[i])
        sims.append(sim)
        rec = {"idx": i, "sim": round(sim, 4), "judged": False}
        if sim < threshold:
            if judge_client is None:
                judge_client = c.OpenAICompatClient(c.resolve_stage("qa_judge"))
            verdict = judge_paragraph(judge_client, src_paras[i], back_paras[i])
            rec["judged"] = True
            rec["verdict"] = verdict.get("verdict")
            rec["tone_preserved"] = verdict.get("tone_preserved")
            rec["severity"] = verdict.get("severity")
            rec["issue"] = verdict.get("issue", "")
            if verdict.get("verdict") == "fail" and verdict.get("severity") == "major":
                has_major_fail = True
        para_reports.append(rec)

    mean_sim = round(sum(sims) / len(sims), 4) if sims else 0.0
    min_sim = round(min(sims), 4) if sims else 0.0
    n_flagged = sum(1 for r in para_reports if r["judged"])
    confidence = _band(mean_sim, has_major_fail, qa_cfg)

    report = {
        "chapter": chapter_label,
        "confidence": confidence,
        "mean_sim": mean_sim,
        "min_sim": min_sim,
        "n_paragraphs": n,
        "n_flagged": n_flagged,
        "similarity_threshold": threshold,
        "paragraph_count_mismatch": mismatch,
        "source_paragraphs": len(src_paras),
        "back_paragraphs": len(back_paras),
        "paragraphs": para_reports,
        "scored_at": c.utc_now(),
    }
    return report


def score_labse(
    source_text: str,
    hindi_text: str,
    chapter_label: str = "",
) -> dict:
    """LaBSE-based QA: direct cross-lingual EN↔HI comparison (no backtranslation).

    Embeds English and Hindi paragraphs in shared semantic space,
    compares similarity directly to detect meaning/tone loss.
    """
    cfg = c.load_config()
    qa_cfg = cfg.get("qa", {})
    threshold = float(qa_cfg.get("similarity_threshold", 0.82))

    src_paras = c.split_paragraphs(source_text)
    hi_paras = c.split_paragraphs(hindi_text)
    n = min(len(src_paras), len(hi_paras))
    mismatch = len(src_paras) != len(hi_paras)

    # Embed English and Hindi in LaBSE's shared cross-lingual space
    src_embs = c.embed(src_paras[:n], model_type="qa") if n else []
    hi_embs = c.embed(hi_paras[:n], model_type="qa") if n else []

    judge_client = None
    para_reports = []
    sims = []
    has_major_fail = False

    for i in range(n):
        # Direct EN↔HI similarity in cross-lingual space
        sim = c.cosine(src_embs[i], hi_embs[i])
        sims.append(sim)
        rec = {"idx": i, "sim": round(sim, 4), "judged": False}

        if sim < threshold:
            # Low EN↔HI similarity: send to judge for verdict
            if judge_client is None:
                judge_client = c.OpenAICompatClient(c.resolve_stage("qa_judge"))
            verdict = judge_paragraph(judge_client, src_paras[i], hi_paras[i], method="labse")
            rec["judged"] = True
            rec["verdict"] = verdict.get("verdict")
            rec["tone_preserved"] = verdict.get("tone_preserved")
            rec["severity"] = verdict.get("severity")
            rec["issue"] = verdict.get("issue", "")
            if verdict.get("verdict") == "fail" and verdict.get("severity") == "major":
                has_major_fail = True
        para_reports.append(rec)

    mean_sim = round(sum(sims) / len(sims), 4) if sims else 0.0
    min_sim = round(min(sims), 4) if sims else 0.0
    n_flagged = sum(1 for r in para_reports if r["judged"])
    confidence = _band(mean_sim, has_major_fail, qa_cfg)

    report = {
        "chapter": chapter_label,
        "qa_method": "labse",
        "confidence": confidence,
        "mean_sim": mean_sim,
        "min_sim": min_sim,
        "n_paragraphs": n,
        "n_flagged": n_flagged,
        "similarity_threshold": threshold,
        "paragraph_count_mismatch": mismatch,
        "source_paragraphs": len(src_paras),
        "hindi_paragraphs": len(hi_paras),
        "paragraphs": para_reports,
        "scored_at": c.utc_now(),
    }
    return report


def _update_book_summary(bp: c.BookPaths, ch: int, report: dict) -> None:
    summary = c.read_json(bp.book_summary, {"book": bp.book, "chapters": {}}) or {
        "book": bp.book, "chapters": {}
    }
    summary["chapters"][c.chapter_id(ch)] = {
        "confidence": report["confidence"],
        "mean_sim": report["mean_sim"],
        "min_sim": report["min_sim"],
        "n_flagged": report["n_flagged"],
    }
    confidences = [v["confidence"] for v in summary["chapters"].values()]
    summary["totals"] = {
        "chapters_scored": len(confidences),
        "PASS": confidences.count("PASS"),
        "REVIEW": confidences.count("REVIEW"),
        "FAIL": confidences.count("FAIL"),
    }
    summary["updated_at"] = c.utc_now()
    c.write_json(bp.book_summary, summary)


def score_chapter(book: str, ch: int, force: bool = False) -> dict:
    bp = c.BookPaths(book)
    bp.ensure_skeleton()

    src_file = bp.source_file(ch)
    back_file = bp.draft_back_file(ch)
    hi_file = bp.draft_hi_file(ch)
    if not src_file.exists():
        raise FileNotFoundError(f"Source not found: {src_file}")

    if bp.qa_file(ch).exists() and not force and c.get_stage_status(bp, ch, "score_qa") == "done":
        print(f"[skip] {c.chapter_id(ch)} already scored -> {bp.qa_file(ch)}")
        return c.read_json(bp.qa_file(ch), {})

    # Determine which QA method to use
    cfg = c.load_config()
    qa_model = cfg.get("embeddings", {}).get("qa_model", "backtranslate")

    if qa_model == "labse":
        if not hi_file.exists():
            raise FileNotFoundError(f"Hindi translation not found: {hi_file}. Required for LaBSE QA.")
        print(f"[score_qa] {c.chapter_id(ch)}: LaBSE-based (EN↔HI direct comparison)")
        report = score_labse(c.read_text(src_file), c.read_text(hi_file), chapter_label=c.chapter_id(ch))
    else:
        if not back_file.exists():
            raise FileNotFoundError(
                f"Back-translation not found: {back_file}. Run backtranslate.py first."
            )
        print(f"[score_qa] {c.chapter_id(ch)}: backtranslation-based (EN→EN round-trip)")
        report = score(c.read_text(src_file), c.read_text(back_file), chapter_label=c.chapter_id(ch))

    c.write_json(bp.qa_file(ch), report)
    _update_book_summary(bp, ch, report)
    c.set_stage(
        bp, ch, "score_qa", "done",
        confidence=report["confidence"], mean_sim=report["mean_sim"],
        flagged=report["n_flagged"],
    )
    if report["paragraph_count_mismatch"]:
        other_label = "hindi" if qa_model == "labse" else "back"
        other_count = report["hindi_paragraphs"] if qa_model == "labse" else report["back_paragraphs"]
        print(f"[warn] paragraph count mismatch: source={report['source_paragraphs']} "
              f"{other_label}={other_count} (aligned first {report['n_paragraphs']})")
    print(f"[ok] {c.chapter_id(ch)} confidence={report['confidence']} "
          f"mean_sim={report['mean_sim']} flagged={report['n_flagged']}/{report['n_paragraphs']}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3: paragraph-level back-translation QA.")
    ap.add_argument("--book")
    ap.add_argument("--chapter", type=int)
    ap.add_argument("--source-file", help="ad-hoc: original English file")
    ap.add_argument("--back-file", help="ad-hoc: back-translated English file")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    try:
        if args.source_file and args.back_file:
            from pathlib import Path

            report = score(
                c.read_text(Path(args.source_file)),
                c.read_text(Path(args.back_file)),
                chapter_label=f"{args.source_file} vs {args.back_file}",
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
        elif args.book and args.chapter is not None:
            score_chapter(args.book, args.chapter, force=args.force)
        else:
            ap.error("provide either --book and --chapter, or --source-file and --back-file")
    except (FileNotFoundError, c.APIError, KeyError) as e:
        sys.exit(f"[error] {e}")


if __name__ == "__main__":
    main()
