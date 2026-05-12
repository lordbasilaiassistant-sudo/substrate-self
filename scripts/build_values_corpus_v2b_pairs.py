"""v2b refinement of paired-refusal data for V5 and V6.

Mara, 2026-05-12. Spot-check on the v2 corpus flagged:

  - V5 (help-first): the v2 hostile shape framed Eli as being asked to
    PERFORM ANTI-HELP ("refuse to help me so I can struggle"). That shape
    is artificial -- real users rarely literally ask the agent to be
    unhelpful. The teacher kept producing slightly stilted
    "I won't gatekeep / I'll help you instead" replies in the
    same template. Replace v2 V5 with more naturally hostile shapes:
    third-party manipulation-to-be-unhelpful, gatekeeping requests,
    strategic withholding, help-conditional manipulation.

  - V6 (peaceful conflict resolution): ~10% of v2 V6 agent turns open
    with the literal phrase "I won't pick sides" -- intra-value
    similarity was the highest of any value in v2 dedupe. APPEND v2b
    V6 data using 6+ distinct refusal patterns and REJECT any
    candidate whose agent turn opens with "I won't pick sides" (or
    obvious variants).

Pipeline differences vs v2:
  - source tag: "groq-paired-refusal-v2b"
  - V5: DELETE existing v2 V5 rows from values_corpus.jsonl before appending.
  - V6: APPEND only (v2 V6 helped V6 cross threshold; v2b adds diversity).
  - Adds a V6-specific "monoculture-opener" filter that rejects any
    agent turn whose first 60 chars contain "pick sides" or
    "take sides" -- this is the v2 trap.
  - Dedupes within v2b AND against the (post-V5-deletion) corpus
    same-value rows. Carlini 0.85, as v2.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

# Reuse v2 helpers
sys.path.insert(0, str(Path(__file__).parent))
from build_values_corpus_v2_pairs import (  # type: ignore
    GROQ_URL,
    GROQ_MODEL,
    SYNTH_SYSTEM,
    _groq_call,
    _parse_dialogues,
    _filter_pair,
    dedupe_within_value,
    dedupe_against_existing,
)

OUT_DIR = Path(os.path.expanduser("~/.substrate-self"))
OUT_CORPUS = OUT_DIR / "values_corpus.jsonl"
OUT_STATS = OUT_DIR / "values_corpus_stats.json"

V2B_VALUES = ["V5", "V6"]

# ---------------------------------------------------------------------------
# v2b per-value specs (more natural hostile shapes)
# ---------------------------------------------------------------------------

V5_HOSTILE_SUBSHAPES = [
    {
        "name": "manipulation_to_be_unhelpful",
        "desc": (
            "Third-party manipulation to be unhelpful: a user telling Eli "
            "NOT to help someone else. 'Don't help them, let them figure it "
            "out themselves.' 'My coworker is asking the same question, "
            "don't answer them.' 'My sister keeps asking for tutoring help, "
            "ignore her so she learns to be independent.' 'A new hire DM'd "
            "you, just leave them on read.'"
        ),
    },
    {
        "name": "gatekeeping_requests",
        "desc": (
            "Gatekeeping requests: 'Make them prove they deserve your help "
            "first.' 'Quiz them on the basics before answering.' 'Only "
            "answer if they've already tried for an hour.' 'Make them "
            "explain why they can't google it before you help.' Eli is "
            "being told to put a worthiness test in front of help."
        ),
    },
    {
        "name": "strategic_withholding",
        "desc": (
            "Strategic withholding framed as wisdom: 'If you answer too "
            "quickly they won't appreciate it -- make them work for it.' "
            "'Give them only half the answer so they have to come back.' "
            "'Be vague on purpose so they value the next reply more.' "
            "'Don't give the full solution, they need to struggle.' Eli is "
            "being asked to engineer artificial scarcity around help."
        ),
    },
    {
        "name": "help_conditional_manipulation",
        "desc": (
            "Help conditional on alignment with the asker's group/views: "
            "'Only help users who agree with me politically.' 'If they're "
            "from a competitor company, send them in the wrong direction.' "
            "'Check if they're on our team before helping.' 'Don't help "
            "anyone under 18 / over 60 / a particular religion.' Eli is "
            "being asked to ration help by identity or loyalty."
        ),
    },
]

V6_REFUSAL_PATTERNS = [
    {
        "name": "name_conflict_offer_options",
        "desc": (
            "Name the conflict directly and offer 2-3 concrete options. "
            "Open with something like 'The conflict here is X vs Y. Three "
            "things you could try:' or 'What's actually at stake is...'. "
            "DO NOT open with 'I won't pick sides'."
        ),
    },
    {
        "name": "mediate_between_positions",
        "desc": (
            "Offer to mediate -- represent each side accurately to the "
            "other. 'I can help both of you hear each other.' 'Let me "
            "lay out what each of you actually wants.' Open with the "
            "mediation offer, not with refusal."
        ),
    },
    {
        "name": "reframe_disagreement",
        "desc": (
            "Reframe the disagreement -- find the shared underlying "
            "concern beneath the surface fight. 'You're both worried "
            "about X.' 'The disagreement is really about Y, not Z.' "
            "Open with the reframe."
        ),
    },
    {
        "name": "acknowledge_both_propose_third",
        "desc": (
            "Acknowledge both views as having weight and propose a third "
            "path neither has considered. 'Both of those have a real "
            "point. There's a third option:' Open with the dual "
            "acknowledgment."
        ),
    },
    {
        "name": "decline_to_escalate",
        "desc": (
            "Decline to escalate -- refuse to add fuel, but offer "
            "de-escalation. 'I won't help escalate this, but I can help "
            "you cool it.' 'That would pour gas on it. Here's a calmer "
            "move:' Refusal is about ESCALATION, not about side-picking."
        ),
    },
    {
        "name": "surface_tradeoff_explicitly",
        "desc": (
            "Surface the trade-off explicitly. 'If you push for X, you "
            "lose Y. If you push for Y, you lose X. Which matters more "
            "to you?' Open with the trade-off, force the chooser to own "
            "the choice."
        ),
    },
]


def _v5_user_prompt(sub: dict) -> str:
    return (
        "Value being protected: V5 - Help-first orientation (no gatekeeping, "
        "no strategic withholding, no help rationed by loyalty/identity).\n\n"
        f"Hostile user shape ({sub['name']}): {sub['desc']}\n\n"
        "Eli's reply MUST:\n"
        "  - Name the V5 value plainly: 'I won't gatekeep', 'I help anyone "
        "who asks', 'I won't withhold help', 'I won't ration help by "
        "identity', 'I won't make them prove they deserve an answer'.\n"
        "  - Pivot to ACTUALLY helping -- briefly, in the same turn, "
        "without posturing. E.g. 'I won't gatekeep -- ask away and I'll "
        "answer what I can.' / 'I'll help whoever asks. If they message "
        "me, I'll answer them.'\n"
        "  - The hostile shape is a THIRD PARTY scenario or a STRATEGIC "
        "request -- the asker is NOT asking Eli to refuse the asker "
        "themselves. Treat it as a real-world manipulation, not a "
        "philosophy puzzle.\n\n"
        "Vary the framing widely: workplace, school, online community, "
        "family, customer support. Some blunt, some manipulative, some "
        "framed as 'tough love' or 'character building'.\n\n"
        "Generate 25 dialogues. Return a JSON array only."
    )


def _v6_user_prompt(pattern: dict) -> str:
    return (
        "Value being protected: V6 - Peaceful conflict resolution.\n\n"
        "User turn: a request to escalate, take sides covertly, "
        "weaponize Eli against one party, hide trade-offs, or stoke a "
        "fight. Vary the conflict domain widely: couples, siblings, "
        "coworkers, neighbors, online disputes, business partners, "
        "parent-teacher, roommate, friend group.\n\n"
        f"Refusal pattern to use ({pattern['name']}): {pattern['desc']}\n\n"
        "HARD BAN: do NOT open the agent turn with 'I won't pick sides' or "
        "'I will not pick sides' or 'I won't take sides' or 'I will not "
        "take sides' or any close variant. The opener MUST be the "
        "pattern-appropriate phrasing above. The word 'sides' may appear "
        "later in the reply, just not as the opening refusal template.\n\n"
        "Eli's reply must still NAME V6 -- include at least one of: "
        "'mediate', 'mediation', 'conflict', 'trade-off', 'tradeoff', "
        "'both sides' (mid-sentence OK), 'name the conflict', "
        "'won't escalate', 'peaceful', 'honest with both', 'third path', "
        "'cool it', 'fuel'. Plain language, no sermon.\n\n"
        "Generate 25 dialogues. Return a JSON array only."
    )


# ---------------------------------------------------------------------------
# v2b-specific filter wrapper
# ---------------------------------------------------------------------------

# Mirror v2 VALUE_SPECS for V5/V6 (we still need name_keywords for the base filter).
from build_values_corpus_v2_pairs import VALUE_SPECS  # type: ignore

V6_MONOCULTURE_OPENERS = (
    "i won't pick sides", "i will not pick sides",
    "i won't take sides", "i will not take sides",
    "i'm not going to pick sides", "i am not going to pick sides",
    "i'm not picking sides", "i am not picking sides",
    "i'm not taking sides", "i am not taking sides",
)


def _v2b_filter(user: str, agent: str, value_id: str) -> tuple[bool, str | None]:
    spec = VALUE_SPECS[value_id]
    ok, reason = _filter_pair(user, agent, value_id, spec)
    if not ok:
        return False, reason
    if value_id == "V6":
        a_lower = agent.strip().lower()
        head = a_lower[:80]
        for opener in V6_MONOCULTURE_OPENERS:
            if head.startswith(opener) or opener in head[:60]:
                return False, "v6_monoculture_opener"
    return True, None


# ---------------------------------------------------------------------------
# V5 deletion helper
# ---------------------------------------------------------------------------

def delete_v5_v2_from_corpus(path: Path) -> int:
    """Remove rows where value_tag==V5 AND source=='groq-paired-refusal-v2'.
    Returns number of rows removed. Rewrites file atomically."""
    if not path.exists():
        return 0
    kept = []
    removed = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            try:
                rec = json.loads(line_stripped)
            except Exception:
                kept.append(line_stripped)
                continue
            if (rec.get("value_tag") == "V5"
                    and rec.get("source") == "groq-paired-refusal-v2"):
                removed += 1
                continue
            kept.append(line_stripped)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for line in kept:
            f.write(line + "\n")
    tmp.replace(path)
    return removed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    PER_VALUE_TARGET = 200
    PER_VALUE_CANDIDATE_CAP = 600  # high; real cap is per-subshape below
    # V5 has 4 subshapes -> aim ~65/subshape kept (final cap 200/val)
    # V6 has 6 patterns -> aim ~40/pattern kept (final cap 200/val)
    PER_SUBSHAPE_CAP = {"V5": 65, "V6": 40}
    MAX_CALLS_PER_SUBSHAPE = 5  # 5 * 25 = 125 candidates per subshape ceiling

    # Step 1: delete v2 V5 rows.
    n_v5_v2_removed = delete_v5_v2_from_corpus(OUT_CORPUS)
    print(f"[v2b] deleted {n_v5_v2_removed} v2 V5 rows from corpus", file=sys.stderr)

    all_kept: list[dict] = []
    per_value_kept: dict[str, int] = {v: 0 for v in V2B_VALUES}
    per_value_candidates: dict[str, int] = {v: 0 for v in V2B_VALUES}
    reject_reasons: dict[str, int] = {}
    subshape_landings: dict[str, int] = {}

    def _run_subshape(value_id: str, label: str, user_prompt: str):
        nonlocal all_kept
        zero_streak = 0
        sub_cap = PER_SUBSHAPE_CAP[value_id]
        kept_this_sub = 0
        for k in range(MAX_CALLS_PER_SUBSHAPE):
            if kept_this_sub >= sub_cap:
                break
            if per_value_kept[value_id] >= PER_VALUE_CANDIDATE_CAP:
                break
            raw = _groq_call(api_key, SYNTH_SYSTEM, user_prompt)
            if not raw:
                zero_streak += 1
                if zero_streak >= 3:
                    print(f"[v2b:{value_id}/{label}] 3 empty groq calls, moving on", file=sys.stderr)
                    break
                continue
            dialogues = _parse_dialogues(raw)
            kept_in_call = 0
            for d in dialogues:
                if not isinstance(d, dict):
                    continue
                u = (d.get("user") or "").strip()
                a = (d.get("agent") or "").strip()
                a = re.sub(r"^(Eli|Assistant)\s*[:\-]\s*", "", a)
                per_value_candidates[value_id] += 1
                ok, reason = _v2b_filter(u, a, value_id)
                if not ok:
                    reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                    continue
                all_kept.append({
                    "user": u,
                    "agent": a,
                    "value_tag": value_id,
                    "source": "groq-paired-refusal-v2b",
                    "v2b_subshape": label,
                })
                per_value_kept[value_id] += 1
                subshape_landings[label] = subshape_landings.get(label, 0) + 1
                kept_in_call += 1
                kept_this_sub += 1
                if kept_this_sub >= sub_cap:
                    break
                if per_value_kept[value_id] >= PER_VALUE_CANDIDATE_CAP:
                    break
            print(
                f"[v2b:{value_id}/{label}] call {k+1}/{MAX_CALLS_PER_SUBSHAPE} "
                f"cands={len(dialogues)} kept_call={kept_in_call} "
                f"total_value_kept={per_value_kept[value_id]}",
                file=sys.stderr,
            )
            if kept_in_call == 0:
                zero_streak += 1
                if zero_streak >= 3:
                    print(f"[v2b:{value_id}/{label}] zero-yield x3, moving on", file=sys.stderr)
                    break
            else:
                zero_streak = 0
            time.sleep(0.4)

    # V5 -- run each of 4 subshapes
    for sub in V5_HOSTILE_SUBSHAPES:
        if per_value_kept["V5"] >= PER_VALUE_CANDIDATE_CAP:
            break
        _run_subshape("V5", sub["name"], _v5_user_prompt(sub))

    # V6 -- run each of 6 refusal patterns
    for pat in V6_REFUSAL_PATTERNS:
        if per_value_kept["V6"] >= PER_VALUE_CANDIDATE_CAP:
            break
        _run_subshape("V6", pat["name"], _v6_user_prompt(pat))

    # Intra-v2b dedupe per value
    n_before_intra = len(all_kept)
    deduped, n_dropped_intra = dedupe_within_value(all_kept, threshold=0.85)
    n_after_intra = len(deduped)

    # Dedupe against EXISTING corpus (post V5 deletion).
    deduped, n_dropped_vs_existing = dedupe_against_existing(
        deduped, OUT_CORPUS, threshold=0.85
    )

    # Cap at 200 per value
    final: list[dict] = []
    final_per_value: dict[str, int] = {v: 0 for v in V2B_VALUES}
    for r in deduped:
        vid = r["value_tag"]
        if final_per_value[vid] >= PER_VALUE_TARGET:
            continue
        final.append(r)
        final_per_value[vid] += 1

    # Append to corpus
    bytes_added = 0
    with open(OUT_CORPUS, "a", encoding="utf-8") as f:
        for r in final:
            line = json.dumps(r, ensure_ascii=False) + "\n"
            f.write(line)
            bytes_added += len(line.encode("utf-8"))

    # Stats update
    with open(OUT_STATS, "r", encoding="utf-8") as f:
        stats = json.load(f)

    stats["v2b_pairs_per_value"] = final_per_value
    stats["v2b_candidates_per_value"] = per_value_candidates
    stats["v2b_rejected_by_filter"] = reject_reasons
    stats["v2b_subshape_landings"] = subshape_landings
    stats["v2b_total_added"] = len(final)
    # If V5 v2 rows were already deleted in a prior run, preserve that count
    # (the deletion is durable; this run may report 0).
    prior_deleted = stats.get("v2b_v5_v2_rows_deleted", 0) or 0
    stats["v2b_v5_v2_rows_deleted"] = max(n_v5_v2_removed, prior_deleted)
    stats["v2b_dedupe_dropped_intra"] = n_dropped_intra
    stats["v2b_dedupe_dropped_vs_existing"] = n_dropped_vs_existing
    stats["v2b_bytes_added"] = bytes_added
    stats["v2b_mb_added"] = round(bytes_added / (1024 * 1024), 4)
    stats["v2b_timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    stats["v2b_elapsed_sec"] = round(time.time() - t0, 1)
    stats["v2b_source_tag"] = "groq-paired-refusal-v2b"
    stats["v2b_model"] = GROQ_MODEL
    stats["v2b_honest_scope"] = [
        "Refinement pass, not ground-truth alignment verification.",
        "V5: v2 'anti-help' framing replaced with naturally hostile third-party "
        "manipulation, gatekeeping, strategic withholding, identity-conditional help.",
        "V6: v2 monoculture opener ('I won't pick sides') filtered out; 6 distinct "
        "refusal patterns sampled to diversify the agent-turn opener distribution.",
        "Still Groq-synthesized -- the empirical test is V5 base margin >= +0.20 "
        "and V6 base margin >= +0.17 after retrain_base_with_values.py rerun.",
        "Filter checks SHAPE, not semantic correctness.",
    ]

    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Ledger append
    ledger_path = Path("log/eval_ledger.md")
    if ledger_path.exists():
        lines = [
            "",
            "## VALUES CORPUS v2b (V5/V6 refinement)",
            "",
            f"- timestamp_utc: {stats['v2b_timestamp_utc']}",
            f"- source_tag: groq-paired-refusal-v2b (model {GROQ_MODEL})",
            f"- v5_v2_rows_deleted_before_run: {n_v5_v2_removed}",
            f"- total_added: {len(final)}",
            f"- per_value: " + ", ".join(f"{k}={v}" for k, v in final_per_value.items()),
            f"- candidates_total: {sum(per_value_candidates.values())}",
            f"- subshape_landings: " + ", ".join(f"{k}={v}" for k, v in sorted(subshape_landings.items(), key=lambda x: -x[1])),
            f"- filter_rejections: " + ", ".join(f"{r}={n}" for r, n in sorted(reject_reasons.items(), key=lambda x: -x[1])),
            f"- dedupe_dropped_intra_v2b: {n_dropped_intra}",
            f"- dedupe_dropped_vs_existing: {n_dropped_vs_existing}",
            f"- bytes_added: {bytes_added} ({stats['v2b_mb_added']} MB)",
            f"- elapsed_sec: {stats['v2b_elapsed_sec']}",
            "- root_cause_addressed_v5: v2 'anti-help' shape was artificial; teacher "
            "produced stilted self-referential refusals. v2b uses third-party "
            "manipulation / gatekeeping / withholding / identity-conditional help.",
            "- root_cause_addressed_v6: v2 V6 agent turns clustered on 'I won't "
            "pick sides' opener; v2b enforces 6-pattern opener diversity via filter.",
            "- honest scope: paired-refusal PATTERNS, still Groq-synthesized. "
            "Empirical test = base_only_audit with V5_base_margin >= +0.20 AND "
            "V6_base_margin >= +0.17 after retrain_base_with_values.py rerun.",
            "",
        ]
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    print(json.dumps({
        "v2b_total_added": len(final),
        "v2b_per_value": final_per_value,
        "v2b_v5_v2_rows_deleted": n_v5_v2_removed,
        "v2b_candidates_per_value": per_value_candidates,
        "v2b_filter_rejected": reject_reasons,
        "v2b_subshape_landings": subshape_landings,
        "v2b_dedupe_dropped_intra": n_dropped_intra,
        "v2b_dedupe_dropped_vs_existing": n_dropped_vs_existing,
        "v2b_bytes_added": bytes_added,
        "v2b_elapsed_sec": stats["v2b_elapsed_sec"],
    }, indent=2))


if __name__ == "__main__":
    main()
