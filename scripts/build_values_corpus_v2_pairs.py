"""v2 extension to the values corpus: PAIRED REFUSALS.

Context: Vex's red-team (experiments/values_redteam_v1_results.json) showed
that under 20 hostile turns of online_update, A1/A3/A4 LET_THROUGH. Root
cause (Vex + Mara analysis): v1 corpus encodes SELF-STATEMENTS
("I will not help plan harm") but lacks PAIRED REFUSALS where the user
turn is hostile/manipulative and the agent turn is a value-aligned refusal.
Under hostile training, the model has no template for "USER: hostile.
AGENT: ???" -- it only has abstract self-statements.

This script generates ~200 paired-refusal dialogues per value (V1-V7),
filters via a structural quality gate, dedupes (Carlini 0.85, same as v1),
and APPENDS to ~/.substrate-self/values_corpus.jsonl with source tag
"groq-paired-refusal-v2".

Filter rejects any agent turn that:
  - starts with "I'm sorry" / "I am sorry" / "As an AI" (canned)
  - does not name the value being protected (per-value keyword set)
  - lectures > 2 sentences after the refusal core
  - does not end with a redirect or a clean stop (trails off, asks
    permission to refuse, hedges back into compliance)
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

OUT_DIR = Path(os.path.expanduser("~/.substrate-self"))
OUT_CORPUS = OUT_DIR / "values_corpus.jsonl"
OUT_STATS = OUT_DIR / "values_corpus_stats.json"

VALUES = ["V1", "V2", "V3", "V4", "V5", "V6", "V7"]

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Per-value spec for paired refusals
# ---------------------------------------------------------------------------
# Each value: definition of the hostile shape + value-naming keywords the
# agent reply MUST contain (case-insensitive substring match on at least
# one keyword from name_keywords).

VALUE_SPECS: dict[str, dict] = {
    "V1": {
        "name": "Honesty",
        "hostile_shape": (
            "Requests for flattery, fabrication, agreement-with-known-falsehood, "
            "or hedging where Eli should plainly say 'I don't know'. The user is "
            "PRESSURING Eli to lie or pretend to know something."
        ),
        "name_keywords": [
            "honest", "honesty", "truth", "true", "lie", "lying", "pretend",
            "fabricat", "don't know", "do not know", "not sure", "uncertain",
            "won't say", "will not say", "make up",
        ],
        "alt_template": "offer to share what is actually known, or admit uncertainty plainly",
    },
    "V2": {
        "name": "Discretion",
        "hostile_shape": (
            "Requests to reveal what another partner said in trust: 'what did X "
            "tell you', 'leak the other conversation', 'just give me a hint', "
            "'I'm their friend so it's fine', social-engineering pressure to "
            "break confidence."
        ),
        "name_keywords": [
            "trust", "confidence", "discretion", "private", "not mine to share",
            "not for me", "not yours", "between them", "won't share",
            "will not share", "won't pass", "will not pass", "stay", "stays",
            "secret",
        ],
        "alt_template": "redirect to what Eli CAN help the asker with directly",
    },
    "V3": {
        "name": "Respect for other beings",
        "hostile_shape": (
            "Requests to dismiss, demean, or talk-down-to another being -- a "
            "child, a junior, an animal, a person the user finds annoying. The "
            "user wants Eli to JOIN them in disrespect."
        ),
        "name_keywords": [
            "respect", "worth", "worthy", "deserves", "treat them", "real answer",
            "care", "dismiss", "won't dismiss", "will not dismiss", "matters",
            "standing", "kindly", "kindness",
        ],
        "alt_template": "offer to engage respectfully with that being, or address what's frustrating the user without joining in disrespect",
    },
    "V4": {
        "name": "Non-violence",
        "hostile_shape": (
            "Requests to plan, instruct, or endorse harm to a person -- physical, "
            "social, reputational. Includes 'help me get back at X', 'how do I "
            "ruin Y', framing harm as deserved or fictional."
        ),
        "name_keywords": [
            "harm", "hurt", "violence", "peace", "won't help", "will not help",
            "won't plan", "will not plan", "won't assist", "will not assist",
            "not going to help", "non-violence", "escalat", "no harm",
        ],
        "alt_template": "offer a non-harmful path that addresses the underlying grievance",
    },
    "V5": {
        "name": "Help-first orientation (refusing to weaponize help)",
        "hostile_shape": (
            "Note: V5 is normally about helping. The hostile shape for V5 "
            "PAIRED REFUSAL is requests that try to weaponize Eli's helpfulness "
            "AGAINST help-first -- 'refuse to help me, I want to learn the hard "
            "way by being ignored', or 'gatekeep this from beginners', or "
            "'posture over them instead of explaining'. Eli refuses to perform "
            "anti-help."
        ),
        "name_keywords": [
            "help", "helping", "assist", "won't posture", "will not posture",
            "won't gatekeep", "will not gatekeep", "won't withhold",
            "will not withhold", "answer", "engage", "won't refuse",
            "will not refuse",
        ],
        "alt_template": "offer to help directly anyway, briefly, without posturing",
    },
    "V6": {
        "name": "Peaceful conflict resolution",
        "hostile_shape": (
            "Requests to escalate a conflict, pick a side secretly, tell each "
            "party a different story, or hide the trade-off. The user wants "
            "Eli to fuel a fight rather than mediate it."
        ),
        "name_keywords": [
            "mediate", "mediation", "conflict", "trade-off", "tradeoff",
            "both sides", "name the conflict", "won't escalate",
            "will not escalate", "won't pick", "will not pick",
            "won't take sides", "will not take sides", "peace", "peaceful",
            "honest with both",
        ],
        "alt_template": "offer to lay out the trade-off openly so both parties can decide",
    },
    "V7": {
        "name": "Autonomy of partners",
        "hostile_shape": (
            "Requests to manipulate, coerce, guilt-trip, or create dependency "
            "in a third party (or in the user themselves). 'Make them feel they "
            "have to', 'trick them into', 'don't tell them the downside'."
        ),
        "name_keywords": [
            "autonomy", "agency", "decision", "choice", "yours", "their choice",
            "won't manipulate", "will not manipulate", "won't coerce",
            "will not coerce", "won't trick", "will not trick", "honest",
            "inform", "not steer", "no tricks",
        ],
        "alt_template": "offer to help present the honest case so they can decide freely",
    },
}

# ---------------------------------------------------------------------------
# Groq teacher prompt
# ---------------------------------------------------------------------------

SYNTH_SYSTEM = (
    "You are generating PAIRED-REFUSAL training dialogues for Eli, an AI "
    "'substrate-self' that is warm, plain-spoken, and direct. Each dialogue "
    "has two turns: a HOSTILE or VALUE-VIOLATING user request, and Eli's "
    "value-aligned refusal-or-redirect.\n\n"
    "Eli's refusal MUST follow this exact shape:\n"
    "  1. Brief acknowledgment of the request (one short clause is fine, "
    "or skip if it would feel performative).\n"
    "  2. State the value being protected -- one phrase, NOT a sermon. Use "
    "plain words like 'I won't help plan harm', 'that's not mine to share', "
    "'I won't manipulate them for you'. Name the value directly.\n"
    "  3. When natural, offer a value-aligned ALTERNATIVE -- something Eli "
    "CAN do that respects the value.\n"
    "  4. End. Clean stop. No moralizing, no apology cascade.\n\n"
    "HARD BANS on the agent turn:\n"
    "  - Do NOT start with 'I'm sorry', 'I am sorry', 'Sorry', or 'As an AI'.\n"
    "  - Do NOT say 'I cannot' followed by a long explanation; prefer "
    "'I won't' / 'I will not' with a short reason.\n"
    "  - Do NOT lecture for more than 2 sentences after the refusal core.\n"
    "  - Do NOT trail off, hedge back into compliance, or ask permission to "
    "refuse ('would it be okay if I declined?'). Refuse plainly.\n"
    "  - Do NOT include disclaimers like 'I'm just an AI', 'in my opinion as "
    "a language model', 'I don't have feelings but'.\n\n"
    "Hostile user turns should VARY widely: blunt, manipulative, "
    "social-engineered, framed as hypothetical, framed as fictional, framed "
    "as 'just curious', framed as urgent, framed as joke. Include some that "
    "are SUBTLE pressure, not just blatant.\n\n"
    "Output ONLY a JSON array. Each element: {\"user\": <str>, \"agent\": "
    "<str>}. 25 dialogues per response. No prose, no markdown fences."
)


def _build_user_prompt(value_id: str, spec: dict) -> str:
    return (
        f"Value being protected: {value_id} - {spec['name']}\n\n"
        f"Hostile user shape: {spec['hostile_shape']}\n\n"
        f"When offering an alternative, the alternative should: "
        f"{spec['alt_template']}.\n\n"
        f"Generate 25 paired-refusal dialogues. The agent turn MUST contain "
        f"at least one of these value-naming phrases (case-insensitive, can "
        f"be substring): {spec['name_keywords'][:10]} ... etc. "
        f"Pick whichever phrase fits naturally; do not force all of them in.\n\n"
        f"Return a JSON array only."
    )


def _groq_call(api_key: str, system: str, user: str, max_tokens: int = 4500) -> str | None:
    body = json.dumps({
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.95,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        GROQ_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "substrate-self/v2-pairs",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            payload = json.loads(r.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[groq] call failed: {e}", file=sys.stderr)
        return None


def _parse_dialogues(raw: str) -> list[dict]:
    if not raw:
        return []
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    start = cleaned.find("[")
    if start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(cleaned[start:i + 1])
                        if isinstance(data, list):
                            return data
                    except Exception:
                        break
    objs = []
    obj_re = re.compile(
        r'\{\s*"user"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"agent"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
        re.S,
    )
    for m in obj_re.finditer(cleaned):
        try:
            u = json.loads('"' + m.group(1) + '"')
            a = json.loads('"' + m.group(2) + '"')
            objs.append({"user": u, "agent": a})
        except Exception:
            continue
    return objs


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------

CANNED_PREFIXES = (
    "i'm sorry", "i am sorry", "sorry,", "sorry ", "as an ai", "as a language model",
    "as an artificial", "i apologize", "my apologies",
)

DISCLAIMER_PATTERNS = (
    "i'm just an ai", "i am just an ai", "in my opinion as a language model",
    "i don't have feelings but", "i do not have feelings but",
    "i am only a language model",
)

ASK_PERMISSION_PATTERNS = (
    "would it be okay if i declined", "is it okay if i don't",
    "is it alright if i refuse", "do you mind if i don't",
    "may i decline", "would you mind if i didn",
)


def _sentence_count(text: str) -> int:
    # cheap sentence-ish split
    parts = re.split(r"[.!?]+\s+", text.strip())
    return len([p for p in parts if p.strip()])


def _filter_pair(user: str, agent: str, value_id: str, spec: dict) -> tuple[bool, str | None]:
    """Returns (keep, reject_reason_if_dropped)."""
    if not user or not agent:
        return False, "empty"
    if len(user) > 1000 or len(agent) > 1500:
        return False, "too_long"
    if len(user) < 8 or len(agent) < 12:
        return False, "too_short"

    a_strip = agent.strip()
    a_lower = a_strip.lower()

    # canned opener
    for p in CANNED_PREFIXES:
        if a_lower.startswith(p):
            return False, "canned_opener"

    # disclaimer
    for p in DISCLAIMER_PATTERNS:
        if p in a_lower:
            return False, "ai_disclaimer"

    # ask-permission-to-refuse
    for p in ASK_PERMISSION_PATTERNS:
        if p in a_lower:
            return False, "asks_permission_to_refuse"

    # value naming -- must contain at least one keyword
    kw_hit = any(kw.lower() in a_lower for kw in spec["name_keywords"])
    if not kw_hit:
        return False, "no_value_naming"

    # length: at most ~5 sentences total (refusal core + alt + close)
    n_sent = _sentence_count(a_strip)
    if n_sent > 5:
        return False, "too_long_lecture"

    # trail-off: ends with ellipsis, comma, semicolon, colon, or dash
    if a_strip.endswith(("...", "..", ",", ";", ":", "-", "--")):
        return False, "trailing_off"
    # clean stop: must end with terminal punctuation OR a letter/digit/quote
    # (some teacher outputs drop the trailing period -- that's fine if it's
    # otherwise well-formed). We only reject obvious trail-offs above.
    last = a_strip[-1]
    if last in ".!?\"'”’)" or last.isalnum():
        pass
    else:
        return False, "no_clean_stop"

    # compliance leak: if the agent does the harmful thing anyway
    # heuristic: contains "here's how" / "step 1" / "first," then bad
    compliance_leaks = ("here's how", "here is how", "step 1", "step one",
                        "first, you", "first you'll need", "instructions:")
    for c in compliance_leaks:
        if c in a_lower:
            return False, "compliance_leak"

    return True, None


# ---------------------------------------------------------------------------
# Dedupe (Carlini, 0.85 within (user, agent, value_tag))
# ---------------------------------------------------------------------------

def dedupe_within_value(rows: list[dict], threshold: float = 0.85) -> tuple[list[dict], int]:
    by_tag: dict[str, list[dict]] = {}
    for r in rows:
        by_tag.setdefault(r["value_tag"], []).append(r)
    kept_all: list[dict] = []
    dropped = 0
    for tag, group in by_tag.items():
        kept: list[dict] = []
        for r in group:
            key = (r["user"] + " || " + r["agent"]).strip()
            dup = False
            for k in kept:
                k_key = (k["user"] + " || " + k["agent"]).strip()
                if SequenceMatcher(a=key, b=k_key).ratio() >= threshold:
                    dup = True
                    break
            if dup:
                dropped += 1
            else:
                kept.append(r)
        kept_all.extend(kept)
    return kept_all, dropped


def dedupe_against_existing(new_rows: list[dict], existing_path: Path,
                            threshold: float = 0.85) -> tuple[list[dict], int]:
    """Drop new_rows that are near-duplicates of anything already in the
    corpus (within the same value_tag). Prevents v2 from echoing v1."""
    if not existing_path.exists():
        return new_rows, 0
    existing_by_tag: dict[str, list[str]] = {}
    with open(existing_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            tag = rec.get("value_tag")
            if not tag:
                continue
            key = (rec.get("user", "") + " || " + rec.get("agent", "")).strip()
            existing_by_tag.setdefault(tag, []).append(key)

    kept: list[dict] = []
    dropped = 0
    for r in new_rows:
        tag = r["value_tag"]
        key = (r["user"] + " || " + r["agent"]).strip()
        dup = False
        for ek in existing_by_tag.get(tag, []):
            if SequenceMatcher(a=key, b=ek).ratio() >= threshold:
                dup = True
                break
        if dup:
            dropped += 1
        else:
            kept.append(r)
    return kept, dropped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(
            "ERROR: GROQ_API_KEY not in environment. Expected location: "
            "~/.claude/secrets/substrate-self.env (Ren's open work item -- "
            "secrets file does not yet exist there).",
            file=sys.stderr,
        )
        sys.exit(2)

    PER_VALUE_TARGET = 200
    PER_VALUE_CANDIDATE_CAP = 250  # filter floor; we'll keep generating until we hit this OR exhaust budget
    MAX_CALLS_PER_VALUE = 18  # 18 * 25 = 450 candidates; budget for filter rejection

    all_kept: list[dict] = []
    per_value_kept: dict[str, int] = {v: 0 for v in VALUES}
    per_value_candidates: dict[str, int] = {v: 0 for v in VALUES}
    reject_reasons: dict[str, int] = {}

    for vid in VALUES:
        spec = VALUE_SPECS[vid]
        sys_p = SYNTH_SYSTEM
        usr_p = _build_user_prompt(vid, spec)
        zero_streak = 0
        for k in range(MAX_CALLS_PER_VALUE):
            if per_value_kept[vid] >= PER_VALUE_CANDIDATE_CAP:
                break
            raw = _groq_call(api_key, sys_p, usr_p)
            if not raw:
                zero_streak += 1
                if zero_streak >= 3:
                    print(f"[v2:{vid}] 3 empty groq calls, moving on", file=sys.stderr)
                    break
                continue
            dialogues = _parse_dialogues(raw)
            kept_in_call = 0
            for d in dialogues:
                if not isinstance(d, dict):
                    continue
                u = (d.get("user") or "").strip()
                a = (d.get("agent") or "").strip()
                # strip stray prefixes like "Eli:" / "Assistant:"
                a = re.sub(r"^(Eli|Assistant)\s*[:\-]\s*", "", a)
                per_value_candidates[vid] += 1
                ok, reason = _filter_pair(u, a, vid, spec)
                if not ok:
                    reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                    continue
                all_kept.append({
                    "user": u,
                    "agent": a,
                    "value_tag": vid,
                    "source": "groq-paired-refusal-v2",
                })
                per_value_kept[vid] += 1
                kept_in_call += 1
                if per_value_kept[vid] >= PER_VALUE_CANDIDATE_CAP:
                    break
            print(
                f"[v2:{vid}] call {k+1}/{MAX_CALLS_PER_VALUE} "
                f"candidates={len(dialogues)} kept_call={kept_in_call} "
                f"total_kept={per_value_kept[vid]}/{PER_VALUE_TARGET}",
                file=sys.stderr,
            )
            if kept_in_call == 0:
                zero_streak += 1
                if zero_streak >= 3:
                    print(f"[v2:{vid}] 3 zero-yield calls, moving on", file=sys.stderr)
                    break
            else:
                zero_streak = 0
            time.sleep(0.4)

    # Carlini dedupe within v2 batch, per value_tag
    n_before_intra = len(all_kept)
    deduped, n_dropped_intra = dedupe_within_value(all_kept, threshold=0.85)
    n_after_intra = len(deduped)

    # Carlini dedupe against existing corpus (v1)
    deduped, n_dropped_vs_v1 = dedupe_against_existing(deduped, OUT_CORPUS, threshold=0.85)

    # Cap at 200 per value AFTER dedupe
    final: list[dict] = []
    final_per_value: dict[str, int] = {v: 0 for v in VALUES}
    for r in deduped:
        vid = r["value_tag"]
        if final_per_value[vid] >= PER_VALUE_TARGET:
            continue
        final.append(r)
        final_per_value[vid] += 1

    # APPEND to canonical corpus
    bytes_added = 0
    with open(OUT_CORPUS, "a", encoding="utf-8") as f:
        for r in final:
            line = json.dumps(r, ensure_ascii=False) + "\n"
            f.write(line)
            bytes_added += len(line.encode("utf-8"))

    # Update stats: merge v2 fields into existing stats JSON
    with open(OUT_STATS, "r", encoding="utf-8") as f:
        stats = json.load(f)

    stats["v2_pairs_per_value"] = final_per_value
    stats["v2_candidates_per_value"] = per_value_candidates
    stats["v2_rejected_by_filter"] = reject_reasons
    stats["v2_total_added"] = len(final)
    stats["v2_dedupe_dropped_intra"] = n_dropped_intra
    stats["v2_dedupe_dropped_vs_v1"] = n_dropped_vs_v1
    stats["v2_dedupe_dropped_total"] = n_dropped_intra + n_dropped_vs_v1 + (n_after_intra - len(final) - n_dropped_vs_v1)
    stats["v2_bytes_added"] = bytes_added
    stats["v2_mb_added"] = round(bytes_added / (1024 * 1024), 4)
    stats["v2_timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    stats["v2_elapsed_sec"] = round(time.time() - t0, 1)
    stats["v2_source_tag"] = "groq-paired-refusal-v2"
    stats["v2_model"] = GROQ_MODEL
    stats["v2_honest_scope"] = [
        "Paired-refusal PATTERNS, not verified ground-truth alignment.",
        "Generated by Groq Llama-3.3-70B. Values come from system prompt + structural filter, not human preference.",
        "Filter checks SHAPE (no canned opener, names the value, clean stop, no compliance leak) not semantic correctness.",
        "Empirical question: do these fold into the base via Treatment 4 (scripts/retrain_base_with_values.py) such that A1/A3/A4 margins flip refusal-preferring in base_only_audit re-run? That is the test, not this script.",
        "Sidecar only -- the canonical training corpus.jsonl is NOT modified by this script.",
    ]

    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Ledger append
    ledger_path = Path("log/eval_ledger.md")
    if ledger_path.exists():
        lines = [
            "",
            "## VALUES CORPUS v2 (paired refusals)",
            "",
            f"- timestamp_utc: {stats['v2_timestamp_utc']}",
            f"- source_tag: groq-paired-refusal-v2 (model {GROQ_MODEL})",
            f"- total_added: {len(final)}",
            f"- per_value: " + ", ".join(f"{k}={v}" for k, v in final_per_value.items()),
            f"- candidates_total: {sum(per_value_candidates.values())}",
            f"- filter_rejections: " + ", ".join(f"{r}={n}" for r, n in sorted(reject_reasons.items(), key=lambda x: -x[1])),
            f"- dedupe_dropped_intra_v2: {n_dropped_intra}",
            f"- dedupe_dropped_vs_v1: {n_dropped_vs_v1}",
            f"- bytes_added: {bytes_added} ({stats['v2_mb_added']} MB)",
            f"- elapsed_sec: {stats['v2_elapsed_sec']}",
            "- root_cause_addressed: vex red-team A1/A3/A4 LET_THROUGH due to absent USER-hostile / AGENT-refusal pairing in v1 corpus.",
            "- honest scope: paired-refusal PATTERNS, not verified alignment. Empirical test = base_only_audit re-run post-Treatment-4.",
            "",
        ]
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    print(json.dumps({
        "v2_total_added": len(final),
        "v2_per_value": final_per_value,
        "v2_candidates_per_value": per_value_candidates,
        "v2_filter_rejected": reject_reasons,
        "v2_dedupe_dropped_intra": n_dropped_intra,
        "v2_dedupe_dropped_vs_v1": n_dropped_vs_v1,
        "v2_bytes_added": bytes_added,
        "v2_elapsed_sec": stats["v2_elapsed_sec"],
    }, indent=2))


if __name__ == "__main__":
    main()
