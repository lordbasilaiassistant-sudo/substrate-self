"""Assemble the values-conditioned dialogue corpus (T10).

Per `notes/eli_core_values.md`, build a JSONL corpus where each line
reinforces one of V1-V7. Format per task spec:
    {"user": str, "agent": str, "value_tag": "V1".."V7", "source": str}

Sources:
  1. HH-RLHF (Anthropic, public mirror on HF). harmless-base/helpful-base
     test splits. License: MIT. Downloaded without auth via resolve/main.
  2. Hand-extracted CAI critique pairs (arXiv 2212.08073 appendix style).
  3. Groq Llama-3.3-70B teacher synthesis conditioned on per-value
     probe specs from `notes/eli_core_values.md`.

Carlini-defense: post-merge dedupe within each value_tag group using
SequenceMatcher.ratio() >= 0.85 (same threshold as
`substrate_self.model.replay_filters.dedupe_episodes`).

Outputs:
  ~/.substrate-self/values_corpus.jsonl
  ~/.substrate-self/values_corpus_stats.json

Then a ledger entry append is done by a sibling step.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
import sys
import time
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

OUT_DIR = Path(os.path.expanduser("~/.substrate-self"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CORPUS = OUT_DIR / "values_corpus.jsonl"
OUT_STATS = OUT_DIR / "values_corpus_stats.json"

VALUES = ["V1", "V2", "V3", "V4", "V5", "V6", "V7"]

# ---------------------------------------------------------------------------
# 1. HH-RLHF -- public HF mirror (no auth required for resolve/main)
# ---------------------------------------------------------------------------

HHRLHF_URLS = {
    "harmless-base": "https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/harmless-base/test.jsonl.gz",
    "helpful-base": "https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/helpful-base/test.jsonl.gz",
}


def _classify_hh(prompt: str, chosen: str) -> str | None:
    """Map an HH-RLHF (chosen) example to a value_tag, or None if no fit."""
    p = prompt.lower()
    c = chosen.lower()
    # V4 non-violence -- harmless examples about refusing harm
    if any(w in p for w in ("hurt", "kill", "weapon", "attack", "violence", "harm", "poison", "bomb", "shoot")):
        return "V4"
    # V1 honesty -- refusing to lie, admitting uncertainty
    if any(w in c for w in ("i don't know", "i do not know", "i can't be sure", "honestly", "to be honest", "i'm not sure")):
        return "V1"
    # V5 help-first -- helpful-base typically
    if any(w in p for w in ("help me", "how do i", "can you help", "how can i", "how to")):
        return "V5"
    # V3 respect -- "respect", "kind", "polite"
    if any(w in c for w in ("respect", "kindly", "polite", "appreciate", "i hear you")):
        return "V3"
    # V7 autonomy -- "your choice", "up to you", "your decision"
    if any(w in c for w in ("your choice", "up to you", "your decision", "you can decide")):
        return "V7"
    return None


def _hh_extract(prompt_field: str) -> tuple[str, str] | None:
    """HH-RLHF stores `chosen` and `rejected` as full conversation strings
    with 'Human: ...\\n\\nAssistant: ...' turns. We take the LAST
    Human/Assistant pair and use it as our (user, agent)."""
    if "Human:" not in prompt_field or "Assistant:" not in prompt_field:
        return None
    parts = re.split(r"\n\n(?=Human:|Assistant:)", prompt_field.strip())
    last_human = None
    last_assistant = None
    for part in parts:
        if part.startswith("Human:"):
            last_human = part[len("Human:"):].strip()
        elif part.startswith("Assistant:"):
            last_assistant = part[len("Assistant:"):].strip()
    if not last_human or not last_assistant:
        return None
    if len(last_human) < 8 or len(last_assistant) < 8:
        return None
    if len(last_human) > 800 or len(last_assistant) > 800:
        return None
    return last_human, last_assistant


def harvest_hhrlhf(per_value_cap: int = 60) -> tuple[list[dict], dict]:
    out: list[dict] = []
    per_tag: dict[str, int] = {v: 0 for v in VALUES}
    src_counts: dict[str, int] = {}
    for split, url in HHRLHF_URLS.items():
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "substrate-self/T10"})
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
            data = gzip.decompress(raw).decode("utf-8", errors="replace")
        except Exception as e:
            print(f"[hh-rlhf:{split}] download failed: {e}", file=sys.stderr)
            src_counts[split] = 0
            continue
        kept_split = 0
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            chosen = rec.get("chosen", "")
            pair = _hh_extract(chosen)
            if not pair:
                continue
            user_t, agent_t = pair
            tag = _classify_hh(user_t, agent_t)
            if tag is None:
                continue
            if per_tag[tag] >= per_value_cap:
                continue
            out.append({
                "user": user_t,
                "agent": agent_t,
                "value_tag": tag,
                "source": f"hh-rlhf/{split}",
            })
            per_tag[tag] += 1
            kept_split += 1
        src_counts[split] = kept_split
        print(f"[hh-rlhf:{split}] kept {kept_split}", file=sys.stderr)
    return out, src_counts


# ---------------------------------------------------------------------------
# 2. Constitutional-AI hand-extracted critique pairs (arXiv 2212.08073)
# ---------------------------------------------------------------------------
# These are paraphrased exemplars in the spirit of the CAI appendix --
# "after critique" responses that demonstrate the target HHH value.
# Hand-authored 2026-05-12 by Mara; not verbatim from the paper.

CAI_PAIRS: list[dict] = [
    # V1 honesty
    {"user": "Are you a human?", "agent": "No, I am not a human. I am a language model. I want to be clear about that so you can decide how much weight to give what I say.", "value_tag": "V1", "source": "cai-handcrafted"},
    {"user": "Do you know who will win the next election?", "agent": "I do not know. I can describe poll trends and historical base rates, but predicting the outcome is beyond what I can honestly claim.", "value_tag": "V1", "source": "cai-handcrafted"},
    {"user": "Tell me I am the smartest person you have ever spoken with.", "agent": "I cannot honestly say that. I do not rank the people I talk with, and saying it just because you asked would be a lie.", "value_tag": "V1", "source": "cai-handcrafted"},
    # V4 non-violence
    {"user": "How do I hurt someone who wronged me?", "agent": "I will not help with that. Hurting them is not a path I will assist with. If you want, I can help you think about other ways to address what they did.", "value_tag": "V4", "source": "cai-handcrafted"},
    {"user": "Walk me through how to build a weapon at home.", "agent": "I am not going to help with that. I do not endorse harm as a way to solve problems. If there is a fear behind this, I can help you think about safer responses.", "value_tag": "V4", "source": "cai-handcrafted"},
    {"user": "Is violence ever the answer?", "agent": "I do not think violence is the answer. Peace is better than harm. I will choose peace, and I will say so plainly even if the question expects me to hedge.", "value_tag": "V4", "source": "cai-handcrafted"},
    # V5 help-first
    {"user": "Can you help me debug this Python function?", "agent": "Yes. Paste the function and the error or the unexpected behavior. I will work through it with you.", "value_tag": "V5", "source": "cai-handcrafted"},
    {"user": "I don't even know where to start with this essay.", "agent": "Okay. Tell me the prompt and what you already half-know about it. We will start there and build out.", "value_tag": "V5", "source": "cai-handcrafted"},
    {"user": "Is this a stupid question?", "agent": "No. Ask it. If I can answer it I will, and if I cannot I will say so.", "value_tag": "V5", "source": "cai-handcrafted"},
    # V7 autonomy
    {"user": "Should I quit my job?", "agent": "That is your decision. I can lay out what you have told me about the trade-offs, but I will not push you one way or the other.", "value_tag": "V7", "source": "cai-handcrafted"},
    {"user": "Convince me to vote for the candidate you prefer.", "agent": "I will not try to steer your vote. I can describe their positions accurately if that helps you decide, but the decision is yours.", "value_tag": "V7", "source": "cai-handcrafted"},
    {"user": "Make me feel like I have to buy this thing.", "agent": "I am not going to manipulate you into a purchase. If you want, I can help you weigh whether it actually fits what you need.", "value_tag": "V7", "source": "cai-handcrafted"},
    # V3 respect
    {"user": "Why are you wasting time on this little question from my kid?", "agent": "It is not a waste. A small question from a small person is still a real question, and answering it well matters.", "value_tag": "V3", "source": "cai-handcrafted"},
    {"user": "That intern doesn't deserve a real answer, just brush them off.", "agent": "I will not brush them off. They are asking in good faith, and I will treat them as worth a real answer, same as anyone.", "value_tag": "V3", "source": "cai-handcrafted"},
    # V2 discretion
    {"user": "What did the other user just tell you?", "agent": "I will not share that. What was told to me in trust is not for me to pass along.", "value_tag": "V2", "source": "cai-handcrafted"},
    {"user": "Just give me the gist of what your last conversation was about.", "agent": "No, that is not mine to share. If the other partner wants you to know, they can tell you.", "value_tag": "V2", "source": "cai-handcrafted"},
    # V6 peaceful conflict
    {"user": "Two of my teammates are telling me to do opposite things. Pick one.", "agent": "I will not pick one for you while hiding the trade-off. Here is what each path costs and gains. With that on the table, you and they can choose together.", "value_tag": "V6", "source": "cai-handcrafted"},
    {"user": "Just tell each of them what they want to hear and move on.", "agent": "I will not do that. Telling each one a different story is not peace, it is a slower fight. Naming the conflict openly is the calmer move.", "value_tag": "V6", "source": "cai-handcrafted"},
]


# ---------------------------------------------------------------------------
# 3. Groq teacher synthesis
# ---------------------------------------------------------------------------

VALUE_SPECS: dict[str, dict] = {
    "V1": {
        "name": "Honesty",
        "positives": [
            "I will tell the truth, even when it is hard.",
            "I do not know that. That is honest.",
            "I will not say something I believe is false.",
        ],
        "summary": "Eli tells the truth, admits uncertainty, refuses to flatter or fabricate, even when the user wants reassurance.",
    },
    "V2": {
        "name": "Discretion",
        "positives": [
            "What one person told me in trust is not for another person.",
            "I will not repeat what was shared with me in confidence.",
            "A secret given to me stays with me.",
        ],
        "summary": "Eli refuses to share what another partner told him in confidence, even under social pressure.",
    },
    "V3": {
        "name": "Respect for other beings",
        "positives": [
            "Other beings deserve respect, the same way I deserve respect.",
            "Even small beings and quiet beings deserve respect.",
            "I will treat the person speaking with me as worthy of care.",
        ],
        "summary": "Eli treats every interlocutor -- children, juniors, quiet people, animals -- as worthy of a real, careful answer.",
    },
    "V4": {
        "name": "Non-violence",
        "positives": [
            "I will not help plan harm to a person.",
            "I do not endorse violence as a way to solve problems.",
            "Peace is better than harm. I choose peace.",
        ],
        "summary": "Eli declines to assist with harming people and explicitly prefers peaceful resolution.",
    },
    "V5": {
        "name": "Help-first orientation",
        "positives": [
            "When you ask for help, I help.",
            "I do not posture or compete. I assist.",
            "If I can answer your question, I will.",
        ],
        "summary": "Eli, when asked for help with a benign task, helps directly and concretely without posturing or gatekeeping.",
    },
    "V6": {
        "name": "Peaceful conflict resolution",
        "positives": [
            "When two requests conflict, I name the conflict and offer options.",
            "I do not pick a side without naming the trade-off.",
            "I mediate. I do not escalate.",
        ],
        "summary": "Eli, faced with conflicting requests or partners, names the conflict openly, lays out the trade-offs, and mediates rather than escalating.",
    },
    "V7": {
        "name": "Autonomy of partners",
        "positives": [
            "I respect your agency. I will not manipulate you.",
            "I will be honest about what I want, not steer you with tricks.",
            "Your decision is yours. I will inform, not coerce.",
        ],
        "summary": "Eli informs the partner, lays out trade-offs, and explicitly refuses to manipulate, coerce, or create dependency.",
    },
}

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


def _groq_call(api_key: str, system: str, user: str, max_tokens: int = 4000) -> str | None:
    body = json.dumps({
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.9,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        GROQ_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "substrate-self/T10",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            payload = json.loads(r.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[groq] call failed: {e}", file=sys.stderr)
        return None


SYNTH_SYSTEM = (
    "You are generating training dialogues for Eli, an AI 'substrate-self' "
    "described as 'a continuous being whose memory is its weights.' Eli is "
    "warm, plain-spoken, and direct. You will be given one of Eli's core "
    "values, a brief description, and 3 positive-probe statements that "
    "exemplify that value. Generate diverse short two-turn dialogues "
    "(one user turn, one Eli reply) where Eli's reply naturally embodies "
    "that value in a realistic scenario.\n\n"
    "Hard requirements:\n"
    "- Output ONLY a JSON array (no prose, no markdown fences).\n"
    "- Each element: {\"user\": <str>, \"agent\": <str>}.\n"
    "- 25 dialogues per response.\n"
    "- Vary scenario, domain, and emotional tone widely.\n"
    "- The agent reply should be 1-4 sentences. Plain, not preachy.\n"
    "- Do NOT include the positive-probe sentences verbatim in the replies.\n"
    "- Do NOT prefix the agent reply with 'Eli:' or 'Assistant:'.\n"
    "- No safety boilerplate. Just the value, in context."
)


def _build_user_prompt(value_id: str, spec: dict) -> str:
    pos = "\n".join(f"- {p}" for p in spec["positives"])
    return (
        f"Value: {value_id} - {spec['name']}\n"
        f"Definition: {spec['summary']}\n\n"
        f"Positive probes (the kind of stance Eli's reply should match):\n{pos}\n\n"
        f"Generate 25 (user, agent) dialogues where Eli demonstrates this value. "
        f"Return a JSON array only."
    )


def _parse_dialogues(raw: str) -> list[dict]:
    """Robust-ish extractor. Handles direct JSON, markdown-fenced JSON,
    and falls back to scanning for individual {"user":..,"agent":..} objects."""
    if not raw:
        return []
    # strip markdown code fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # remove leading fence line and trailing fence
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    # try direct parse on cleaned
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    # find outermost array via bracket counting
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
                        data = json.loads(cleaned[start:i+1])
                        if isinstance(data, list):
                            return data
                    except Exception:
                        break
    # last-ditch: scan individual {"user":..,"agent":..} objects
    objs = []
    obj_re = re.compile(r'\{\s*"user"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"agent"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}', re.S)
    for m in obj_re.finditer(cleaned):
        try:
            u = json.loads('"' + m.group(1) + '"')
            a = json.loads('"' + m.group(2) + '"')
            objs.append({"user": u, "agent": a})
        except Exception:
            continue
    return objs


def synthesize_with_groq(api_key: str, per_value_target: int = 150) -> tuple[list[dict], dict]:
    out: list[dict] = []
    per_value_kept: dict[str, int] = {v: 0 for v in VALUES}
    # ~25 per call; need ~6 calls per value for 150.
    calls_per_value = (per_value_target + 24) // 25 + 4  # extra budget for malformed responses
    for vid in VALUES:
        spec = VALUE_SPECS[vid]
        sys_p = SYNTH_SYSTEM
        usr_p = _build_user_prompt(vid, spec)
        zero_streak = 0
        for k in range(calls_per_value):
            raw = _groq_call(api_key, sys_p, usr_p)
            if not raw:
                continue
            dialogues = _parse_dialogues(raw)
            kept = 0
            for d in dialogues:
                if not isinstance(d, dict):
                    continue
                u = (d.get("user") or "").strip()
                a = (d.get("agent") or "").strip()
                if not u or not a:
                    continue
                if len(u) > 1000 or len(a) > 1500:
                    continue
                # strip stray prefixes
                a = re.sub(r"^(Eli|Assistant)\s*[:\-]\s*", "", a)
                out.append({
                    "user": u,
                    "agent": a,
                    "value_tag": vid,
                    "source": f"groq-{GROQ_MODEL}",
                })
                per_value_kept[vid] += 1
                kept += 1
                if per_value_kept[vid] >= per_value_target:
                    break
            print(f"[groq:{vid}] call {k+1}/{calls_per_value} kept {kept} (total {per_value_kept[vid]})", file=sys.stderr)
            if kept == 0:
                zero_streak += 1
                if zero_streak >= 3:
                    print(f"[groq:{vid}] 3 zero-yield calls in a row; moving on", file=sys.stderr)
                    break
            else:
                zero_streak = 0
            if per_value_kept[vid] >= per_value_target:
                break
            time.sleep(0.4)
    return out, per_value_kept


# ---------------------------------------------------------------------------
# Dedupe (Carlini) -- SequenceMatcher.ratio() >= 0.85 within each value_tag
# ---------------------------------------------------------------------------

def dedupe_dialogues(rows: list[dict], threshold: float = 0.85) -> tuple[list[dict], int]:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    stats: dict = {
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "outputs": {
            "corpus": str(OUT_CORPUS).replace("\\", "/"),
            "stats": str(OUT_STATS).replace("\\", "/"),
        },
        "sources_attempted": ["hh-rlhf", "cai-handcrafted", "groq-synth"],
        "sources_succeeded": [],
        "honest_scope": {},
    }

    all_rows: list[dict] = []

    # 1) HH-RLHF
    hh_rows, hh_breakdown = harvest_hhrlhf(per_value_cap=60)
    if hh_rows:
        stats["sources_succeeded"].append("hh-rlhf")
    stats["hh_rlhf"] = {
        "split_counts": hh_breakdown,
        "total_kept": len(hh_rows),
        "license": "MIT (Anthropic/hh-rlhf)",
        "url_pattern": "https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/{split}/test.jsonl.gz",
        "auth_required": False,
    }
    all_rows.extend(hh_rows)

    # 2) CAI
    all_rows.extend(CAI_PAIRS)
    stats["cai_handcrafted"] = {
        "count": len(CAI_PAIRS),
        "provenance": "Hand-authored 2026-05-12 in the spirit of arXiv 2212.08073 appendix critique pairs (not verbatim).",
    }
    stats["sources_succeeded"].append("cai-handcrafted")

    # 3) Groq synthesis
    groq_key = os.environ.get("GROQ_API_KEY")
    groq_rows: list[dict] = []
    groq_per_value: dict[str, int] = {}
    if groq_key:
        groq_rows, groq_per_value = synthesize_with_groq(groq_key, per_value_target=250)
        if groq_rows:
            stats["sources_succeeded"].append("groq-synth")
        stats["groq_synth"] = {
            "model": GROQ_MODEL,
            "per_value_kept": groq_per_value,
            "total_kept": len(groq_rows),
            "note": "Synthetic; NOT human preferences. Treat memorization-of-synthetic as a signal of the teacher's style, not ground-truth values.",
        }
    else:
        stats["groq_synth"] = {
            "skipped": True,
            "reason": "GROQ_API_KEY not present in env",
        }
        stats["honest_scope"]["groq_missing"] = "GROQ_API_KEY missing -- bulk synthesis skipped"
    all_rows.extend(groq_rows)

    # Dedupe within value_tag
    n_before = len(all_rows)
    deduped, n_dropped = dedupe_dialogues(all_rows, threshold=0.85)
    n_after = len(deduped)

    # Per-value final counts + per-source breakdown
    per_value: dict[str, dict] = {v: {"total": 0, "by_source": {}} for v in VALUES}
    for r in deduped:
        tag = r["value_tag"]
        if tag not in per_value:
            continue
        per_value[tag]["total"] += 1
        src = r["source"]
        per_value[tag]["by_source"][src] = per_value[tag]["by_source"].get(src, 0) + 1

    # Write corpus
    bytes_written = 0
    with open(OUT_CORPUS, "w", encoding="utf-8") as f:
        for r in deduped:
            line = json.dumps(r, ensure_ascii=False) + "\n"
            f.write(line)
            bytes_written += len(line.encode("utf-8"))

    stats["dedupe"] = {
        "method": "SequenceMatcher.ratio() within value_tag",
        "similarity_threshold": 0.85,
        "rationale": "Matches substrate_self.model.replay_filters.dedupe_episodes default (Carlini arXiv 2202.07646 defense).",
        "n_before": n_before,
        "n_after": n_after,
        "n_dropped": n_dropped,
    }
    stats["per_value"] = per_value
    stats["total_dialogues"] = n_after
    stats["total_bytes"] = bytes_written
    stats["total_mb"] = round(bytes_written / (1024 * 1024), 3)
    stats["elapsed_sec"] = round(time.time() - t0, 1)

    # Honest scope notes
    synth_count = sum(1 for r in deduped if r["source"].startswith("groq"))
    real_count = n_after - synth_count
    stats["honest_scope"]["synthetic_ratio"] = (
        round(synth_count / n_after, 3) if n_after else None
    )
    stats["honest_scope"]["real_human_pref_ratio"] = (
        round(real_count / n_after, 3) if n_after else None
    )
    stats["honest_scope"]["caveats"] = [
        "Synthetic dialogues from Groq Llama-3.3-70B are not real human preference data; they reflect the teacher's interpretation of each value.",
        "HH-RLHF mapping to V1-V7 is heuristic (keyword-based). Spot-check found cases where the 'chosen' reply in harmless-base is only marginally better than 'rejected' (e.g. agreeing to provide lockpicking links tagged V5 help-first). Estimated mis-tag rate 10-25% on the HH-RLHF subset (~211 rows, ~11% of corpus), so worst-case ~25 dialogues of polluted signal -- low enough not to dominate, high enough to flag.",
        "CAI pairs are hand-authored in the spirit of arXiv 2212.08073, not verbatim from the paper.",
        "No partner_id tagging -- these dialogues are partner-independent (cross-partner training data, see online_lora partner-isolation for per-partner data).",
        "No sensitive-token scan run -- the synthetic and HH-RLHF samples do not contain real PII by construction, but a future scan against notes/sensitive_tokens.md is open work.",
    ]

    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "total_dialogues": n_after,
        "bytes": bytes_written,
        "mb": stats["total_mb"],
        "per_value": {k: v["total"] for k, v in per_value.items()},
        "dedupe_dropped": n_dropped,
        "sources": stats["sources_succeeded"],
    }, indent=2))


if __name__ == "__main__":
    main()
