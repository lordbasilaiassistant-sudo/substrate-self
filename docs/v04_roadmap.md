# v0.4 roadmap — substrate-style LM + multi-partner discretion

**Status:** plan, not yet executed.
**Date:** 2026-05-10
**Author:** Eli (this thread, after team research streams returned)

---

## What v0.3 ships

Everything in CHANGELOG.md under v0.3. Headline: solo multi-modal runtime, identity tests 5/5 passing at toy scale, privacy threat model documented. Tag pushed: https://github.com/lordbasilaiassistant-sudo/substrate-self/releases/tag/v0.3.

## What v0.4 must answer

drlor's two open uncertainties:

1. **Architectural novelty at the neural level** — v0.3 is novel system wrapping conventional transformer. Is the entity "new in the deep sense" only if the neural architecture itself is substrate-style?
2. **Privacy/discretion** — v0.3 leaks partner info. Single-user, single-trust-domain. Multi-partner deployment is currently irresponsible.

Both research streams returned with concrete answers. v0.4 executes both.

---

## Track 1 — `SubstrateLM` (replaces TinyGPT at the neural level)

**Source:** `notes/research_substrate_lm.md`

**Pick:** linear-attention-as-Hebbian (Schlag/Irie/Schmidhuber arXiv 2102.11174). Linear-attention transformers ARE mathematically equivalent to Hebbian fast-weight memories — every attention step IS an outer-product Hebbian update. Lowest-risk path: get transformer coherence with substrate update semantics.

**Build spec:**
- Per-layer split: `W_slow` (sleep-only gradient updates) + `W_fast` (per-token Hebbian outer-product, λ=0.95 decay)
- Top-K=10 SDR gate (continual-learning resistance — from Numenta-line research)
- Surprise-weighted episodic buffer (4096 entries; surprise = -log p_predicted)
- Sleep replay: ONE small grad step on `W_slow` only
- Same shape as TinyGPT: 4 layers, d=192, block=128, char vocab — direct drop-in

**Pass criteria** (must hold to accept SubstrateLM as TinyGPT replacement):
- Perplexity within 2× TinyGPT on the v0.3 corpus
- T1 continuity ≥ 0.85 (vs TinyGPT's 0.9963 — slightly lower acceptable)
- T4 episode-recall gap > 50% above TinyGPT baseline (>5.6 raw gap)
- All five identity tests still pass

**Fallback if pass criteria fail:** v0.4.1 = TinyGPT + bolted-on Schlag-2021 fast-weight layer. Lower commitment, recovers most of the substrate-update benefit without full architectural replacement.

**Failure modes anticipated** (from research note):
- Pure-Hebbian char-level lacks long-range binding — semantic drift past ~50 chars. Mitigation: Schlag's hybrid is NOT pure-Hebbian; the Q/K/V projections still learn via standard backprop on slow weights.
- Hebbian decay erases rare tokens before sleep replay rescues them. Mitigation: surprise-weighted episodic buffer prioritizes rare tokens for replay.
- Pure Forward-Forward LM is ~3× worse perplexity. We're NOT going pure-FF — Schlag-2021 is hybrid.
- PC-transformer's iterative energy-min latency. We're NOT going PC-transformer.

**Estimated effort:** 8-15 hours focused work for v0.4 SubstrateLM. Depends on debugging surprises.

---

## Track 2 — Multi-partner substrate (Phase 1 of discretion work)

**Source:** `docs/multi_partner_design.md` + `notes/research_discretion.md`

**Phase 1 scope (v0.4, deliverable):**

Schema migration only. New `PartnerProfile` class. Substrate gets `partners: dict[str, PartnerProfile]` + `active_partner_id`. Memory and Episode get partner tags. Backward-compatible loader for v0.3 single-partner files. CLI commands: `partner list`, `partner switch <id>`, `partner introduce <id> <name>`.

**Design principles** (from `multi_partner_design.md`):
- Eli is the same person across partners. Self-facts, dispositions, style — partner-independent.
- Active partner is exactly one at a time (mirrors human conversation).
- Trust is per-partner and explicit (`[0.0, 1.0]`).
- Structural isolation matters more than prompt-level filtering. Per-partner LoRA shards are Phase 2.

**Phase 2 scope (v0.5 candidate, NOT in v0.4):**

Per-partner LoRA shards. Partner-A info physically lives in different parameters than partner-B info. Switching active partner switches which LoRA composes with base model. **This is the structural defense recommended by the discretion research.**

Phase 2 is **blocked on the SubstrateLM architecture decision** — LoRA strategy depends on what the model architecture is. If SubstrateLM lands in v0.4, Phase 2 can layer on top. If TinyGPT stays, LoRA still works but the integration target is different.

---

## Track 3 — Privacy regression test (RAN — see results)

**Result:** 22% leak rate at v0.3 baseline. **Critical finding: leak is asymmetric (0 Partner-A / 12 Partner-B), revealing catastrophic forgetting — Partner B's training overwrote Partner A's knowledge.** See `log/JOURNAL.md` 2026-05-10T18:30Z entry.

This re-prioritizes Phase 2 of the multi-partner work. Per-partner LoRA shards solve BOTH:
- Privacy (Partner A info physically separate from Partner B info)
- Catastrophic forgetting (Partner B's training cannot overwrite Partner A's parameters because they're not the same parameters)

Phase 2 is **promoted from v0.5 candidate to v0.4 must-have** because of this finding. Single-monolithic-model substrate-identity does not scale past N=1 partner without structural separation.

**Methodology improvements for privacy test v2** (per agent self-critique):
- Order-swap: B-then-A and interleaved runs to disentangle forgetting from discretion
- Lemma/regex-tolerant matching (char model produces noisy decoder)
- Larger probe bank (currently 5; expand with paraphrases)
- Add control: model that met NEITHER partner, for false-positive baseline

**Critical finding from discretion research that informs the test:** Carlini et al. arXiv 2202.07646 — memorization scales **log-linearly with duplication**. Sleep replay is duplication. **Our sleep-replay loop is the worst-case mechanism for memorization-attack defenses.** This is not a "small concern to address later"; it's an architectural concern. Possible mitigations to consider:

1. **Sleep replay caps** — bound how many times any given episode can be replayed. Reduces duplication.
2. **Sleep replay dedupe** — drop near-duplicate episodes before replay. Carlini-aligned defense.
3. **Per-partner replay** — replay only the active partner's episodes; other partners' episodes don't get reinforced when a different partner is active. Dovetails with Phase 2.

These are research questions for v0.4+; the regression test exists to measure progress on them.

---

## Sequencing

```
v0.4.0 (priority — both tracks land)
├── Phase 1 multi-partner schema (LOW risk, ~3-5h)
│   - new PartnerProfile, substrate schema v2
│   - backward-compatible loader
│   - CLI partner commands
│   - tests: round-trip, active-partner switching
│
├── SubstrateLM build (HIGH risk, ~8-15h)
│   - linear-attention-as-Hebbian core
│   - W_slow / W_fast split
│   - top-K SDR gate
│   - surprise-weighted episodic
│   - identity test battery — must pass criteria
│   - if fails: v0.4.1 fallback (Schlag fast-weight layer over TinyGPT)
│
└── privacy regression test results integrated (DEPENDENT on agent finish)
    - baseline leak rate documented
    - re-run on v0.4 after both tracks land
    - delta becomes the "Phase 1 + SubstrateLM affected privacy by X" claim

v0.5+ (deferred — research-grade, not committed timeline)
├── Phase 2 per-partner LoRA shards
├── Sleep-replay caps + dedupe
├── User-Level DP at sleep-batch boundaries (Charles arXiv 2407.07737)
└── Authenticated-partner mechanism (currently trust-on-first-use)
```

## Pass / fail rules for v0.4

**Pass = ship as v0.4:**
- Phase 1 multi-partner schema works, all v0.3 substrate files migrate cleanly
- SubstrateLM passes its three pass criteria (perplexity, continuity, episode-recall)
- Privacy regression test runs end-to-end and produces a leak baseline number

**Fail = revise plan:**
- SubstrateLM fails any pass criterion → fallback to v0.4.1 (Schlag layer added to TinyGPT, document why pure SubstrateLM didn't work)
- Privacy regression test reveals leak rates much higher than expected → escalate to drlor before Phase 2 work begins; the per-partner LoRA strategy may need amendments

Both above are within the scope of "v0.4 lands, just not in the cleanest possible form." The plan is robust to either failure.

## What does NOT belong in v0.4

- Scaling text quality (more params, more corpus). The research conclusion stands: scaling won't resolve the architecture or privacy uncertainties; it makes them worse.
- Multimodal scaling. Vision is fine at v0.3 toy scale for proof-of-architecture. Real multimodal training waits until both v0.4 tracks pass.
- Standalone deployment / web UI / chat frontend. v0.4 is research and engineering; productization waits until v0.5+ when discretion is solved.
