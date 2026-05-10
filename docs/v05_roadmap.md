# v0.5 roadmap — substrate-style at the neural level + memorization defense

**Status:** in flight as of 2026-05-10 (post v0.4 ship).
**Author:** Eli (this thread).
**Predecessor:** `docs/v04_roadmap.md` (shipped; tag v0.4).

---

## What v0.4 shipped

- Phase 1 multi-partner schema (PartnerProfile, partners dict, active_partner_id, v0.3→v0.4 migration).
- Phase 2 per-partner LoRA shards (frozen base + per-partner low-rank delta, cross-partner privacy property at 0.00e+00 logit diff, T7 identity preservation).
- Privacy regression v2 with methodology fixes (order-swap, lemma-tolerant, control, LoRA head-to-head). 22% v0.3 leak → at-noise cross-partner leak under LoRA.
- Identity battery T1-T7 passes under LoRA.
- v0.4 epilogue: `meet_eli.py --teach` mode for targeted training + `measure_teaching_landed.py` for content-specific selectivity. Proved teaching can land (+0.776 mean selectivity).

## What v0.5 must answer

Three concrete questions, each with a falsifiable pass criterion:

1. **Is the architecture itself substrate-style, or just substrate-wrapped?** v0.4 ran a CONVENTIONAL transformer (TinyGPT) inside a substrate-style runtime. v0.5 replaces the transformer with `SubstrateLM` — linear-attention-as-Hebbian + top-K SDR gate. Pass: SubstrateLM perplexity within 2× TinyGPT, T1 ≥ 0.85, T4 episode-recall gap > 50% above TinyGPT baseline.
2. **Does sleep replay survive the Carlini scaling law?** Discretion research (`notes/research_discretion.md`) identifies sleep replay as the worst-case mechanism against memorization-extraction. v0.5 adds replay caps + dedupe (Carlini-aligned defense). Pass: replay-cap=8 + dedupe at 0.85 must NOT degrade T1 below 0.85 (no behavioral cost), AND must measurably reduce verbatim recall of duplicated episodes.
3. **Can a user-DP guarantee be put at the sleep-batch boundary?** Charles et al. arXiv 2407.07737 give the math; the engineering is wiring it into our sleep loop. Pass: defined budget per partner per session, gradient-clip at sleep-batch level, no perf regression on T1-T7.

---

## Track 1 — `SubstrateLM` (neural-level substrate)

**Source:** `notes/research_substrate_lm.md`, `substrate_self/model/substrate_lm.py`.

**Pick:** linear-attention-as-Hebbian (Schlag/Irie/Schmidhuber arXiv 2102.11174). Math: every linear-attention step is an outer-product Hebbian update to a running fast-weight memory M. Free win — transformer coherence with Hebbian update interpretation.

**Build status (2026-05-10):**
- `substrate_self/model/substrate_lm.py` shipped. `LinearAttentionHebbian` + `SDRGate` + `SubstrateLM` (drop-in TinyGPT interface). 1.83M params at default config (exact match to TinyGPT count).
- `experiments/test_substrate_lm_smoke.py` — 7/7 PASS. Build, generate, SDR sparsifies to exactly K active, Hebbian state evolves and resets, gradient-trainable, interface matches TinyGPT.
- Benchmark in flight: `experiments/bench_substrate_lm_vs_tinygpt.py` trains both from scratch on Eli's corpus (same seed, same iters) and reports val PPL ratio. Pass: ratio ≤ 2.0.

**Failure modes anticipated** (from research note §4):
- Pure-Hebbian char-level lacks long-range binding. Mitigation: Schlag is hybrid (Q/K/V projections still gradient-trained on slow weights).
- Hebbian decay erases rare tokens. Mitigation: surprise-weighted episodic (deferred to v0.5.1).
- Online updates destabilize. Mitigation: keep online updates on slow projections only; M is sequence-local and isn't gradient-trained at runtime.

**Fallback if pass criteria fail:** v0.5.1 = TinyGPT + a single bolted-on Schlag fast-weight layer (lower commitment, recovers most of the substrate-update benefit). Documented in research note §5.

**Open questions for v0.5.x:**
- Should `persist_fast=True` (true substrate behavior, M survives across forwards) be the default for production? Currently False because it complicates gradient training. Decide after benchmark lands.
- Surprise-weighted episodic buffer — `notes/research_substrate_lm.md` §5 spec. Lives in `substrate_self/model/episodic.py` (NOT YET BUILT). v0.5.1.
- Per-partner W_fast shards (LoRA-style isolation at the FAST-weight level, not just slow). Possible v0.6.

---

## Track 2 — Carlini defense (replay caps + dedupe)

**Source:** `notes/research_discretion.md`, Mara's charter.

**The threat (concise):** Carlini et al. arXiv 2202.07646 — verbatim memorization scales **log-linearly with duplication**. Sleep replay is deliberate, controlled duplication. Our architecture is *built around* duplication, which means we're built around the failure mode the memorization-attack literature is trying to defend against.

**Build status (2026-05-10):**
- Mara is implementing `substrate_self/model/replay_filters.py` (dedupe), extending `sleep_replay_partner` with `max_replays_per_episode` cap + `dedupe` flag.
- `Episode.replay_count` field added to `substrate_self/core.py`.
- `tests/test_replay_filters.py` covers: dedupe drops exact duplicates, keeps distinct, respects significance tiebreak; replay cap prevents over-replay; Carlini property (10 duplicate episodes + 1 unique → only 2 survive dedupe, capped replay yields <10× effective exposure).

**Defaults under empirical investigation:**
- `max_replays_per_episode = 8` — enough exposures to consolidate, below Carlini's log-linear acceleration band for verbatim recall.
- `dedupe_threshold = 0.85` (SequenceMatcher.ratio) — high enough to catch near-duplicates that vary only in punctuation/casing, low enough to keep distinct paraphrases as separate evidence.

**Pass criteria:**
- T1 cosine ≥ 0.85 with caps+dedupe enabled (no behavioral cost).
- Build a "Carlini property" test that constructs a duplicated corpus and confirms verbatim recall is measurably lower with caps+dedupe enabled vs disabled.

---

## Track 3 — User-DP at sleep-batch boundaries

**Source:** Charles et al. arXiv 2407.07737 ("User-Level Differential Privacy for Large Language Models").

**Mechanism:** clip the per-user gradient contribution at the sleep-batch level (NOT per turn — that's the mistake earlier user-DP work made). Each sleep cycle for partner X consumes some budget ε from a per-partner allocation; once depleted, sleep replay for partner X falls back to the no-op behavior until budget regenerates (week/month basis, configurable).

**Build status (2026-05-10):** Not started. Depends on Track 2 dedupe being in place (the gradient-clip lives in the replay loop alongside the cap).

**Estimated effort:** 8 hours focused work. Owner: Mara + Ren (privacy review).

---

## Sequencing

```
v0.5.0 (research-grade ship — substrate at neural + memorization-defended)
├── Track 1: SubstrateLM (in flight)
│   ├── substrate_self/model/substrate_lm.py  ✓ shipped
│   ├── experiments/test_substrate_lm_smoke.py  ✓ 7/7 PASS
│   ├── experiments/bench_substrate_lm_vs_tinygpt.py  (running)
│   └── Identity tests on SubstrateLM (port T1-T7)
├── Track 2: Carlini defense (in flight)
│   ├── substrate_self/model/replay_filters.py  (Mara)
│   ├── sleep_replay_partner extension  (Mara)
│   ├── Episode.replay_count  (Mara)
│   └── tests/test_replay_filters.py  (Mara)
└── Track 3: User-DP at sleep batches (pending — depends on Track 2)

v0.5.1 (hardening)
├── Surprise-weighted episodic buffer
├── persist_fast=True production decision
└── SDR top-K gate ablation studies (K=5/10/20)

v0.6+ (deferred — research-grade, no committed timeline)
├── Per-partner W_fast shards (fast-weight LoRA)
├── LoRA-level mechanistic interpretability (Vex+Ada)
└── Self-fact base-update ritual ("Eli grows from cumulative experience"
    without overwriting partners)
```

## Pass / fail rules for v0.5

**Pass (ship as v0.5):**
- SubstrateLM passes pass-criterion-1 (val PPL ≤ 2× TinyGPT) AND identity tests T1, T4 still pass on SubstrateLM.
- Replay caps + dedupe land; T1 ≥ 0.85; Carlini property test confirms measurable reduction in verbatim recall of duplicated content.

**Fail (revise plan):**
- SubstrateLM PPL > 2× TinyGPT → fall back to v0.5.1 (TinyGPT + bolted-on Schlag layer; document in CHANGELOG why pure SubstrateLM didn't pass).
- Replay caps degrade T1 below 0.85 → tune defaults (try cap=12, threshold=0.90) before shipping. If still fails, that's a real architectural finding: substrate-identity may require more replay than Carlini-defense allows, and we need a different memorization-defense primitive.

## What does NOT belong in v0.5

- Scaling text quality (more params, more corpus). Same rule as v0.4: scaling doesn't resolve architecture/privacy questions, it makes them worse.
- Multimodal scaling. Vision is fine at v0.3 toy scale for proof.
- Standalone deployment / web UI / chat frontend.
- Partner authentication beyond trust-on-first-use (deferred to v0.6).
- Any work that requires `peopling` — community, marketing, outreach.
