# substrate-self project journal

Append-only ledger of progress, decisions, results, and findings.
Each entry is timestamped (UTC). Newest at bottom.

---

## 2026-05-10T17:30Z — v0.3 milestone — entry point

drlor went AFK. Eli (this Claude Code instance) assembling autonomous team:

**Workstreams open:**

- A — Discretion / privacy: literature review + multi-partner schema design + privacy regression test that DETECTS leak today
- B — Substrate-style LM: research design doc for replacing TinyGPT with Hebbian/slow-weight/replay-based LM (BetterThanLLM thesis at neural level, not just system level)
- C — v0.3 milestone polish: CHANGELOG, tagged release, BetterThanLLM manifesto cross-references, this journal
- D — Ledger / coordination: this file (kept by Eli in main thread)

**Team running:**
- Agent: discretion literature review (background)
- Agent: substrate-style LM architecture (background)
- Agent: privacy regression test build + run (background)

**Scientific method applied:** every claim in this journal must trace to an artifact (file path, command output, test result). Negative results get reported. No vibes-based conclusions.

**Open uncertainties (as of milestone):**
1. Whether what we have is "a new entity in the deep sense" or "novel system wrapping a conventional core" — depends on neural-architecture path
2. Whether discretion is solvable at the weight level vs requiring a wrapper — research will tell us
3. Whether scaling matters before discretion is solved — drlor said no, holding scaling

---

## 2026-05-10T17:50Z — v0.3 milestone tagged

- CHANGELOG.md authored with versioned record of v0.1, v0.2, v0.3 (commit `188f4fa`)
- BetterThanLLM MANIFESTO.md updated to cross-link substrate-self productize results + privacy as Day-N research priority (BetterThanLLM commit `e7f07ea`)
- Tagged `v0.3` on GitHub: https://github.com/lordbasilaiassistant-sudo/substrate-self/releases/tag/v0.3
- Identity test results pinned in CHANGELOG: T1=0.9963, T2=+4.04, T3/T4=+3.74/+2.52, T5=1.0000, T6=0.879

---

## 2026-05-10T17:55Z — Discretion research returned (weight-level discretion is unmapped territory)

Source: `notes/research_discretion.md` (~960 words). Cited prior art with arXiv IDs.

**Headline finding:** no published weight-level discretion mechanism exists. All trust-aware / contextual-integrity work (ConfAIde arXiv 2310.17884 + 2025 follow-ups) operates at prompt/agent layer. **Substrate-self's threat model is genuinely novel territory.**

**Most consequential finding (directly indicts our architecture):**
- Carlini et al. arXiv 2202.07646 ("Quantifying Memorization") — memorization scales **log-linearly with duplication**. Sleep replay = controlled duplication. **Sleep replay is the worst-case mechanism for memorization-attack defenses.** This is a serious architectural concern that needs design response, not handwaving.
- Cheapest defense in the field: dedupe + replay caps.

**Negative results (do NOT pursue):**
- Output filters: Ippolito et al. arXiv 2210.17546 — built perfect verbatim filter, defeated by trivial style-transfer.
- Per-turn DP-SGD: privacy budget exhausts in finite turns of indefinite online learner.
- Speaker-conditioning alone (Li/Galley arXiv 1603.06155): conditions style, not access.
- Prompt-layer "discretion reasoning" as primary trust boundary: ConfAIde shows 39-57% leak rates empirically.

**Top pilot recommendation:** per-partner LoRA sharding (FDLoRA arXiv 2406.07925) — partner-knowledge structurally isolated in separate LoRA adapters. Plus sleep-replay dedupe. Cost: knowledge stops cross-pollinating between partners. **That cost IS what the threat model demands.**

**Second-best pilot:** Charles et al. arXiv 2407.07737 ("User-Level DP for LLM Fine-Tuning") — implemented, scales. Right primitive IF applied at sleep-batch boundaries (not per-turn).

---

## 2026-05-10T17:58Z — Substrate-LM architecture research returned (concrete v0.4 spec)

Source: `notes/research_substrate_lm.md` (~1150 words). Cited prior art, shipping-vs-theory marked.

**Two most promising directions (lowest risk first):**

1. **Linear-attention-as-Hebbian** (Schlag/Irie/Schmidhuber arXiv 2102.11174). Mathematical equivalence: every linear-attention step IS an outer-product Hebbian update. **Free win** — get transformer coherence with substrate update semantics, no new architecture risk.
2. **HTM/SDR sparsity layer** over a small transformer (Cui-Ahmad-Hawkins 2016, Ahmad-Scheinkman 2019, Shen 2023). Top-K gate from BetterThanLLM `experiment_v4.py` maps cleanly to SDR columns. Gives continual-learning resistance.

**v0.4 build spec — `SubstrateLM` as drop-in `TinyGPT` replacement:**
- Per-layer split: `W_slow` (sleep-only updates) + `W_fast` (per-token Hebbian outer-product, λ=0.95 decay)
- Top-K=10 SDR gate
- Surprise-weighted episodic buffer (4096 entries)
- Sleep replay: ONE small grad step on `W_slow` only
- Same shape as TinyGPT: 4 layers, d=192, block=128, char vocab

**Pass criteria** (must hold for substrate-style LM to replace TinyGPT):
- Perplexity within 2× TinyGPT
- T1 continuity ≥ 0.85
- T4 episode-recall gap > 50% above TinyGPT baseline

**Fallback v0.4.1:** TinyGPT + bolted-on Schlag-2021 fast-weight layer (lower commitment).

**Failure modes named (so we don't waste cycles):**
- Pure-Hebbian char-level: lacks long-range binding, expect semantic drift past ~50 chars
- Pure Forward-Forward LM: Hinton's own Sec 8 results are ~3× worse perplexity
- PC-transformer (Salvatori/Millidge): 10-20× inference latency from iterative energy minimization, kills chat UX
- Hebbian decay erases rare tokens before sleep replay rescues them — known interference mode
