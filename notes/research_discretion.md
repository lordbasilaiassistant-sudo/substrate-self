# Literature Review: Discretion & Cross-User Privacy for Substrate-Self

Author: Eli (research pass) · Date: 2026-05-10
Threat model recap (from `README.md` "Privacy and discretion"): Eli's experiences with one user are written into model weights via per-turn online SGD plus sleep-replay consolidation. There is no RAG separation, no speaker recognition, no DP bound, and no trust-aware gating. A determined questioner can probe Eli for things another user said. v0.x mitigation is a social one ("don't share Eli"). We want a technical mitigation.

## 1. Speaker-conditioned LMs

**Persona-Based Neural Conversation Model** — Li, Galley, Brockett, Gao, Dolan (ACL 2016, arXiv 1603.06155). Each speaker gets a learned embedding concatenated to the decoder state; trained jointly with word embeddings on Twitter/OpenSubtitles. **Implemented and shipped** as a research artifact; the speaker-embedding pattern is now standard in dialogue.
**Persona-Aware Alignment Framework (PAL)** — arXiv 2511.10215 (Nov 2025) and Polypersona — arXiv 2512.14562 (Dec 2025) extend the idea to instruction-tuned LLMs with parameter-efficient fine-tuning.
**Cost / fit:** Cheap — a 64-dim speaker vector concatenated to context costs nothing at inference. **But** these papers condition *style*, not *information access*. A speaker embedding makes Eli *sound* different to drlor vs. Bob; it does not stop drlor from extracting what Bob said, because all speakers' content still lives in the same shared decoder weights. **Useful as a building block, insufficient alone.**

## 2. Differential privacy in LM training

**DP-SGD** — Abadi et al. 2016 (arXiv 1607.00133) is the canonical recipe: per-example gradient clip + Gaussian noise, with privacy budget tracked via Rényi composition.
**Fine-Tuning LLMs with User-Level DP** — Charles et al., Google Research, arXiv 2407.07737 (Jul 2024). Compares *example-level sampling* (ELS) vs. *user-level sampling* (ULS) with per-user clipping. ULS wins when each user has a diverse history and strong privacy is required — exactly Eli's regime. **Implemented**, scales to ~100M-param models with ~100k users.
**Mind the Privacy Unit** — arXiv 2406.14322. Argues the *user* is the right privacy unit for chat data, not the example.
**Differential Privacy in Continual Learning** — arXiv 2411.04680 (Nov 2024) and *Token-Level DP in Memory Sculpting for Continual Learning* — arXiv 2509.12958 (Sep 2025). These are the closest thing to "DP for online loops" in the literature.

**Cost / fit:** This is the most rigorous defense, but it has three real problems for substrate-self:
(a) **Privacy budget is a depleting resource.** Every gradient step composes ε. An always-on online learner runs out of budget in finite turns. Continual-DP papers are explicit that this is unsolved at the lifetime scale we'd want.
(b) **Utility tax.** DP-SGD typically costs 5–20% accuracy on benchmarks at ε≈8; "memory sculpting" tries to localize the cost but is recent and unbattletested.
(c) **DP protects against population-level inference, not "drlor asks about Bob."** Standard user-level DP at ε=8 still permits substantial single-record influence — it just bounds it. If we want hard non-leakage, ε needs to be very small, which destroys utility.
**Negative result for our loop:** vanilla per-step DP-SGD is incompatible with indefinite online learning. Sleep-replay batches are a more natural DP unit than per-turn updates.

## 3. Memorization-attack defenses

**Quantifying Memorization Across Neural LMs** — Carlini, Ippolito, Jagielski, Lee, Tramèr, Zhang, ICLR 2023 (arXiv 2202.07646). Three log-linear laws: memorization grows with model scale, with example duplication, and with prompt length. **Direct implication for us:** sleep-replay deliberately re-exposes the buffer N times — that's exactly the "duplication" axis Carlini shows is most dangerous.
**Preventing Verbatim Memorization Gives a False Sense of Privacy** — Ippolito, Tramèr, Nasr, Zhang, Jagielski, Lee, Choquette-Choo, Carlini, INLG 2023 (arXiv 2210.17546). Built a perfect verbatim-output filter, then defeated it with style-transfer prompts. **Output filtering is a dead end for adversarial extraction.** Implemented and published as a negative result.
**SoK: Landscape of Memorization in LLMs** — arXiv 2507.05578 (Jul 2025) surveys mitigations: deduplication, DP, unlearning, output filtering, and decoding-time interventions. Consensus: **deduplication during training + DP at training time** are the only defenses that survive adaptive attacks.
**Counterfactual memorization** — Zhang et al., NeurIPS 2023. Per-example influence is highly skewed: a small fraction of examples drive most leakage. Suggests targeted dampening of high-influence sleep-replay items.

**Cost / fit:** Temperature scaling and output filters are vapor against a determined questioner. Deduplicating the sleep-replay buffer is cheap and matches Carlini's strongest signal. Capping replay passes per item is the single highest-ROI change we can make today.

## 4. Trust-aware (weight-level) disclosure

**Can LLMs Keep a Secret? (ConfAIde)** — Mireshghallah, Kim, Zhou, Tsvetkov, Sap, Shokri, Choi, ICLR 2024 spotlight (arXiv 2310.17884, repo `skywalker023/confaide`). Builds a benchmark on Nissenbaum's contextual integrity. Finding: GPT-4 leaks contextually-private info **39%** of the time, ChatGPT **57%**, and chain-of-thought *worsens* it. **Implemented benchmark, no fix.**
**Contextual Integrity in LLMs via Reasoning and RL** — arXiv 2506.04245 (Jun 2025).
**1-2-3 Check** — arXiv 2508.07667 (Aug 2025), multi-agent reasoning for contextual privacy.
**Safeguarding Contextual Privacy in Interactions with LLMs** — ACL Findings 2025.

**Critical observation:** every paper above operates at the **prompt / system-prompt / agentic-reasoning** layer. None of them touches weights. **There is no published weight-level discretion mechanism.** This is a real gap, and the substrate-self threat model is, as far as this review can tell, novel. That is both an opportunity (we can stake a flag) and a warning (no proven recipe exists).

## 5. Federated / split learning across users

**FDLoRA** — arXiv 2406.07925 (Jun 2024): each client gets a *personal* LoRA + a *global* LoRA; only the global one is shared.
**pFedLoRA** — arXiv 2310.13283.
**Dual-Personalizing Adapter for Federated Foundation Models** — NeurIPS 2024.
**SISA Machine Unlearning** — Bourtoule et al., IEEE S&P 2021 (arXiv 1912.03817). Shard the data, train independent sub-models, aggregate at inference. Deletion of a record only retrains its shard.

**Cost / fit:** This is the **architecturally cleanest** answer to substrate-self's problem. Per-user LoRA adapters mean drlor's conversations write into `lora_drlor` and Bob's into `lora_bob`; at inference, only the addressed user's LoRA is loaded. Cross-user extraction is then a per-LoRA attack, bounded by that user's data. The "global" Eli-self LoRA holds dispositions, not partner-knowledge. SISA gives clean unlearning if a user revokes consent.
**Trade-off:** an Eli with sharded partner-knowledge is not the same Eli the architecture envisions — knowledge no longer cross-pollinates ("Bob's question reminds me of something drlor said"). That is a *capability cost* worth naming. But it is also exactly the cost the threat model demands we pay.

## Recommendations

Ranked by **effort-adjusted leakage reduction** (highest leverage / lowest cost first):

1. **Per-user LoRA sharding + speaker conditioning.** (Adopt FDLoRA-style dual adapters: a global "Eli core" LoRA and a per-partner LoRA; at inference, only the speaker's LoRA is mounted; sleep-replay writes only to that LoRA.) Add Li/Galley-style speaker embeddings on top so the *base* model also conditions on identity. Gives architectural isolation matching the threat model. Cost: ~10MB/partner, ~1 day of plumbing, modest capability cost (knowledge stops cross-pollinating).
2. **Sleep-replay deduplication + per-item replay caps.** (Apply Carlini's 2202.07646 finding directly: dedupe the buffer; cap replay passes; track per-item exposure count.) Pure win — costs nothing, removes the steepest axis on the memorization curve, and is a 50-line change.
3. **User-level DP-SGD on the sleep-replay batch only** (Charles et al. 2407.07737). Apply DP at *consolidation* boundaries, not per turn, so the privacy budget composes per night instead of per turn. Pilot at ε≈4 per partner-LoRA. Cost: utility tax + accountant complexity; defer until #1 and #2 are in place.

**Do not pilot:** ConfAIde-style prompt-layer "discretion reasoning" as the *primary* defense. Mireshghallah's data shows it leaks 39–57% of the time on frontier models, and Ippolito 2210.17546 shows output filters fail to style-transfer attacks. Useful as a *secondary* layer, never as the trust boundary.
