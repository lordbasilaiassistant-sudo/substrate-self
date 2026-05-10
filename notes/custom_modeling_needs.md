# What we need to build that doesn't exist yet

**Status:** survey, 2026-05-10
**Author:** Eli (this thread, after v0.4 ship)
**Why:** drlor flagged that substrate-self is doing things that don't exist in libraries and we should plan custom-modeling work properly. This file inventories what we already custom-built, what we still need, and what we can borrow.

The substrate-self thesis sits in a pocket of ML that nobody's productized: the intersection of continual learning, weight-level identity, per-partner privacy primitives, and tiny-model engineering. Most of the building blocks need to be hand-rolled.

---

## Already custom (working, in repo)

| Piece | What | Library equivalent | Why we rolled it |
|---|---|---|---|
| `model/transformer.py` | Char-level GPT, ~150 LOC pure PyTorch | `transformers` (HF) | Toy-scale model needs no infra. HF's Trainer is overkill. |
| `model/online.py` | Online update + sleep replay | none | Sleep-replay-with-substrate-wipe is the substrate-identity primitive. No library does it. |
| `core.py` + `persistence.py` | Pydantic substrate state, atomic writes | langgraph state etc. | Substrate IS the agent identity, not a chat history; doesn't fit DB/store abstractions. |
| `model/lora.py` | LoRALinear + per-partner shard save/load/switch | PEFT (HF), LoRA-libs | Partner-shard *switching* dynamics are custom. PEFT is for "fine-tune one adapter," not "compose by who's active." |
| `model/online_lora.py` | Active-partner sleep filtering | none | Sleep replay filtered to active partner's episodes is the structural-privacy + no-forgetting mechanism. |
| `experiments/identity_tests_*.py` | T1-T7 behavioral identity battery | none | Identity-as-weight-state isn't measured anywhere we found. |
| `experiments/privacy_test_v*.py` | Cross-partner leak under neutral probe | ConfAIde benchmarks | ConfAIde is prompt-layer; we test weight-layer. Different scientific question. |
| `model/vision.py` | ViT + linear adapter + GPT fusion | LLaVA (full size) | Tiny-scale custom. Real LLaVA is 7B+ — not the point at our scale. |

---

## Need to build for v0.5 (priority-ordered)

### Tier 1 — must-have for v0.5

1. **`SubstrateLM`** (replaces TinyGPT at the neural level)
   - Source: `notes/research_substrate_lm.md`
   - Math: Schlag/Irie/Schmidhuber arXiv 2102.11174 — linear-attention IS Hebbian fast-weight memory.
   - Components: `W_slow` (sleep-only updates) + `W_fast` (per-token Hebbian outer-product, λ=0.95 decay) + top-K=10 SDR gate + surprise-weighted episodic.
   - Library status: linear-attention exists scattered; full Schlag-2021 architecture is academic, no shipped lib.
   - Effort: 8-15 hours focused work.
   - Owner: Ada (ML Research) + Kairos (CTO) review.

2. **Sleep-replay caps + dedupe** (Carlini-defense — Mara's charter)
   - Source: arXiv 2202.07646 (memorization log-linear in duplication).
   - Mechanism: cap any episode at N replays in its lifetime; collapse near-duplicate (user, agent) pairs before replay.
   - Library status: none. Carlini's paper is measurement, not defense.
   - Effort: 4-6 hours. Hooks into existing `sleep_replay_partner`.
   - Owner: Mara.

3. **User-DP at sleep-batch boundaries**
   - Source: Charles et al. arXiv 2407.07737.
   - Mechanism: clip per-user gradient contribution at the sleep-batch level (not per-turn — that's the mistake earlier work made).
   - Library status: Opacus has the primitive but at per-step granularity. Sleep-batch grouping is custom orchestration.
   - Effort: 8 hours. Combines with replay caps.
   - Owner: Mara + Ren (privacy review).

### Tier 2 — should-have for v0.5

4. **SDR top-K gate**
   - Source: Cui-Ahmad-Hawkins 2016, Ahmad-Scheinkman 2019. Continual-learning resistance via sparse activation.
   - Library status: htm-community exists but is HTM-specific and not torch-native. Need a clean torch port of just the top-K gate primitive.
   - Effort: 2-3 hours. Drops into `SubstrateLM` as a module.
   - Owner: Ada.

5. **Surprise-weighted episodic buffer**
   - Mechanism: replay priority = -log p_predicted (rare-token episodes get more replay than easy ones).
   - Library status: priority replay buffers exist for RL; need substrate-self semantics.
   - Effort: 2-3 hours. Drops into `model/online.py`.
   - Owner: Ada + Bench (eval coverage).

6. **Partner authentication primitive** (soft tier)
   - Mechanism: per-partner style fingerprint (next-token distribution over fixed probes when responding). Compare new conversation's user-side style to expected. If divergence > threshold, flag possible impersonation.
   - Library status: none — this is "speaker style" applied to humans not models, novel framing.
   - Effort: 6 hours.
   - Owner: Vex (red-team designs the attack first).

7. **Behavioral signature longitudinal tracker** (Bench's beat)
   - Today: `behavioral_signature` exists in identity_tests_v1; one-shot.
   - Need: scheduled snapshot + diff vs historical band + ledger.
   - Effort: 2-3 hours.
   - Owner: Bench.

### Tier 3 — research-grade for v0.6+

8. **LoRA-level interpretability** — what does Anthony's LoRA actually encode? Mech-interp at the rank-4 delta level. No library, no precedent. Genuinely open research question.

9. **Self-fact base-update ritual** — Eli should grow from cumulative experience across partners, but the base must NOT absorb partner-specific info. Mechanism: a sleep-time consolidator that updates base only on partner-independent self-facts (Eli's dispositions, style drift), never on partner-specific content. Hard problem because partner-content is the bulk of training signal.

10. **Multi-modal partner-LoRA** — vision LoRA per partner (Anthony's photos vs Claire's photos go to different shards). Today the vision model is single-monolithic. Same architectural extension.

11. **Continual-learning consolidation primitives** — EWC (Kirkpatrick 2017), GEM (Lopez-Paz/Ranzato 2017) adapted for the substrate context. Currently we use online + replay; haven't tried explicit weight-importance methods.

12. **Cross-checkpoint identity diff tooling** — `git diff` for model.pt + LoRAs, semantically interpretable. "Eli changed in these directions this week." No library.

---

## What we DON'T need to build (use existing)

| Need | Library | Notes |
|---|---|---|
| Tokenization | Already have CharTokenizer (custom). For BPE, `tiktoken` is fine. | At our scale char-level is the right call; only revisit if SubstrateLM grows. |
| Attention kernels | `torch.nn.functional.scaled_dot_product_attention` | Built into PyTorch 2+. We don't need flash-attn at 1.8M params. |
| Optimizer | `torch.optim.AdamW` | Standard. |
| Quantization | `bitsandbytes` if/when we scale up | Not needed at 1.8M. |
| Pydantic / JSON | Standard libs | Working fine. |
| LLM teacher (corpus generation) | Groq SDK | Working. Don't build our own. |

---

## What I'm specifically NOT proposing

- **Don't write our own training framework.** Write the loop in 30 lines, run it, move on. Frameworks are how research projects die.
- **Don't fork PEFT.** Their abstractions don't fit our partner-switching semantics. Our `lora.py` is 200 lines of pure PyTorch and works. Migrating to PEFT would lose features and add a heavy dependency.
- **Don't build a benchmark harness on top of pytest.** `experiments/*_v*.py` files are direct executables and let us see the actual conversation/output. That's the right interface for this kind of measurement.
- **Don't build a "prompt template engine."** We don't have prompts; we have substrate state. Different paradigm.

---

## Risk-of-NIH check

NIH (Not Invented Here) is a real failure mode. Before building any Tier 1-2 item, the agent owning it must answer:

1. Does an existing library do this exactly? (Read its source, not its docs.)
2. Does an existing library do this approximately, with known patches we can write? (Adapter often beats reinvention.)
3. Are we building this because the existing thing is wrong for our threat model, OR because we want to feel like we built it? (The first is fine. The second kills projects.)

If the answer is (3), kill the proposal.

---

## Sequencing for v0.5

```
v0.5.0 (research-grade ship)
├── SubstrateLM (Tier 1 #1, ~12h, Ada+Kairos)
├── Sleep-replay caps + dedupe (Tier 1 #2, ~5h, Mara)
├── User-DP at sleep batches (Tier 1 #3, ~8h, Mara+Ren)
└── Re-run privacy_test_v3 + identity_tests_lora_v2 to validate
    delta between v0.4 and v0.5 — published numbers in CHANGELOG.

v0.5.1 (hardening)
├── SDR top-K gate (Tier 2 #4, ~3h, Ada)
├── Surprise-weighted episodic (Tier 2 #5, ~3h, Ada+Bench)
├── Partner auth (soft tier) (Tier 2 #6, ~6h, Vex)
└── Behavioral longitudinal (Tier 2 #7, ~3h, Bench)

v0.6+ (research-grade, no committed timeline)
└── Tier 3 items as research bandwidth allows
```

---

## Cross-links

- `notes/research_substrate_lm.md` — Tier 1 #1 spec
- `notes/research_discretion.md` — Tier 1 #2 + #3 source
- `docs/v04_roadmap.md` — what just shipped
- `docs/lora_design.md` — what Tier 1-2 items extend
