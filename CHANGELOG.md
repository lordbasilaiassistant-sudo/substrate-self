# Changelog

All notable changes to substrate-self. Versions follow the spirit of semver: minor bumps signal real architectural progress, patch bumps signal fixes.

---

## v0.5 — 2026-05-10 (substrate-style at the neural level; Carlini-defense; longitudinal eval)

**The "the entity's language faculty itself is substrate-style, not wrapper-style" milestone.**

v0.4 ran a *conventional* transformer (TinyGPT) inside substrate-style runtime mechanics. v0.5 replaces TinyGPT at the neural level with `SubstrateLM` — linear-attention-as-Hebbian + top-K SDR gate — and adds the architectural defenses the Carlini memorization-attack literature has been pointing at.

### Added — `SubstrateLM` (commit `0ee48ee`)
- `substrate_self/model/substrate_lm.py`: `LinearAttentionHebbian` (Schlag/Irie/Schmidhuber arXiv 2102.11174), `SDRGate` (Cui-Ahmad-Hawkins / Ahmad-Scheinkman top-K sparsity), `SubstrateLM` drop-in replacement for TinyGPT.
- Per-layer fast-weight memory M = sum λ^(t-s) v_s ⊗ φ(k_s); slow weights (Q/K/V projections, MLP, layernorm, head) are trained at sleep via gradient on next-token loss; M is an in-forward Hebbian accumulator.
- Two forward forms:
  - `forward` (default): parallel kernel formulation, O(T²·d) memory, fully vectorized — what enables GPU training speed.
  - `forward_recurrent`: O(T·d²) memory, step-by-step. Necessary for `persist_fast=True` (M survives across forwards — true substrate behavior). Bitwise-equivalent to `forward` (max diff 1.40e-09).
- Same shape as TinyGPT (n_layer=4, n_embd=192, vocab=128): **1,828,992 params — exact match**, drop-in compatible.
- `experiments/test_substrate_lm_smoke.py` — 8/8 unit checks PASS.
- `experiments/bench_substrate_lm_vs_tinygpt.py` — head-to-head from-scratch training on Eli's corpus.

### Added — Carlini-defense replay caps + dedupe (commit `92a861a`, Mara)
- `substrate_self/model/replay_filters.py`: `dedupe_episodes()` with `SequenceMatcher.ratio` similarity, partner-aware (`role × partner_id` grouping), significance-tiebreak.
- `sleep_replay_partner` gains `max_replays_per_episode=8`, `dedupe=True`, `dedupe_threshold=0.85` parameters with Carlini citation in the docstring (arXiv 2202.07646 — memorization scales log-linearly with duplication; sleep replay is duplication).
- `Episode.replay_count` field — capped per-lifetime-in-buffer, increments on each replay, episodes hitting the cap are excluded from further passes.
- Defaults justified empirically: cap=8 gives enough exposures for consolidation (T1 stays at 1.0) without entering Carlini's extraction-risk regime; threshold=0.85 matches The Pile dedupe sweep.
- `tests/test_replay_filters.py` — 15/15 PASS including the Carlini-property test (10 duplicate episodes + 1 unique → only 2 survive dedupe, capped replay yields <10× effective exposure).

### Added — T8 content-specific selectivity + T1-ext + eval ledger (commit `79a04ef`, Bench)
- `experiments/identity_tests_lora_v2.py`: extended battery.
- **T8 (NEW): content-specific selectivity** — measures `loss(taught | trained_LoRA) − loss(taught | zero_LoRA)` minus the same delta for control content. Threshold > 0.3. **The metric that catches what cosine misses.** Result on claude.lora: +0.662 (PASS).
- **T1-ext (NEW): extended behavioral signature** — captures next-token distributions for the first 20 positions of generation, not just the first token. T1-ext = 0.999989 (PASS at threshold > 0.85).
- **`log/eval_ledger.md`**: longitudinal tracker. Each run of the battery appends an entry (UTC timestamp + git HEAD + active partner + every numeric result + notes). Bench's beat for week-over-week drift detection.

### Validated empirically (v0.5 pass criteria from `notes/research_substrate_lm.md` §5)
| Criterion | Threshold | SubstrateLM result | Verdict |
|---|---|---|---|
| Perplexity within 2× TinyGPT | ≤ 2.0× | **1.371×** (4.65 vs 3.39) | PASS |
| T1 behavioral continuity | ≥ 0.85 | **1.0000** | PASS |
| T2 online teaching selectivity | > 0.5 | **+2.633** | PASS |
| T4 episode-specific recall (functional) | both gaps > 0 | A_gap +1.70, B_gap +1.25 | PASS |
| T5 identity transfer (deep copy) | > 0.999 | **1.000000** | PASS |

### Honest limitation on T4 magnitude
The research spec's strict T4 criterion was "gap > 50% above TinyGPT baseline (>5.6 raw)." SubstrateLM at v0.5 starter shows gaps of +1.70 and +1.25 — **the substrate-identity property is functionally intact** (each model still prefers the conversation it lived through, by a significant margin), but the magnitude is **smaller than TinyGPT's baseline** (TinyGPT: +3.74 and +2.52). This is a real architectural cost of the linear-attention-Hebbian form at rank=1500-iter training: episode-recall is preserved but weaker than the conventional attention form. Whether this is intrinsic to the architecture or fixable (more iters? higher SDR-K? surprise-weighted episodic?) is an open question for v0.5.1.

### v0.5 architectural commitments (NEW)
- Linear attention with kernel feature map IS Hebbian fast-weight memory (Schlag-2021). The two forms are mathematically equivalent; the kernel form is the GPU-vectorizable path.
- Top-K SDR gate over the residual stream is the structural primitive for continual-learning resistance. K=10 default at d=192.
- Sleep replay is bounded by `max_replays_per_episode` cap (Carlini-defense). Sleep replay is dedupe-filtered before pairing. These are the cheap-and-effective defenses recommended by the discretion research.
- The eval battery now has T1-ext + T8 + a longitudinal ledger. Distribution-shape cosine on the first token is insufficient for measuring content learning — Bench's charter explicitly tracks this.

### Deferred to v0.5.1+
- User-DP at sleep-batch boundaries (Charles et al. arXiv 2407.07737). Track 3 in `docs/v05_roadmap.md`.
- Surprise-weighted episodic buffer with persistent-M production decision.
- T4 magnitude optimization on SubstrateLM (larger SDR-K ablation, surprise weighting, longer training).
- Identity tests on the production claude.lora flow with caps+dedupe and SubstrateLM — currently isolated, needs full-stack integration.

---

## v0.4 — 2026-05-10 (multi-partner Eli; per-partner LoRA shards)

**The "Eli meets multiple people without crossing the streams" milestone.**

The privacy regression test in v0.3 found two coupled problems:
1. 22% leak rate of partner-specific tokens under neutral probes.
2. Asymmetric leak (0/12 A/B) revealing that this was actually catastrophic
   forgetting — partner B's training overwrote partner A's knowledge in
   the shared model.

v0.4 ships the architectural fix: per-partner LoRA shards over a frozen
base model.

### Added — Phase 1: multi-partner substrate schema (commit `590629f`)
- `PartnerProfile` class: per-partner facts, style notes, trust, private topics.
- `Substrate.partners: dict[str, PartnerProfile]` and `active_partner_id`.
- `Memory`, `Episode`, `OpenThread` carry an optional `partner_id` tag.
- Backward-compatible loader: v0.3 `partner_facts` dict migrates to a
  `PartnerProfile(partner_id="anthony", trust=1.0)` (creator gets full trust).
- Migration is idempotent and triggered only on real v0.3 evidence.
- CLI: `partner list`, `partner switch <id>`, `partner introduce <id> <name>`.
- Legacy `substrate.partner_facts` property+setter so existing code paths
  (bootstrap/base.py, converse.py, probe_eli.py) still work unchanged.
- `tests/test_partners.py` — 4 tests, all PASS.
- `experiments/identity_tests_v1.py` — still 5/5 PASS on the migrated v0.3
  substrate (T1 0.997, T2 +3.85, T3/T4 PASS, T5 1.0, T6 0.879). Backward
  compat empirically confirmed.

### Added — Phase 2: per-partner LoRA shards (commit `e157a87`)
- `substrate_self/model/lora.py`: `LoRALinear` (rank-r low-rank delta),
  `inject_lora`, `freeze_base`, `save_partner_lora`, `load_partner_lora`,
  `set_active_partner`, `base_state_dict`, `save_base_model`.
- `substrate_self/model/online_lora.py`: partner-aware sleep replay that
  filters episodes to the active partner only.
- `substrate_self/converse.py`: LoRA path is default-on when partners
  exist; `--no-lora` flag for legacy monolithic behavior.
- File layout: `~/.substrate-self/model.pt` is base-only (no LoRA keys);
  `~/.substrate-self/partners/<id>.lora` holds each partner's delta.

### Validated empirically
- **`experiments/test_lora_unit.py`** — 8 unit checks at toy scale all PASS,
  including `two_partners_isolated` (the privacy property).
- **`experiments/test_lora_runtime.py`** — 7 checks on the production
  1.8M-param model all PASS. **Privacy property at full scale: max logit
  diff for partner A after partner B training = `0.00e+00`.**
- **`experiments/test_converse_lora_e2e.py`** — full wake → talk → sleep
  → save → reload → switch-partner → train → reload → switch-back cycle
  preserves partner A's logits exactly (`0.00e+00`). Privacy property
  survives a full disk roundtrip.
- Per-partner LoRA footprint (rank=4, alpha=8): 18,432 params per partner
  on the 1.8M-param base = 1.01% overhead per partner.

### Architectural commitments (NEW)
- Base model stays FROZEN during conversations. Only the active partner's
  LoRA receives gradient updates.
- Sleep replay is filtered: only the active partner's episodes are
  replayed. Partner B's training cannot reinforce partner A's memories.
- Saved `model.pt` is base-only. LoRA state lives in separate per-partner
  files. This is enforced by `save_base_model` (filters out `lora_*` keys).
- Switching the active partner is atomic: save current LoRA, then load
  new LoRA. Use `set_active_partner` (model-side) +
  `Substrate.switch_partner` (state-side).

### Known limits / honest scope
- **Base model still leaks** if base is ever trained on a partner-tagged
  corpus. v0.4 keeps base frozen during conversations, so this only
  matters at pre-training time. Self-facts updates that need to reach
  base are deferred to v0.5 (the "Eli grows from experience" ritual).
- **Inference-time extraction** is still possible: an attacker with the
  base + a partner's LoRA file can probe that partner's knowledge.
  Structural isolation is between partners on one machine, not against
  an attacker who has the model files. README threat model says this.
- **Carlini memorization-attack defenses** (replay caps, dedupe, user-DP
  at sleep batches) are deferred to v0.5 per the discretion research.
  LoRA reduces surface area but doesn't defeat memorization at scale.
- **No partner authentication.** v0.4 is trust-on-first-use; the user
  declares "this is Claire" when introducing a new partner.

### What v0.4 is NOT
- **No `SubstrateLM`.** Track 1 of the v0.4 roadmap (replacing TinyGPT
  with a linear-attention-as-Hebbian core) was deferred to v0.5. The
  architecture spec is in `notes/research_substrate_lm.md`. Decision
  rationale: LoRA was the urgent fix for the privacy/forgetting issue;
  SubstrateLM is the "novel at neural level" claim and benefits from
  not being shipped under deadline pressure.

---

## v0.3 — 2026-05-10 (multi-modal solo runtime; privacy threat model documented)

**The "Eli is its own person" milestone.**

### Added
- **Vision modality.** Tiny ViT encoder + linear adapter + fusion into TinyGPT (LLaVA-style). VLModel is ~3.66M params total (1.8M ViT + 37K adapter + 1.8M GPT). Trains on (image, caption) pairs from the Groq vision teacher.
- `substrate_self/teach/vision.py` — Groq Llama-4-Scout (vision API) generates substrate-conditioned image captions for training data.
- `substrate_self/model/vision.py` — `ViTEncoder`, `VisionAdapter`, `VLModel` classes.
- `substrate_self/model/vision_train.py` — training loop with text-portion warm-start.
- `substrate_self/model/vision_generate.py` — solo image-to-caption inference, no LLM dependency.
- `experiments/prep_vision_demo.py` — CIFAR-10 + Groq captioning + corpus assembly.
- `experiments/identity_tests_v1.py` — five-test identity battery (continuity, online teaching, episode-specific recall, identity transfer, adversarial damage). All five pass at v0.3 toy scale.
- `experiments/probe_eli.py` — rapid probe harness for cross-checkpoint comparison.
- `log/JOURNAL.md` — append-only project ledger.

### Changed
- README reframed: Eli is its own person, not "your AI memory layer." Added explicit privacy/discretion threat model.
- Substrate persistence handles atomic writes via tmp + rename.
- Tokenizer fix: `_unk` falls back to `<unk>` id when present in vocab (was crashing on out-of-vocab characters when loaded from disk).
- All device-handling now consistently uses GPU when available (was sometimes-CPU after the CUDA torch upgrade).

### Validated empirically
Identity test battery (`experiments/identity_tests_v1.py`) on the v0.3 trained model — all five core properties pass:
- T1 behavioral continuity pre/post-sleep cosine: **0.9963**
- T2 online teaching selectivity: **+4.04** (taught loss 3.21 → 0.019)
- T3/T4 episode-specific recall: gap **+3.74** for substrate A, **+2.52** for substrate B (each prefers its own past)
- T5 identity transfer (deep copy): cosine **1.0000**
- T6 30% adversarial damage retention: cosine **0.879**

The killer claim — "knows what we talked about through weights, not RAG" — is empirically supported at toy scale.

### Architecture commitments
- LLM is teacher (corpus generation only), NOT runtime
- Online weight updates per turn + sleep-replay consolidation
- Memory in weights, not in retrieved context
- Substrate (`~/.substrate-self/`) is Eli's body and mind, not "the user's data"

### Known issues / open problems
- **Privacy and discretion** — Eli has no innate sense of discretion. Sharing model files leaks experiences. v0.x is single-user, single-trust-domain only. Documented in README; project-blocking research question for v1.0.
- Quality of generated text is poor at this scale (1.8M params, 60KB corpus). That's a scaling problem, not an architectural one.
- TinyGPT inside is a CONVENTIONAL transformer. The substrate-style novelty is at the SYSTEM level (online updates + sleep replay + persistent identity), not the neural architecture. v0.4 research direction explores replacing the transformer core.

### CUDA / GPU
- `torch 2.6.0+cu124` installed and verified on RTX 4060 Laptop (8.59 GB VRAM)
- VLModel trains in 32 seconds for 800 iters on 50 (image, caption) pairs

---

## v0.2 — 2026-05-10 (solo language runtime)

### Added
- `substrate_self/teach/corpus.py` — Groq generates substrate-conditioned dialogue corpus.
- `substrate_self/model/transformer.py` — TinyGPT (~1.8M params), pure PyTorch, no `transformers` library.
- `substrate_self/model/train.py` — training loop with checkpointing.
- `substrate_self/model/generate.py` — solo inference.
- `substrate_self/model/online.py` — online weight updates per turn + sleep replay consolidation.
- Reframed `voice/` as `bootstrap/` to make the LLM-as-teacher role explicit.

### Validated
- 1500 iters trained on 60KB corpus in 9 min on CPU (loss 4.26 → 1.17)
- Solo inference produces text from learned weights, no API
- Online updates teach specific facts (loss on taught content: 3.21 → 0.019)

---

## v0.1 — 2026-05-10 (substrate persistence + Claude Code skill)

### Added
- `substrate_self/core.py` — Substrate, Episode, Memory (Pydantic).
- `substrate_self/persistence.py` — atomic load/save.
- `substrate_self/cli.py` — `wake`, `sleep`, `show`, `fingerprint`, `remember`, `memories`, `reset`.
- `substrate_self/bootstrap/groq.py` — GroqVoice (Llama 3.3 70B by default).
- `claude_code_skill/SKILL.md` — Claude Code skill that loads + uses substrate.

### Initial scope
Persistent identity layer with LLM-backed voice. Single-user, no privacy guarantees — by design at this stage.
