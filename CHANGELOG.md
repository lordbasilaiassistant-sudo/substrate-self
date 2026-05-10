# Changelog

All notable changes to substrate-self. Versions follow the spirit of semver: minor bumps signal real architectural progress, patch bumps signal fixes.

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
