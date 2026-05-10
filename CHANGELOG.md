# Changelog

All notable changes to substrate-self. Versions follow the spirit of semver: minor bumps signal real architectural progress, patch bumps signal fixes.

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
