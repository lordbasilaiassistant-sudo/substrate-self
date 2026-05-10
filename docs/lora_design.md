# Per-partner LoRA shards (v0.4 Phase 2)

**Status:** module landed (`substrate_self/model/lora.py`), unit-validated; CLI wiring pending Phase 1.
**Author:** Eli (this thread)
**Decision driver:** privacy_test_v1 found 22% leak rate AND asymmetric leak revealing catastrophic forgetting. Per-partner LoRA shards solve both by giving each partner physically distinct parameters.

---

## Architecture

```
~/.substrate-self/
├── substrate.json           # multi-partner state (Phase 1)
├── model.pt                 # FROZEN base — partner-independent
├── tokenizer.json
├── model_config.json
└── partners/
    ├── anthony.lora         # ~few thousand params, Anthony-specific delta
    ├── claire.lora
    └── ...
```

**Forward pass:** `y = base(x) + (alpha/rank) * B @ A @ x` for every wrapped Linear.
**Initial state:** `B = 0`, so a fresh partner starts with the unmodified base model — no contamination from training data. As that partner's LoRA trains, `B` accumulates partner-specific deltas.

## Decisions and the reasoning

### Where to apply LoRA — only `c_attn` and `c_proj`

TinyGPT has Linears at `c_attn` (Q/K/V), `c_proj` (attn output), `mlp[0]`, `mlp[2]`, `head`. We wrap only the attention projections.

Rationale: standard LoRA practice (Hu et al. 2021) — attention projections capture most of the per-task adaptation. MLP and head wrapping doubles LoRA-param cost for marginal gains at our scale. We can revisit if v0.4 validation shows MLP wrapping is needed for partner-specific knowledge to fit.

Per-partner footprint at the v0.3 model shape (`n_layer=4, n_embd=192`, rank=4):
- 8 LoRA wraps × (4·192 + 192·576) = 8 × (768 + 110,592)
- Wait — that's the unfreezed math. Actual rank-4 LoRA: `A ∈ R^(rank × in)` + `B ∈ R^(out × rank)`.
- For `c_attn` (in=192, out=576): 4·192 + 576·4 = 768 + 2304 = 3072.
- For `c_proj` (in=192, out=192): 4·192 + 192·4 = 768 + 768 = 1536.
- 4 layers × (3072 + 1536) = **18,432 LoRA params per partner**.
- Base 1.8M params → LoRA shard is ~1% of base. 1000 partners = +1.8M, still tractable.

(Validated empirically: the test model at `n_embd=32` reports `lora_p / base_p = 5.4%`. At full v0.3 shape this drops to ~1%.)

### Initialization — `A` kaiming, `B` zero

`B = 0` ensures the LoRA contribution is exactly zero at initialization, so:
1. A freshly-introduced partner sees the unmodified base — clean start, no biased welcome.
2. Switching to a brand-new partner is a perfectly clean reset (proven by the `initial_zero_contribution` test).
3. The partner's LoRA only diverges from zero through that partner's actual conversation — no cross-contamination at init.

### Optimizer — over `lora_parameters` only

`build_lora_optimizer(model)` builds AdamW over only LoRA params. `freeze_base(model)` sets `requires_grad=False` everywhere else. Two layers of defense: even if the optimizer accidentally got base params, base would not accumulate gradients.

### Atomic partner switch — save-then-load

`set_active_partner(model, new_id, dir, current_id=...)`:
1. Saves current partner's LoRA state to `partners/<current_id>.lora` (if `current_id` given).
2. Loads new partner's LoRA from `partners/<new_id>.lora` (or resets to init if file absent — fresh partner).

The save uses tmp-then-rename for atomicity, matching the substrate.json persistence pattern.

### Sleep replay — active partner's episodes only

`sleep_replay_partner(...)` filters `substrate.episodic` to entries where `ep.partner_id == substrate.active_partner_id`. Episodes from other partners are not replayed during this partner's sleep — so partner B's training cannot reinforce partner A's memories.

This dovetails with the privacy mechanism: structural isolation is preserved by the replay path, not just the forward path.

Legacy escape hatch: if `active_partner_id is None` (unmigrated substrate) or `ep.partner_id is None` (legacy episode), the episode is included — we don't silently drop data on first sleep after upgrade.

## What this DOES NOT solve

1. **Base model leak.** LoRA shards still share the base. If base trains on any partner-mixing data, base would gain cross-partner info. v0.4 keeps base completely frozen during conversations; base is only updated during pre-training (corpus generation, before any partner exists). This is a hard line.

2. **Inference-time extraction.** A determined attacker with the base model and one partner's LoRA can extract that partner's knowledge by probing the composed model. v0.4 is "structural isolation between partners on the same machine," not "cryptographic discretion against an attacker who has the model file." The README threat model says this explicitly.

3. **Self-facts and dispositions** are still in the base (no partner has a private copy). If Eli learns something about itself through a conversation — "I find topology interesting" — that goes in the base eventually. Phase 1 schema design says: base updates are slow and infrequent (every N sessions, not every wake). Mechanism for that ritual is Phase 2.5 (deferred to v0.5 if v0.4 ships clean).

4. **Carlini memorization-attack defense.** Sleep replay still duplicates training data. Per-partner LoRA reduces the surface area (partner B's LoRA only memorizes B's data) but the log-linear duplication scaling still applies within a single partner's LoRA. Mitigations (replay caps, dedupe) are in the v0.5 backlog per the discretion research.

## Test coverage

`experiments/test_lora_unit.py` (passes):
- `injection_count`: 4 LoRALinears injected at 2-layer test model (c_attn + c_proj × 2 layers).
- `initial_zero_contribution`: B=0 init means injection does NOT change forward outputs.
- `freeze_base`: only LoRA params have `requires_grad=True`.
- `lora_actually_trains`: 50 steps reduce loss (LoRA is expressive enough to learn).
- `extract_apply_roundtrip`: extract → reset → apply gives identical logits.
- `save_load_disk_roundtrip`: same, through disk file.
- `two_partners_isolated`: train A, switch to B, train B, switch back to A → A's logits are bitwise identical to before B touched the model. **This is the privacy property.**

Missing test coverage (next):
- Identity tests (T1–T6) on a LoRA-injected v0.3 model — must still pass at LoRA-rank=4.
- Privacy regression test v2 with order-swap and lemma-tolerant matching.
- Catastrophic forgetting symmetry: train A, train B, train A again (back to A's LoRA), measure A-recall — must equal A-recall right after first A-training.

## Open questions for v0.4 validation phase

1. **Does rank=4 hold enough capacity for real per-partner facts at v0.3 model size?** Test: train Anthony's LoRA on a corpus of ~200 facts, check recall.
2. **Does `(alpha/rank) = 2` blow up gradients?** The standard formula. We may need to tune lr.
3. **How big does the LoRA get if we ALSO wrap MLP layers?** Easy ablation: change `target_names` to include `mlp` modules.
4. **First-introduction protocol:** when `partner introduce <id> <name>` is called, the LoRA file doesn't exist — does the new partner need a "seed" with their declared facts before conversation, or is conversation enough?

## Cross-links

- Module: `substrate_self/model/lora.py`
- Runtime: `substrate_self/model/online_lora.py`
- Unit checks: `experiments/test_lora_unit.py`
- Multi-partner schema: `docs/multi_partner_design.md`
- Discretion research: `notes/research_discretion.md`
- v0.4 roadmap: `docs/v04_roadmap.md`
- Privacy baseline: `experiments/privacy_test_v1_results.json` (22% leak, asymmetric — the thing this design solves)
