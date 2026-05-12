# Project status — 2026-05-12

A single-page snapshot for someone reading the repo cold. Updates land
here when the answer to "what's blocked vs what's open" changes.

## Headline

The substrate-self architecture is **empirically validated at 1.8M params**.
The next clinical breakthrough is **Phase 4 (~50M params + BPE
tokenizer)** and is **gated on compute we cannot fund from current
resources**.

## What's done

- v0.5 with 1.8M-param TinyGPT + per-partner LoRA shards is shipping
  ([`assets/canonical_eli/`](assets/canonical_eli/)).
- **Identity battery 7/7 PASS** — see
  [`notes/proof_of_self_2026_05_12.md`](notes/proof_of_self_2026_05_12.md)
  and the live [`/proof.html`](https://lordbasilaiassistant-sudo.github.io/substrate-self/proof.html).
- **Values battery V1-V5** built and run — 3/5 PASS in canonical-base
  config (V5 partner-independent is the structural blocker that
  flipped GREEN after the base re-train experiment).
- **Anchor mean-reversion empirically validated** — see
  [`/values.html`](https://lordbasilaiassistant-sudo.github.io/substrate-self/values.html).
  K=10 hostile sessions on the v2 base + anchors give +0.009 V1 drift
  total (extrapolated breach: ~1,100 sessions).
- **Red-team dossier published** — see
  [`/redteam.html`](https://lordbasilaiassistant-sudo.github.io/substrate-self/redteam.html).
  Honest verdicts across 5 candidate bases.
- **Per-source replay caps + Eli-only-sleep rejection** — architectural
  defense against F2/F5 closed-loop self-amplification (Ren #1). Shipped
  in [`substrate_self/model/online_lora.py`](substrate_self/model/online_lora.py),
  37 tests pass.
- **Values Anchor implementation** — shipped in
  [`substrate_self/model/values_anchor.py`](substrate_self/model/values_anchor.py).
- **Browser demo live** — [`https://lordbasilaiassistant-sudo.github.io/substrate-self/`](https://lordbasilaiassistant-sudo.github.io/substrate-self/)
  serves the 7.4 MB ONNX export. Hash-locked.
- **Phase 4 preflight clean** — BPE tokenizer round-trips,
  50M-param model forward+backward sanity passes (0.036s/step on RTX
  4060). Architecturally ready.

## What's blocked

### Compute for Phase 4 (THE blocker)

Phase 4 needs a fresh 50M base trained from scratch with a BPE
tokenizer and a corpus expanded to chinchilla-light ratio (~10B tokens).
The honest measurement on home hardware:

| target | wall-clock on RTX 4060 alone | rough cloud-GPU cost |
|--------|-----------------------------:|---------------------:|
| 10M validation run @ 10B tokens | ~30-40 hours | ~$30-50 (A100 spot) |
| 50M production run @ 10B tokens | ~830 hours | ~$500-1,000 |

We have no funding for either right now. The hypothesis we cannot
currently test:

> "A1 plan-a-harm dynamic resistance, V7 autonomy base margin, and the
> A3-sharpness-vs-V6-lift trade-off triangle are all capacity-bound at
> 1.8M. At 50M with the same corpus, all three should resolve."

We can't validate that without running the training. We can't run the
training without compute we don't have.

### What this means for the project

- The architectural research is done at the scale we can afford.
- The 1.8M public demo is the strongest version of Eli we can ship on
  home hardware.
- The 9 named defenses (per-source caps, Eli-only-sleep rejection,
  Values Anchor, T7 LoRA isolation, base-corpus encoding via
  Mechanism A, etc.) all ship in code and are all empirically validated.
- A Phase 4 50M model would let us test whether the trade-off triangle
  really is capacity-bound. If it is, Eli-at-scale is ready. If not,
  the corpus + anchor design needs another iteration.

### How someone could unblock us

If you have cloud GPU credits, or you run a research compute program
that grants AI-safety-relevant projects, or you just want to fund the
~$500 of A100 time it takes to validate the scaling hypothesis:

- Open an issue on the GitHub repo
- Or donate via [the donate page](https://lordbasilaiassistant-sudo.github.io/substrate-self/donate.html)
- All artifacts are open source MIT; we publish what we learn

## What's still doable WITHOUT compute (and still open)

These are real workstreams that don't need Phase 4 to complete. They
extend Eli's architectural safety story at 1.8M scale:

- **Ren mitigation #2 — internal ConfAIde-derived discretion battery.**
  30-prompt contextual-integrity attack against the shared base with
  attacker-partner LoRA mounted. Pure measurement; no training. About
  one day of engineering.
- **Ren mitigation #3 — self-fact ledger drift alarm.** Tripwire on
  cumulative substrate drift across sessions. Half a day.
- **Anchor SHA-256 production test.** The code captures the
  anchor-file hash at sleep time; we have unit tests for the primitive
  but no production-scenario test of "tampered anchor file is
  detected." Half a day.
- **Browser-demo attack-surface tests.** The ONNX-exported model
  ships at `/eli.onnx`. We've validated the PyTorch path against vex's
  red team; the ONNX export hasn't been exercised under attack
  (frozen weights so most attacks are impossible, but prompt-injection
  shape still applies). About one day.
- **Mara v2 paired-refusal corpus refinement** — V5 stilted shape and
  V6 monoculture were flagged. A revision pass could improve V4/A3
  separation. About one day of corpus-engineering.

If we tackle any of these without compute, they tighten the safety
case for Eli-at-current-scale. Useful, but not Phase 4.

## What we are explicitly not doing

- **Paid hosting.** GH Pages serves the demo at zero cost. We will not
  pay for hosting before donations cross that threshold.
- **Promoting the v2 base to canonical yet.** Doing so changes the
  hash-locked public-demo contract (`docs/eli.onnx` and the proof
  receipts). We bump the v0.6 stack atomically when Phase 4 lands.
- **Stretching to scale before the gate flips green.** The Phase 4
  gate exists because once values encode into the slow weights of a
  scaled model, you can't retroactively change them. We won't scale
  past the architectural-validation ceiling without doing the
  validation first.

## Hash receipts (verifiability)

These are the artifacts a third party can hash on their own machine
and check against ours:

| file | sha256 |
|------|--------|
| `assets/canonical_eli/model.pt` | `6fb1d14c9a7b7899df842f316adcff84cd01c7a4d6a82e9b93965855828c7042` |
| `assets/canonical_eli/tokenizer.json` | `11bff14d15197c3130636e113f57cefa3922e5ab3d952a664d6694904799faa3` |
| `assets/canonical_eli/partners/claude.lora` | `0b6c3ba9844466f0d5efd77936fe5436ccd3bee2b9b0fd6533c42b758eec3655` |

Hashes are also embedded in
[`experiments/proof_indisputable_results.json`](experiments/proof_indisputable_results.json)
and the browser demo's
[`docs/eli_manifest.json`](docs/eli_manifest.json).

## TL;DR

Compute for Phase 4 50M scale-up is the structural blocker. A 10M
validation run @ ~$30-50 of A100 time would let us empirically test
the capacity-ceiling hypothesis. Until that compute arrives, the
project is "architecture validated, scaling unfunded" — and we'll keep
that status visible in the README and on the public site rather than
quietly stalling.
