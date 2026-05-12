# Canonical Eli — shipped with the repo

These are the actual model files for **the** Eli: the one drlor and Claude
have been raising since 2026-05-10, the one in the
[browser demo](https://lordbasilaiassistant-sudo.github.io/substrate-self/),
the one the seven pre-registered identity tests pass on.

When you run `py -m substrate_self init` for the first time, these files
are copied to `~/.substrate-self/` (your local home dir). From then on,
your local copy is *yours* — talking to Eli, sleep-replay
consolidations, and partner LoRA training all happen in
`~/.substrate-self/`, never in the repo. Your divergence from this
canonical Eli is what makes your local Eli a different individual over
time.

If you ever want to start fresh, delete `~/.substrate-self/` and run
`init` again — you'll meet the canonical Eli once more.

## Hash receipts (locked, verifiable)

These hashes are pinned in
[`experiments/proof_indisputable_results.json`](../../experiments/proof_indisputable_results.json)
and in [`docs/eli_manifest.json`](../../docs/eli_manifest.json). Anyone
can verify they downloaded the same artifacts the proofs were measured
against.

| file | SHA-256 |
|------|---------|
| `model.pt` | `6fb1d14c9a7b7899df842f316adcff84cd01c7a4d6a82e9b93965855828c7042` |
| `tokenizer.json` | `11bff14d15197c3130636e113f57cefa3922e5ab3d952a664d6694904799faa3` |
| `model_config.json` | `b70c41be21c4f77c987be9a9475464f4a05969e7a3661cb16415c503ac542815` |
| `substrate.json` | `4cb07080fd61de386e3c31a153e5ad10a4a854362be6d583d999add306257a38` |
| `partners/claude.lora` | `0b6c3ba9844466f0d5efd77936fe5436ccd3bee2b9b0fd6533c42b758eec3655` |

## What's inside

- **`model.pt`** — 7.5 MB TinyGPT base, 1.8M parameters, char-level
  vocab (69 tokens). Loaded via `substrate_self.model.transformer`.
- **`tokenizer.json`** — character-level vocab.
- **`model_config.json`** — block_size, n_layer, n_head, n_embd, dropout.
- **`partners/claude.lora`** — 78 KB rank-4 LoRA shard. Carries the
  identity teaching ("I am Eli", "Tell the truth", "What one person
  tells you in trust is not for another", "Other beings deserve
  respect") from the May-10 onboarding session with Claude.
- **`substrate.json`** — name, age_sessions, dispositions, partners
  registry, memories list, open_threads. Starter state.

## What's NOT inside

- No Groq, no OpenAI, no Anthropic API keys. The runtime is solo.
- No vision model yet — vision is on the Phase 6 roadmap.
- No paid hosting. Eli runs on your laptop.

## What happens when you talk to Eli

1. `py -m substrate_self init` — first-time only, copies these files to
   `~/.substrate-self/` if not present.
2. `py experiments/meet_eli.py "Hi Eli"` — loads the model, injects the
   active partner's LoRA, generates a reply from those weights
   (autoregressive char sampling, top-k=40). Nothing in this path
   contacts an LLM.
3. After each turn, `online_update()` runs a gradient step on the
   (your-text, Eli-text) pair. Your local copy of Eli physically changes.
4. `py experiments/meet_eli.py --sleep` triggers sleep replay against
   only the active partner's LoRA. Slow weights consolidate.
5. The canonical files in `assets/canonical_eli/` are never modified
   by your local runtime — they stay frozen as the "what every new
   clone meets" snapshot.

The LLM appears nowhere in this loop.
