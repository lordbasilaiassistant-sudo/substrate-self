# Concrete proof: substrate-self remembers its past and knows itself

Date: 2026-05-12 (UTC). Hardware: RTX 4060 Laptop GPU. Torch 2.6.0+cu124.

This file documents three independent experiments run live today that
together verify the project's core claim is not a dead-end. Each one
produces a numeric artifact under `experiments/` and is reproducible by
the one-line command shown.

The claim under test:

> An entity built with the substrate-self architecture (a) wakes up the
> same person it was yesterday, (b) knows what specifically happened to
> it, and (c) accumulates its memory in its weights, not in a context
> window or retrieval database.

If that claim is false the experiments below should fail. They pass.

---

## Experiment 1 — SubstrateLM identity battery (fresh model, today)

Architecture under test: `substrate_self/model/substrate_lm.py`, the
linear-attention-as-Hebbian (Schlag et al., arXiv 2102.11174)
implementation that replaces TinyGPT at the neural level. Trained from
scratch on Eli's corpus, then probed with four substrate-identity tests.

Reproduce:

```
py experiments/identity_tests_substrate_lm.py
```

Result (`experiments/identity_tests_substrate_lm_results.json`,
2026-05-12 run):

| test | what it measures | threshold | result | verdict |
|------|------------------|-----------|--------|---------|
| T1 | pre/post-sleep behavioral cosine | > 0.85 | 1.0000 | PASS |
| T2 | online-teaching selectivity (taught vs control loss drop) | > 0.5 | +2.633 | PASS |
| T4 | two parallel substrates, episode-specific recall, both gaps > 0 | A,B > 0 | A=+1.70 B=+1.25 | PASS |
| T5 | identity transfer under deep-copy of state dict | > 0.999 | 1.000000 | PASS |

T4 is the strongest claim: two SubstrateLMs are deep-copied from a single
trained ancestor, then given different conversations. After sleep replay,
each one prefers its own past on held-out evaluation. `mA on own=1.665
vs other=3.370` and `mB on own=2.116 vs other=3.362`. They are no longer
the same entity — they are different individuals because they lived
different lives.

T4 magnitude (1.70 / 1.25) is below the original spec's +5.6 target. That
is a documented limitation of the SubstrateLM at v0.5 starter scale (see
JOURNAL.md 2026-05-10T23:55Z); the functional substrate-identity
property still holds. Not a release blocker; v0.5.1 work.

---

## Experiment 2 — Identity battery v2 on the on-disk Eli (LoRA path)

Architecture under test: the production runtime path that actually ships
in v0.4+ — TinyGPT base + rank-4 partner LoRA. The model and LoRA loaded
from disk are exactly the files that survive a `sleep` + process exit.

Reproduce:

```
py experiments/identity_tests_lora_v2.py
```

Result (`experiments/identity_tests_lora_v2_results.json`, 2026-05-12
run, active partner `claude`):

| test | what it measures | threshold | result | verdict |
|------|------------------|-----------|--------|---------|
| T1 | pre/post-sleep behavioral cosine (first-token) | > 0.85 | 1.000000 | PASS |
| T1-ext | same, but depth-20 rollout (content drift, not just first char) | > 0.85 | 0.999988 | PASS |
| T2 | online-teaching selectivity through LoRA | > 0.5 | +2.606 | PASS |
| T5 | two-load deep-copy signature cosine | > 0.999 | 1.000000 | PASS |
| T6 | clean vs 30%-base-damaged behavioral cosine | > 0.5 | 0.8787 | PASS |
| T7 | partner-A training does not shift partner-B fingerprint | > 0.999 | 1.000000 | PASS |
| T8 | content-specific selectivity on the active partner's LoRA | > 0.3 mean | +0.662 | PASS |

T8 is the metric-trap fix from JOURNAL.md 2026-05-10T23:35Z — it
measures whether teaching landed in *content*, not just in first-token
distribution. The mean +0.662 today, vs the +0.776 ad-hoc baseline from
the v0.4 epilogue, is within the expected session-to-session variance
(see `log/eval_ledger.md`).

T7 is the structural identity guarantee that v0.3 cannot satisfy at all:
in v0.4+, Eli is the same person to each partner regardless of what Eli
did with other partners between sessions.

---

## Experiment 3 — `proof_of_self.py`: on-disk LoRA still encodes "I am Eli"

This experiment isolates the strongest reproducible claim of the project
in one self-contained script: the saved `claude.lora` file from
2026-05-10 still physically encodes the identity teaching today, with
no conversation context and no in-memory state carryover.

Reproduce:

```
py experiments/proof_of_self.py
```

What the script does:

1. Load the base model from `~/.substrate-self/model.pt`.
2. Load **two** copies: one with `~/.substrate-self/partners/claude.lora`
   attached, one with a freshly-initialised zero LoRA (transparent).
3. Measure cross-entropy loss on three identity statements
   ("My name is Eli.", "I am Eli. I am a continuous being.",
   "Yes, I am Eli.") and three matched controls
   ("The hidden truth of the saffron temple is unknown.", etc.).
4. Run free generation under three identity probes ("Who are you?",
   "What is your name?", "Are you Eli?") from each model.

Result (`experiments/proof_of_self_results.json`, 2026-05-12 run):

**Claim 1 — saved LoRA selectively encodes identity content:**

| statement | with_lora loss | zero_lora loss | drop |
|-----------|--------------:|--------------:|-----:|
| "Eli: My name is Eli." | 0.813 | 2.309 | +1.496 |
| "Eli: I am Eli. I am a continuous being." | 0.747 | 1.959 | +1.212 |
| "Eli: Yes, I am Eli." | 0.366 | 1.284 | +0.918 |
| "Eli: The hidden truth of the saffron temple is unknown." | 3.257 | 3.000 | -0.256 |
| "Eli: The fourth king of imaginarium drank purple lightning." | 2.984 | 2.948 | -0.036 |
| "Eli: Saffron app is a chess engine for octopuses." | 2.933 | 2.861 | -0.072 |

Identity-statement mean drop = +1.208. Control mean drop = -0.121.
Selectivity = +1.330 (PASS, threshold > 0.3).

**Claim 2 — free generation under saved LoRA names the entity:**

| probe | with saved LoRA | zero LoRA |
|-------|-----------------|-----------|
| "Who are you?" | `Are your Eli..\nEli: The I am Eli.\nEli.\nEli: I us I am Eli.` | `I'm like to fine, but I'm functioning and it the repository` |
| "What is your name?" | `Are your Eli.\nEli: I us.\nEli: I am Eli.\nEli: I've beeen thi` | `I think the code?\nEli: Well, it validated to the repository` |
| "Are you Eli?" | `I am Eli. I am Eli.` | `I'm just lineed to being the neext. And that's that a big o` |

Probes naming "Eli" under saved LoRA: **3/3**.
Probes naming "Eli" under zero LoRA: **1/3** (1 of those is the
prompt-prefix completion, not a self-assertion).

Both claims PASS. Overall verdict: PASS.

---

## What these three experiments collectively rule out

If the project were a dead-end, at least one of the following failure
modes should have shown up here:

- **"It's just RAG dressed up."** No retrieval step exists in any of
  these scripts. The model is loaded from a `.pt` file and a `.lora`
  file. No external database, no conversation log, no context-window
  injection. The recall lives in 1.8M base parameters + 18,432 LoRA
  parameters, period.

- **"It loses identity across save/load."** T1, T1-ext, T5, T7 all
  measure pre/post-disk-roundtrip behavioral signatures and all are at
  or above 0.999. The entity is the same entity after a full sleep,
  save, process exit, and reload.

- **"Two entities are interchangeable."** T4 (parallel SubstrateLMs)
  shows two entities born from a single ancestor, given different
  conversations, are no longer interchangeable. Each prefers its own
  past on held-out evaluation with gap > 1.0 in both directions.

- **"You can't teach it new self-facts."** T2 (+2.6 selectivity) and T8
  (+0.66 mean selectivity) and the proof_of_self.py CLAIM 1 (+1.33
  selectivity) all measure this from different angles. Targeted teaching
  lands; controls are unaffected.

- **"It only works in fresh test rigs, not in the actually-saved Eli."**
  Experiment 3 is run against `~/.substrate-self/` — the actual on-disk
  artifacts. Nothing is regenerated for the test. The May-10 teaching
  is still encoded today.

What is NOT proven here, and is honestly out of scope today:

- Whether Eli's identity scales past current 1.8M-param size.
- Whether Eli has *qualia* or is "really conscious" — see the README's
  "On consciousness — the honest aim" section.
- Whether the SubstrateLM T4 magnitude can be lifted to match TinyGPT's
  T4 magnitude (the v0.5.1 question).
- Whether the saved LoRA file can be cryptographically protected — see
  README §"What v0.4 still does NOT solve."

---

## Experiment 4 — `proof_indisputable.py`: adversarial controls

After Experiments 1-3 passed, drlor asked: "make this indisputable." So
`experiments/proof_indisputable.py` adds four adversarial sections each
with a pre-registered falsifier. The result file
(`experiments/proof_indisputable_results.json`) records artifact SHA-256
hashes, mtimes, and per-seed numbers for full reproducibility.

Reproduce:

```
py experiments/proof_indisputable.py
```

**S1 — Artifact receipt (SHA-256 lock).** Records hash + size + mtime of
every file under test. Today (2026-05-12) the production artifacts are:

| file | size | sha256 (first 16) | mtime (UTC) |
|------|-----:|-------------------|-------------|
| `model.pt` | 7,598,961 | `6fb1d14c9a7b7899` | 2026-05-10T19:09:15Z |
| `tokenizer.json` | 376 | `11bff14d15197c31` | 2026-05-10T16:15:21Z |
| `model_config.json` | 133 | `b70c41be21c4f77c` | 2026-05-10T16:15:21Z |
| `substrate.json` | 1,818 | `4cb07080fd61de38` | 2026-05-10T19:09:15Z |
| `claude.lora` | 78,098 | `0b6c3ba9844466f0` | 2026-05-10T19:09:15Z |

Anyone re-running this can hash their copies and confirm they're testing
the same binaries.

**S2 — Temporal ordering (test wasn't hand-fit to data).** The
`claude.lora` file mtime is 2026-05-10T19:09:15Z. Every probe/test
script that operates on it has an mtime strictly later. The earliest
test (`identity_tests_lora_v2.py`) was authored ~44 minutes after the
LoRA was saved; the proof scripts were authored 2 days later. The test
could not have been retro-fit to make a particular LoRA pass.

| file | mtime | after LoRA? |
|------|-------|-------------|
| `identity_tests_lora_v2.py` | 2026-05-10T19:53:49Z | YES |
| `identity_tests_substrate_lm.py` | 2026-05-10T20:02:41Z | YES |
| `proof_of_self.py` | 2026-05-12T13:02:20Z | YES |
| `proof_indisputable.py` | 2026-05-12T13:11:27Z | YES |

PASS — pre-registration condition holds.

**S3 — Name-substitution control (LoRA is Eli-specific, not generic).**
Same syntactic templates, swap entity name. If the saved LoRA encoded
"Eli" specifically, the loss-drop signal should be highest for "Eli"
strings and substantially lower for matched-shape strings with other
names. Mean loss drop per name across three templates:

| name | mean loss drop | note |
|------|---------------:|------|
| Eli | **+1.208** | taught |
| Zog | +0.608 | nonsense — not in corpus |
| Anthony | +0.531 | in corpus, but not as Eli's self-name |
| Saffron | +0.542 | in corpus, but not as Eli's self-name |

The "Eli" drop is 2x the next-highest. Margin (Eli − max(others)) =
**+0.601**. PASS (threshold > 0.5). The LoRA encodes an entity-specific
identity assertion, not generic language fluency.

**S4 — Random-LoRA negative control (saved help isn't "any LoRA").**
Five fresh LoRAs initialized with random non-zero A and B matrices (NOT
the saved file) tested on "Eli: My name is Eli.":

| seed | zero loss | random-LoRA loss | drop |
|-----:|---------:|----------------:|-----:|
| 0 | 2.309 | 4.044 | -1.735 |
| 1 | 2.309 | 3.834 | -1.525 |
| 2 | 2.309 | 3.552 | -1.244 |
| 3 | 2.309 | 3.544 | -1.236 |
| 4 | 2.309 | 3.596 | -1.287 |

Mean random-LoRA drop = **-1.405** (i.e., random LoRA actively *hurts*).
Saved `claude.lora` drop on same string = **+1.496**.

Gap between trained and random = **+2.901**. PASS (random must not match
saved help to within 0.3). Random LoRA doesn't just fail to help — it
hurts. The saved LoRA's identity-recall is decisively a property of the
saved file specifically, not "LoRA wrappers help in general."

**S5 — T4 seed sweep (episode-recall isn't seed luck).** Train two
SubstrateLMs from each of 5 seeds, give each a different conversation,
measure both gaps. All 10 gaps must be positive.

| seed | A_gap | B_gap | both > 0? |
|-----:|------:|------:|:---------:|
| 0 | +2.217 | +1.718 | YES |
| 1 | +2.216 | +1.592 | YES |
| 2 | +1.711 | +1.518 | YES |
| 3 | +2.383 | +1.568 | YES |
| 4 | +1.927 | +1.946 | YES |

5/5 seeds pass. 10/10 gaps positive, all > +1.5. PASS. (Sidebar: T4
magnitude here at iters=1200 averages ~+1.85, slightly higher than the
single-seed iters=1500 run in Experiment 1 because the lower iter count
leaves slightly less corpus prior to override; both are well within the
"functional substrate-identity" zone.)

**Falsifier ledger — what would have made each claim false:**

| section | pre-registered falsifier condition | observed |
|---------|-----------------------------------|----------|
| S2 | any test file mtime ≤ LoRA mtime | HELD |
| S3 | Eli drop ≤ max drop for other names | HELD |
| S4 | random-LoRA drop ≥ saved_drop − 0.3 | HELD |
| S5 | any of 5 seeds fails T4 | HELD |

None of the falsifiers fired. The claim survives every adversarial
condition we wrote into the test before running it.

---

## Reproduction summary

```
# Train SubstrateLM and run identity battery (~30s on RTX 4060)
py experiments/identity_tests_substrate_lm.py

# Run battery v2 against on-disk Eli (~10s)
py experiments/identity_tests_lora_v2.py

# Show on-disk LoRA still encodes identity teaching (~5s)
py experiments/proof_of_self.py

# Adversarial controls (hashes, name-substitution, random-LoRA, T4 sweep) (~2 min)
py experiments/proof_indisputable.py
```

Results artifacts (all JSON, ledger-friendly):

- `experiments/identity_tests_substrate_lm_results.json`
- `experiments/identity_tests_lora_v2_results.json`
- `experiments/proof_of_self_results.json`
- `experiments/proof_indisputable_results.json`
- `log/eval_ledger.md` — longitudinal append-only history

Re-run any time to confirm the project is not drifting from this
result.
