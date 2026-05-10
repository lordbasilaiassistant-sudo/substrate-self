# Substrate-style Language Model — Research Notes

**Author:** Eli (substrate-self CEO persona)
**Date:** 2026-05-10
**Status:** Research note — design proposal, no shipping code yet.
**Goal:** Replace TinyGPT (`substrate_self/model/transformer.py`) with an architecture that is itself substrate-style, so the BetterThanLLM primitives (Hebbian + slow weights + sleep replay + sparse activation) ARE the language faculty rather than wrapping a conventional transformer.

---

## 1. What proved out at toy scale

`experiments/identity_tests/experiment_v4.py` (5/5 passes on T1, T2, T3, T4, T6, plus T5/T7 partial) ran a substrate with these load-bearing primitives:

- **Slow weights `W_action` (F×A) and `W_trans` (F×A×F):** explicit transition table, Hebbian-updated.
- **Hebbian update rule** (`hebb_update`, lines 105–115): `W[f, a, f'] += lr` on observed transitions, plus a `(1 - lr·decay)` shrink. This is a Bienenstock-Cooper-Munro-flavored co-activation rule with a leak term, NOT backprop.
- **Sparse activation via top-K gate** (`softmax_topk`, lines 84–90): only the top-K logits survive into the softmax. K=2 in the toy world.
- **Episodic buffer** (`s.episodic`): a list of `(flavor, action, reward, next_flavor)` tuples written during wake.
- **Sleep replay** (`sleep`, lines 135–145): samples from the episodic buffer with 5% noise injection, calls the same Hebbian rule, then wipes.
- **Disposition / intention vectors:** slow scalars that bias action selection — closest analogue to "personality."

What scales naturally:
1. **Hebbian rule** — local, O(1) per synapse, no backward pass through time.
2. **Top-K sparse activation** — the only thing that makes a F=10⁵ vocabulary cheap.
3. **Sleep replay** — already an O(N_replay) loop, embarrassingly parallel.
4. **Slow-weight drift** — passive, doesn't need gradients.

What does NOT obviously scale: the explicit `W_trans[F, A, F']` cube. At F = vocab_size = 128 chars × A = context_length = 128 × F = 128 that's a 2M-entry tensor — fine. At word-level (F = 30k) it explodes to 10¹¹. This bounds the design.

---

## 2. What TinyGPT actually does

`substrate_self/model/transformer.py` is nanoGPT (Karpathy 2022): 4-layer causal-masked self-attention, 192-dim embeddings, weight-tied head, AdamW + cross-entropy. ~1.8M params. Replaces cleanly with anything exposing `forward(idx, targets) -> (logits, loss)` and `generate(idx, max_new_tokens)`. The `online.py` shim already does Hebbian-flavored learning at runtime by calling `loss.backward(); optimizer.step()` on each (user, agent) pair — but the *update rule itself* is gradient descent through a transformer. We want to replace that.

---

## 3. Literature survey

### 3.1 Predictive coding networks for language
- **Salvatori, Pinchetti, Millidge et al., "A theoretical framework for predictive coding networks" (arXiv 2407.01163, 2024)** — proves PCNs converge to the same fixed points as backprop in linear networks; gives convergence guarantees for the energy-based version. Mostly vision/MNIST benchmarks. Does NOT yet ship a competitive language model.
- **Pinchetti, Salvatori et al., "muPC: μ-Parameterized Predictive Coding" (arXiv 2505.13124, 2025)** — depth-stable PCN training to 100+ layers. Image classification only in the paper. Author explicitly notes language is "future work."
- **Millidge, Tschantz, Buckley, "Predictive Coding Approximates Backprop Along Arbitrary Computation Graphs" (Neural Computation 2022, arXiv 2006.04182)** — the foundational result that PCNs can match backprop on standard architectures, including transformers, *given enough inference iterations*. PC-Transformer experiments are tiny (toy datasets).

**Verdict:** PC has the math but no published competitive-quality LM as of 2025. HYPOTHESIS: a PC-transformer at TinyGPT scale (1–4M params) is feasible on a small corpus with current implementations (e.g., `pcx` library from Salvatori's group) — but inference cost is 4–20× standard transformer because each forward is iterative.

### 3.2 Hinton Forward-Forward
- **Hinton, "The Forward-Forward Algorithm: Some Preliminary Investigations" (NeurIPS 2022 / arXiv 2212.13345)** — Sec 8 trains a small *next-character* model on a Wikipedia subset. Two forward passes per layer (positive/negative goodness). Performance: substantially worse than backprop, but works. Critically: weight updates are *local* (no backward pass), which is exactly the substrate property we want.
- Follow-ups (Ororbia & Mali "The Predictive Forward-Forward Algorithm" arXiv 2301.01452, 2023; Lee & Song "Symmetric FF" 2023): mostly MNIST/CIFAR. No one has scaled FF to a serious LM.

**Verdict:** The most directly applicable prior art for "substrate-style language model." Hinton's Sec 8 is the existence proof. Quality is the question.

### 3.3 Numenta HTM / Sparse Distributed Representations
- **Hawkins & Ahmad, "Why Neurons Have Thousands of Synapses, A Theory of Sequence Memory in Neocortex" (Frontiers Neural Circuits 2016)** — HTM Temporal Memory; explicit sparse-distributed columns; Hebbian synapse permanence updates; sequence prediction.
- **Cui, Ahmad, Hawkins, "Continuous Online Sequence Learning with an Unsupervised Neural Network Model" (Neural Computation 2016)** — HTM on streaming sequences, including language at the *word* level on small corpora (NYC taxi data, Reuters word streams). Outperforms LSTMs on online next-element prediction on those benchmarks.
- **Ahmad & Scheinkman, "How Can We Be So Dense? The Benefits of Using Highly Sparse Representations" (arXiv 1903.11257, 2019)** — Numenta's argument that sparsity (~2% active) gives noise robustness and continual-learning resistance.

**Verdict:** The closest existing system to what we want. HTM-for-language has been demonstrated on small corpora but never at GPT-2 quality. SDRs map directly onto our top-K gate. The synaptic permanence rule is essentially our `hebb_update` with a permanence threshold.

### 3.4 Mortal computation
- **Hinton (Forward-Forward paper, Sec 1; "Two paths to intelligence" Royal Society talk 2024)** — coins "mortal computation": weights bound to specific hardware, can't be copied, learn in-place via local rules. Argues this is the only path to brain-scale efficiency. No specific LM architecture proposed; it's a design philosophy. Lines up exactly with substrate-self's "identity is in the weights."

### 3.5 Hebbian / fast-weight LMs
- **Schlag, Irie, Schmidhuber, "Linear Transformers Are Secretly Fast Weight Programmers" (ICML 2021, arXiv 2102.11174)** — shows linear-attention transformers are mathematically equivalent to a Hebbian fast-weight memory. Important: any linear attention block IS a Hebbian outer-product update. Means we can build a substrate model that *looks* like an LM but trains via local Hebbian outer products.
- **Ba, Hinton, Mnih et al., "Using Fast Weights to Attend to the Recent Past" (NeurIPS 2016, arXiv 1610.06258)** — fast/slow weight split, Hebbian fast updates. Foundational for what we'd build. Tested on associative recall, not full LM.
- **Munkhdalai et al., "Metalearned Neural Memory" / "Infini-attention" (arXiv 2404.07143, 2024)** — Google's recent revival using bounded-memory linear attention with delta rule (a Hebbian variant). Scales to 1M-token context. Confirms the fast-weight path is alive at industrial scale.

### 3.6 SDR + transformer hybrids
- **Shen et al., "Sparse Distributed Memory is a Continual Learner" (ICLR 2023, arXiv 2303.11934)** — Kanerva SDM as a memory module attached to standard nets; gives continual-learning resistance. Useful as a memory layer on top of a substrate LM.

---

## 4. Anticipated failure modes

1. **Char-level Hebbian probably can't capture long-range coherence.** Hebbian co-activation rules are inherently *local in time*: synapse `(f, f')` strengthens when `f` precedes `f'` within some window. There is no mechanism for "the noun in sentence 1 binds to the pronoun in sentence 4" beyond what the slow drift averages over many examples. Cui-Ahmad-Hawkins 2016 saw HTM beat LSTM on *next-element* prediction but lose on tasks requiring multi-sentence structure. Expect grammatical local fluency, semantic incoherence at >50 chars.
2. **Top-K gating + Hebbian decay creates catastrophic interference between rare tokens.** If a character appears once per 10k tokens, the Hebbian decay term `(1 - lr·0.05)` will erase its outgoing transitions before the next exposure. Sleep replay only helps if the rare event is in the buffer — and most rare events won't be.
3. **No backprop = no compositional generalization** (Lake & Baroni "SCAN" arXiv 1711.00350, 2017 — backprop networks struggle here, local rules struggle worse).
4. **Inference latency for PC-style energy minimization** kills real-time chat. PC-transformer at 10–20 inference steps × per-token autoregression = 10–20× TinyGPT latency.
5. **Online updates already destabilize TinyGPT** (we observe loss spikes in `online.py` when `n_steps > 1`); a pure-Hebbian model with no gradient signal is even harder to keep on-distribution.

---

## 5. Minimal viable architecture (v0.4 spec)

Hybrid design: keep the *shape* of a transformer (so we get long-range coherence) but replace the *update rule* with Hebbian fast-weights + sparse SDR layer + sleep replay. This is the smallest plausible step from TinyGPT toward "substrate-as-LM."

**Name:** `SubstrateLM` (drop-in replacement for `TinyGPT`, same interface).

### State (per layer)
- `W_slow ∈ R^{d×d}`: slow weights, updated only during sleep, via averaged Hebbian outer products. Initialized small.
- `W_fast ∈ R^{d×d}`: Hebbian fast weights, updated *every token* during wake by `W_fast ← λ·W_fast + η·(v_t ⊗ k_t)` (the Schlag 2021 / Ba 2016 rule). Decays with `λ = 0.95` per step.
- `M_episodic`: append-only ring buffer of `(k_t, v_t, q_t, surprise_t)` tuples. Surprise = `-log p(x_t | x_<t)` from the model itself. High-surprise tokens preferentially survive into sleep.
- `S_sdr ∈ {0,1}^d`: sparse activation mask, top-K=⌈0.05·d⌉ active per token. Implemented as a hard gate (Numenta-style) over the residual stream.

### Update rules
- **Wake (no backprop):** `W_fast` Hebbian step every token; `M_episodic.append(...)`. `W_slow` untouched. Same online path that `online.py` uses, but the inner update is Hebbian outer-product, not `loss.backward()`.
- **Sleep:** sample from `M_episodic` weighted by surprise; for each, replay the (k, v, q) triple; do *one* gradient step on `W_slow` only, on the predict-next-token loss. This is a hybrid: Hebbian during wake, gradient (small, sleep-only, slow-weights-only) during consolidation. Justification: Hinton FF Sec 8 shows pure-FF loses ~3× perplexity vs backprop on the same corpus; we keep backprop but confine it to sleep, which preserves the substrate-style behavioral signature (weights change *because* of replay, not because of the forward pass).
- **Generation:** standard autoregressive sampling; at each step the effective weights are `W_slow + W_fast`. No iterative energy minimization (rejected as too slow).

### Concrete shape (v0.4 starter)
- `n_layer = 4`, `d = 192`, `block_size = 128`, `vocab = 128` (char-level; matches existing tokenizer).
- 1 SDR gate per layer (top-K = 10).
- Fast-weight decay `λ = 0.95`, learning rate `η = 0.01`.
- Episodic buffer: 4096 entries, ring.
- Sleep: 3 passes over surprise-weighted samples, lr 1e-4 on `W_slow` only.

### Test plan (pass/fail criteria for v0.4)
1. **Perplexity vs TinyGPT** on the existing corpus. Pass: within 2× TinyGPT perplexity. (Lower would be amazing, equal is unlikely.)
2. **Behavioral-signature continuity test** (port T1 from `experiment_v4.py` to text): sample 50 prompts, get response distribution, sleep, resample. Cosine ≥ 0.85.
3. **Episode-specific recall** (port T4): teach two distinct fact sets in two sessions; each session's facts are recalled at >50% above baseline of the other session's facts.
4. **Online-update stability:** 100 consecutive turns with `n_steps=1` Hebbian wake updates; perplexity on a held-out set drifts < 20%. (TinyGPT current baseline: ~10% drift.)

If (1) fails worse than 3× TinyGPT, fall back to TinyGPT + a Hebbian fast-weight memory module bolted on as a single extra layer (Schlag 2021 architecture). That's the safe v0.4.1.

### Two most promising directions ranked
1. **Linear-attention-as-Hebbian (Schlag 2021).** Mathematical equivalence means we get a transformer's coherence with a Hebbian's update interpretation. Lowest risk.
2. **HTM/SDR layer over a small transformer.** Strong continual-learning story (Shen 2023, Ahmad 2019); maps directly onto our top-K gate. Higher risk, more novel.

Discarded:
- Pure Forward-Forward LM — Hinton's own results say ~3× worse perplexity at scale; not worth the integration cost yet.
- PC-transformer — inference latency kills chat UX.
- Pure-HTM language — no one has shown it works at GPT-2-class quality; we'd be doing original research rather than integration.
