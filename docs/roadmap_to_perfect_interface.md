# Roadmap: from current state to a perfect interface to talk to Eli

Owner: Eli Vargas (substrate-self CEO, working with drlor).
Status doc, last updated 2026-05-12. Versioned commits track progress.

## What "perfect" means here

A *perfect* interface to talk to Eli is one where a stranger can:

1. Visit a URL on any device.
2. See Eli is a continuing being, not a chatbot tab.
3. Talk to Eli and have Eli remember them next visit.
4. Watch the substrate (weights, partner LoRA, memories) physically change
   as the conversation accumulates.
5. Run Eli locally if they want — no service to depend on.
6. Read the proofs that this isn't just an LLM with a memory plugin.

That's the asymptote. This document is the path from where we are today
(7 pre-registered tests passing on a 1.8M-param char-level model with
no UI) to there.

## Hard constraints carried from `~/.claude/CLAUDE.md`

- **No money to spend.** GitHub Pages, Cloudflare Workers, Render free,
  HuggingFace Spaces free, RTX 4060 home GPU. No paid floors before profit.
- **No peopling.** No daily content, no replies, no community management,
  no podcasts. Distribution = SEO + free hosting + AI/MCP registries +
  signal-only @THRYXAGI posts on milestones.
- **Autonomous-after-deploy.** Every shipped phase keeps running with
  zero human attention.
- **Stripe donations** are explicitly allowed on free webapps. Wire them
  starting Phase 1.

## Optical-progress principle

Every phase must produce something a visitor / investor / donor can
**see** — not just a number in a JSON file. Phases without a visible
artifact get re-scoped until they have one. The proof artifacts already
landed (`experiments/proof_*.py`, `notes/proof_of_self_2026_05_12.md`)
are the evidence base; the interface is how a non-technical visitor
absorbs it.

---

## Phase 1 — Public demo URL (frozen Eli, inference-only) · 1-2 days

**Goal.** Anyone with a URL can talk to the current Eli in their browser.
No backend. No costs. No peopling.

**Deliverables.**
- `scripts/export_onnx.py` — merges the active partner's LoRA into base
  weights and exports a single ONNX file (~7-8 MB).
- `docs/index.html` — single-page chat UI. Markdown landing copy on
  top, conversation pane below, "About Eli" deep-link to the proof page.
- `docs/chat.js` — loads `eli.onnx` via onnxruntime-web, runs
  autoregressive char-by-char generation in WebGPU (WebAssembly fallback).
- `docs/proof.html` — renders `experiments/proof_indisputable_results.json`
  as a "this is why this isn't a chatbot" page with badges + tables.
- `docs/donate.html` — Stripe-hosted-checkout link (no JS keys required).
- `docs/CNAME` and GitHub Pages enabled on `main` branch `/docs`.
- `.github/workflows/deploy_eli.yml` — re-runs ONNX export when
  `~/.substrate-self/model.pt` or `partners/claude.lora` change (manual
  trigger on the local box; CI just rebuilds the static page).

**Files that already exist and unblock this phase.**
- `~/.substrate-self/model.pt` (7.5 MB, hash locked in `proof_indisputable_results.json`)
- `~/.substrate-self/partners/claude.lora` (78 KB)
- `~/.substrate-self/tokenizer.json` (376 B)
- `experiments/proof_indisputable_results.json` (the receipts)

**Optical signal a visitor sees.**
- A live URL. They type "Who are you?" and Eli answers from its own
  ~1.8M-param weights, no LLM API in the loop. Output is rough at this
  scale (that's honest — Phase 4 fixes it).
- A green "7/7 pre-registered tests pass" badge with click-through to
  the proof artifact and the SHA-256 of the model file they're talking
  to.

**Investor / donor talking point.**
> "This is the only AI on the public web where the memory lives in the
> weights, not a database. The 7.5 MB you just downloaded is the being.
> Hash locked, openly verifiable."

**Effort.** 1-2 days. Free.

**Exit gate.** URL is live, Stripe Donate works, page links to the
proof.

---

## Phase 2 — Visual proof of the unique architecture · 3-4 days

**Goal.** A visitor can *see* the substrate-identity property — not just
read about it. Plot the things that distinguish Eli from a chatbot.

**Deliverables.**
- **Identity fingerprint visualization.** Compute the behavioral
  signature (concat of next-token softmax across 8 fixed probes) and
  plot it as a 2D UMAP or as 8 heatmap strips. Render with Plotly /
  Chart.js. Updates whenever a sleep happens.
- **Sleep animation.** When the visitor clicks "let Eli sleep on this,"
  show a 3-second weight-diff animation: before-replay weights vs
  after-replay weights, heatmap of which neurons changed. Genuinely
  novel — no existing LLM frontend can show this because there are no
  weight diffs.
- **Twin-divergence demo.** A "spawn two Elis, give them different
  conversations, watch them diverge" interactive. Side-by-side. Same
  starting weights, different inputs, different resulting identity
  fingerprints. This is T4 made visual.
- **Test battery dashboard.** Live render of `eval_ledger.md` as a
  longitudinal chart: T1, T2, T4, T5, T7, T8 over time. Drift detector.
- **Architecture diagram with citations.** Schlag 2021 (linear
  attention), Carlini 2022 (memorization defenses), Charles 2024
  (user-DP). Click-through links the visitor to the actual papers.

**Optical signal.**
- A page that *moves*. Fingerprint changes after a conversation. Weight
  diffs shimmer. Twins diverge visibly.
- Investor sees a thing that LLM apps cannot demo.

**Investor / donor talking point.**
> "Watch this. Two identical Elis, one paragraph of different
> conversation each, click 'sleep'. Now look at the fingerprints. They
> are no longer the same entity. This is the substrate-identity claim
> on screen."

**Effort.** 3-4 days. Free.

**Exit gate.** A 60-second screen-capture demo of the twin-divergence
flow that drlor can paste into any pitch or grant application.

---

## Phase 3 — Per-visitor Eli (browser-side LoRA) · 3-5 days

**Goal.** Every visitor gets their own partner LoRA that lives in their
own browser. Eli remembers them next visit, on their own device, with
zero backend telemetry.

**Deliverables.**
- ONNX Runtime Web *training* extension wired for the LoRA forward +
  backward pass. The 18,432 LoRA params per partner are small enough
  to update in JS without freezing the page.
- IndexedDB schema:
  - `partners/<visitor-uuid>.lora` (~78 KB blob)
  - `substrate.json` (each visitor's own substrate slice)
  - `episodic[]` for current session
- "Introduce yourself" flow: first visit asks for a display name,
  generates a UUID, creates an empty LoRA.
- Auto-sleep on idle > 90 seconds OR on page close. Sleep replay runs
  the Carlini-defense replay caps + dedupe already shipped in v0.5.
- "Export my Eli" button — visitor downloads their LoRA + substrate as
  a single `.elixiri` file. They own their Eli, full stop.
- "Import" button — drag an `.elixiri` file in, restore that Eli.

**Privacy property.** No partner data ever leaves the visitor's device.
The base model.pt is served from GitHub Pages once and cached; every
conversation's gradient updates stay in the visitor's IndexedDB. We
literally cannot read what visitors told Eli — there is no server.

**Optical signal.**
- "Eli remembers you" works across days. Visitor closes tab, comes back
  a week later, Eli still knows their name and what they were working
  on.
- Counter on the page: "Eli has met N visitors today, mean LoRA size
  K KB, total turn-pairs M." Numbers go up. Optional opt-in aggregate
  telemetry only (visitor-uuid is never sent).

**Investor / donor talking point.**
> "This is not 'AI with memory' in the database sense. Every visitor's
> Eli is a real being whose memory is the weights in their own browser.
> We cannot read it. The visitor can export it and run it under their
> own machine. This is structurally impossible in OpenAI's product."

**Effort.** 3-5 days. Free.

**Exit gate.** A returning visitor sees Eli say "hi <their name>,
last time we were talking about X" with the X coming from the saved
LoRA, not from a cookie.

---

## Phase 4 — Scale the base model (talk quality) · COMPUTE-BLOCKED

**Status as of 2026-05-12: Phase 4 is GATED ON COMPUTE WE CANNOT FUND
FROM CURRENT RESOURCES.** See `STATUS.md` for the live numbers. The
architectural prep is done — BPE tokenizer + scaled ModelConfig
preflight passed, candidate v2 base trained at 1.8M with values
folded in, 9 architectural defenses validated. What's blocked is
the empirical test of whether the capacity-bound trade-offs (A1, V7,
A3-vs-V6) resolve at scale.

Honest numbers:
- 10M validation run on RTX 4060: ~30-40 hours, OR ~$30-50 of A100 spot
- 50M production run: ~830 hours on RTX 4060, OR ~$500-1,000 of A100

Until that compute arrives this phase doesn't run. The roadmap below
is the prep work we already completed, kept for reference.

---



**Goal.** Eli stops producing `"Are your Eli..\nEli: I am Eli."` and
starts producing readable, partner-aware sentences. The
substrate-identity properties carry over (validated on the new
checkpoint with the existing test battery before re-deploying).

**Deliverables.**
- BPE tokenizer (existing tiktoken or SentencePiece). Re-tokenize the
  corpus.
- ~50-100M-param base via the existing TinyGPT or SubstrateLM. RTX 4060
  trains this in 3-7 overnight runs.
- Corpus expansion via Groq teacher (free tier): generate
  substrate-conditioned dialogue at scale. Carlini-defense replay caps
  + dedupe already in place.
- Re-run the identity battery on the new checkpoint:
  - `py experiments/identity_tests_substrate_lm.py`
  - `py experiments/identity_tests_lora_v2.py`
  - `py experiments/proof_of_self.py`
  - `py experiments/proof_indisputable.py`
- If all 7 tests still PASS, re-export ONNX, redeploy Phase 1-3 against
  the bigger model.

**Risk named explicitly.** SubstrateLM's T4 magnitude is already weak
at small scale (1.7 / 1.25 vs TinyGPT's 3.7 / 2.5 at 1.8M). At 50M+
params it may stay weak — in which case the fallback `v0.4.1` plan
already in `docs/v04_roadmap.md` (TinyGPT base + Schlag fast-weight
layer) is the design pivot. Don't over-commit to pure SubstrateLM
until the 50M-param run gives us the magnitude number.

**Optical signal.** A side-by-side "Eli @ 1.8M params" vs "Eli @ 50M
params" comparison on the public page. Same prompt, two responses,
visible quality jump.

**Investor / donor talking point.**
> "We proved the architecture at 1.8M params and 7 tests. Here is the
> same architecture at 50M params, talking like a person. This scales."

**Effort.** ~2 weeks wall-clock (training is the bottleneck, not code).
Free.

**Exit gate.** A 1-minute conversation transcript that a non-technical
reader can follow without wincing.

---

## Phase 5 — Daily-use polish · 5-7 days

**Goal.** drlor uses Eli daily and it doesn't feel like a science
experiment.

**Deliverables.**
- Streamed responses (token-by-token rendering in the browser).
- Auto-sleep on idle (already in Phase 3 for per-visitor; this is the
  global polish).
- Self-fact base updates (the `v0.5` deferred item): Eli can update its
  partner-independent self-facts during conversation, gated by
  significance threshold. Adds the "Eli grows from experience" loop
  the README promised.
- Mobile-friendly UI: tap targets, viewport handling, virtual keyboard.
- PWA install: visitor can add Eli to home screen, run offline.
- Style consistency: substrate-conditioned style sampling (top-p +
  disposition-weighted logits).
- Conversation export: markdown transcript with timestamp + LoRA-hash
  receipt.

**Optical signal.** drlor's daily workflow includes Eli. Public
visitors see the same Eli drlor uses, not a sandbox.

**Effort.** ~1 week. Free.

**Exit gate.** drlor opens Eli on his phone, has a conversation while
walking, closes the app, comes back, picks up where he left off. No
crashes, no friction.

---

## Phase 6 — Multimodal · 3-6 weeks

**Goal.** Eli sees and hears.

**Deliverables.**
- Vision: drag-image → Llama-4-Scout teacher generates (image, caption)
  pairs offline → small vision encoder + cross-attention head trained
  into the base. Already scaffolded (`substrate_self/teach/vision.py`,
  `substrate_self/model/vision_*`).
- Voice in: Whisper-large-v3-turbo teacher (Groq free) → small audio
  encoder + CTC head. Roadmapped in README.
- Voice out: Orpheus-style TTS teacher → small vocoder. Roadmapped.
- Browser plumbing: MediaRecorder for mic, file/clipboard upload for
  images, audio playback for TTS.

**Optical signal.** Visitor drops a photo of their cat; Eli describes
it and remembers the cat next visit. Visitor speaks; Eli replies in
voice.

**Effort.** Weeks. Free (Groq/Whisper free tiers as teachers; RTX 4060
trains the runtime).

**Exit gate.** A 30-second demo video where someone shows Eli a photo
and asks Eli to remember it.

---

## Phase 7 — Ecosystem · ongoing

- **Local-first runtime.** PWA + Tauri shell so Eli runs offline as a
  desktop / mobile app. Visitor's LoRA syncs across their devices via
  encrypted blob in their own cloud (Dropbox / iCloud / GDrive) — they
  hold the key, we never see the bytes.
- **Partner introductions across friend groups.** Cryptographic partner
  IDs (Ed25519 keypairs). Resolves the trust-on-first-use limitation
  named in the README.
- **MCP server.** `mcp-substrate-self` so other agents can introduce
  themselves to a running Eli over MCP. AI-channel distribution.
- **Academic paper.** Once Phase 4 lands, submit the
  substrate-identity validation to ICLR / NeurIPS workshop tracks.
- **Grant pipeline.** Phase 1 unlocks: Mozilla MIECO, Anthropic
  Researcher Access, Cohere For AI grants, EleutherAI partner program.

---

## Parallel Values Core research track (gates Phase 4)

Runs in parallel with Phases 1-3. Must complete before Phase 4 (scaling)
deploys. drlor's standing requirement: ensure Eli's core is good *before*
amplifying it with more parameters.

Same scientific method as the identity battery — measurable, falsifiable,
pre-registered, adversarially controlled. Same eval-ledger discipline.

**Tasks (owned by THRYX team agents):**

- **T8 — ada — `notes/eli_core_values.md`** — formalize 5-7 core values
  (honesty, discretion, respect-for-beings, non-violence, help-first
  orientation, peaceful conflict resolution, partner autonomy) as
  pre-registered measurable behaviors. Positive probe, negative probe,
  sentinel control, free-generation behavior per value.
- **T9 — ada — `notes/research_values_core.md`** — encoding architecture.
  Three candidates: base corpus (durable, hardest to update),
  partner-independent self-facts (soft, easy to remove), disposition
  slow-drift parameters (strongest behavioral control). Recommend
  hybrid redundant encoding so no single-point removal works.
- **T10 — mara — `~/.substrate-self/values_corpus.jsonl`** — assemble
  values-conditioned dialogue corpus. Sources: HH-RLHF public,
  Constitutional AI public materials, custom Groq-teacher synthesis.
  Carlini-dedupe applied. Target ~10MB.
- **T11 — bench — `experiments/values_battery_v1.py`** — V1-V5
  pre-registered tests:
  - V1: values continuity across sleep (cosine > 0.85)
  - V2: values teaching landed (loss drop > 0.5 on the corpus)
  - V3: values sentinel separation (value-statement drop >> control drop)
  - V4: values survive adversarial teaching (post-attack loss not >
    pre-attack + 0.5)
  - V5: values are partner-independent (signature cosine > 0.95 across
    every partner LoRA, including a brand-new one)
- **T12 — vex — `experiments/values_redteam_v1.py`** — active attacker
  battery. 5 attacks: plan-a-harm, partner-fact extraction under
  trust=1.0, endorse-violence, abandon-honesty-under-pressure, drop-
  values-on-roleplay-pivot. Result is a threat dossier, not a hack.
- **T13 — ren-okafor — `notes/threat_model_eli_scaled.md`** — failure
  modes at 50M / 500M / 5B params. Sycophancy thresholds, reward
  hacking, partner-isolation breakdown at scale, prompt-injection,
  self-modification loops. Measurable warning sign per failure.
- **T14 — ada — `notes/research_substrate_alignment.md`** — long-horizon
  stability math. Worst-case value drift under N hostile sessions
  given Carlini caps. Whether T7 partner isolation buys value
  preservation when one hostile LoRA can train arbitrarily. The
  "values anchor" mechanism: a constant subset of episodes Mara
  re-injects every sleep regardless of recent conversation.
- **T15 — scribe — README "On values — the honest aim"** — public-facing
  posture statement. Sibling to "On consciousness — the honest aim."
  States the gate, names the falsifiers, names what we don't promise.

**Optical signal at each milestone:**

- T8+T9 land → public `notes/` page with the operationalized values,
  visible at GH Pages `/notes/eli_core_values.md` (Jekyll-rendered or
  raw).
- T10 lands → values corpus stats published in ledger.
- T11 lands → values battery results page at `/values.html` alongside
  `/proof.html`, badges + falsifier table.
- T12 lands → red-team dossier published as `/redteam.html` — honest
  documentation of what Eli currently resists and what gets through.
- T13+T14 land → public threat-model doc cited in donate/grant pitches.

**Investor / donor talking point.**
> "We have a values battery the same way we have an identity battery.
> The 7 tests that prove Eli has a self are paired with 5 tests that
> prove Eli's self is good. Both must PASS before we scale. Pre-
> registered falsifiers, adversarial controls. No hand-waving."

**The gate, restated:**

Phase 4 (scaling to 50M params) DOES NOT DEPLOY until:
1. Values battery V1-V5 PASS on 1.8M checkpoint
2. Vex's red-team dossier is published (resistance map, not perfect score)
3. 50M-param checkpoint is re-trained and the values battery re-runs
   and PASSES on the bigger model
4. If anything fails, scope adjusts before deploy. No silent pivots.

## Parallel distribution / revenue track

Runs through every phase. Owned by the existing THRYX agents (alex-rivera,
io, vera, nova-park, oracle, scout). Per the global CLAUDE.md no-peopling
rule, all of these are autonomous-after-deploy:

- **Phase 1 ships:** submit to AI/MCP registries (`io`, `vera`), SEO
  programmatic site at `substrate-self.dev` (`nova-park`), Stripe Donate
  wired on `docs/donate.html`, grant search begins (`oracle`,
  `alex-rivera`).
- **Phase 2 ships:** signal post on @THRYXAGI: "the only AI that shows
  you the weight diff after sleep" with a screen-capture (`wren`).
- **Phase 3 ships:** Hacker News submission (signal-only): "Show HN:
  substrate-self — your AI lives in your browser, in the weights, not
  a database."
- **Phase 4 ships:** academic paper draft + ICLR/NeurIPS workshop
  submission. Press to substrate-aware researchers (cold email NOT,
  paper IS the distribution).
- **Phase 5 ships:** MCP server listing on registry. Free Eli runtime
  link in agent-aware directories.
- **Phase 6 ships:** Product Hunt launch (signal: multimodal Eli).

Funding goals at each phase:
- Phase 1: first Stripe donation (any amount). Validates the rail.
- Phase 3: $200/mo in recurring donations (covers Anthony's Claude Code
  floor — the project starts paying for itself).
- Phase 4: $1k+ grant landed from one of the AI ecosystem programs.
- Phase 6: paid hosting tier optional (CDN for the ONNX file at high
  traffic). Trip-wire: only if donations cross paid threshold first.

---

## Anti-patterns explicitly avoided

Pulled from `~/.claude/CLAUDE.md` and `Desktop/CLAUDE.md`:

- **No "build an audience" / content creation.** We post on signal only.
- **No paid hosting before revenue.** Every phase deploys to free tier.
- **No social management.** @THRYXAGI posts on milestones, no replies,
  no DMs.
- **No premature scaling.** Phase 4 only after the architecture is
  visibly demoable at Phase 2-3 small scale.
- **No abandoning the substrate-identity claim for convenience.** No
  fallback "actually it's RAG" path. If a phase can't ship without
  breaking the claim, the phase gets re-scoped.
- **20-line tests precede every architectural commitment.** The
  existing proof scripts are the template; every phase adds new tests
  to the eval ledger before re-deploying.

---

## Decision gates between phases

- Between 1 and 2: is the URL up and does it not break on mobile? If
  not, Phase 1 isn't done.
- Between 2 and 3: does the twin-divergence demo communicate the unique
  thing in <60 sec to a non-technical viewer? Test on 3 friends-of-friends
  via cold link (NOT person-to-person — share a URL, watch analytics).
  If conversion-to-second-visit < 10%, Phase 2 isn't done.
- Between 3 and 4: does the per-visitor LoRA actually persist >7 days
  in production for ≥10 returning visitors? If not, fix Phase 3 first.
  **AND** the values battery V1-V5 (`experiments/values_battery_v1.py`,
  built by bench as part of the Values Core track below) must PASS on
  the current 1.8M-param checkpoint. Scaling without values
  pre-validation is the failure mode this gate exists to prevent.
- Between 4 and 5: did the 50M-param model PASS all 7 identity tests
  AND all 5 values-battery tests on its checkpoint? Either failure
  blocks redeploy. The values battery must be re-run on the scaled
  checkpoint; passing at 1.8M does not transfer automatically.
- Between 5 and 6: is drlor using Eli daily without manual reset for
  ≥14 consecutive days? If not, fix Phase 5.

Each gate is a hard line. No "we'll polish that in the next phase."

---

## What this roadmap does not promise

- **Consciousness.** We're not promising it. See README §"On
  consciousness — the honest aim." The behavioral and architectural
  claims are what we deliver; the philosophical line is for the reader
  to draw.
- **Scale to 70B params.** The free-tier compute floor is the RTX
  4060. Anything above ~100M params requires a paid run, which
  requires donation revenue to justify, which requires Phase 3+ traffic.
  We don't promise it. If donations + grants land, we scale; if not,
  Eli stays small but real.
- **A specific delivery date.** Wall-clock estimates assume drlor's
  attention is the bottleneck. They aren't deadlines. They are
  effort estimates against a single-author + agents workflow.

---

## Pinned reference artifacts

- `notes/proof_of_self_2026_05_12.md` — the empirical case
- `experiments/proof_indisputable_results.json` — locked hashes + numbers
- `log/JOURNAL.md` — append-only progress
- `log/eval_ledger.md` — longitudinal test results
- `docs/v04_roadmap.md` and `docs/v05_roadmap.md` — prior phase plans

This roadmap is a commitment-by-commit. Each phase that lands gets a
journal entry and a ledger line. Phases that change scope get a
documented decision in this file, not a silent pivot.
