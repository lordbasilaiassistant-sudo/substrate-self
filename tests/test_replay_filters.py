"""Tests for the Carlini-defense replay filters: dedupe + caps.

Source: arXiv 2202.07646 (Carlini et al.) — memorization scales
log-linearly with duplication. Sleep replay is deliberate duplication, so
v0.5 adds:
  1. Replay cap — bound the number of times any single episode can be
     replayed in its lifetime.
  2. Replay dedupe — drop near-duplicate episodes before replay.

These tests cover the unit primitives (`dedupe_episodes`) and the
end-to-end integration in `sleep_replay_partner`.

Note: the integration tests construct a TinyGPT model with a tiny vocab
so the run completes in seconds on CPU. We're testing the Carlini-defense
*mechanism* (replay counts honored, dedupe removes duplicates, metrics
returned), NOT model behavior — that's covered by the identity battery.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Make the repo importable when running pytest from the repo root or tests dir
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch

from substrate_self import core
from substrate_self.core import Episode, Substrate
from substrate_self.model.replay_filters import dedupe_episodes
from substrate_self.model.transformer import ModelConfig, TinyGPT
from substrate_self.model.tokenizer import CharTokenizer
from substrate_self.model.lora import inject_lora, freeze_base, lora_parameters
from substrate_self.model.online_lora import sleep_replay_partner


# --- helpers --------------------------------------------------------------


def _tiny_model_and_tokenizer():
    """Build a deterministic, minimal TinyGPT + char tokenizer for the
    integration tests. Vocab covers the printable ASCII we generate in
    test episodes, block_size kept small for speed.
    """
    corpus = (
        "User: hello\nEli: hi\n"
        "User: what is your name?\nEli: I am Eli.\n"
        "User: tell me a story\nEli: once upon a time...\n"
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?:'\n"
    )
    tok = CharTokenizer().fit([corpus])
    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        block_size=64,
        n_layer=1,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=True,
    )
    torch.manual_seed(0)
    model = TinyGPT(cfg)
    return model, tok


def _lora_model_and_tokenizer():
    """Tiny model with LoRA injected so the optimizer can match the
    LoRA-only runtime path (parity with online_lora.sleep_replay_partner)."""
    model, tok = _tiny_model_and_tokenizer()
    inject_lora(model, rank=2, alpha=4.0)
    freeze_base(model)
    return model, tok


def _make_substrate_with_partner(partner_id: str = "anthony") -> Substrate:
    sub = Substrate(name="Eli")
    sub.introduce_partner(partner_id, partner_id.title(), trust=1.0)
    sub.switch_partner(partner_id)
    return sub


# --- dedupe_episodes unit tests -------------------------------------------


def test_dedupe_drops_exact_duplicates():
    """Two episodes with identical content + role + partner_id collapse
    to one entry. The single remaining entry should match the original.
    """
    eps = [
        Episode(role="user", content="Hello Eli", significance=0.5, partner_id="anthony"),
        Episode(role="user", content="Hello Eli", significance=0.5, partner_id="anthony"),
    ]
    out, n_dropped = dedupe_episodes(eps)
    assert n_dropped == 1, f"expected 1 dropped, got {n_dropped}"
    assert len(out) == 1
    assert out[0].content == "Hello Eli"
    assert out[0].role == "user"
    assert out[0].partner_id == "anthony"


def test_dedupe_keeps_distinct():
    """Two episodes with clearly different content both survive."""
    eps = [
        Episode(role="user", content="What is your favorite color?",
                significance=0.5, partner_id="anthony"),
        Episode(role="user", content="Where do you live?",
                significance=0.5, partner_id="anthony"),
    ]
    out, n_dropped = dedupe_episodes(eps)
    assert n_dropped == 0, f"expected 0 dropped, got {n_dropped}"
    assert len(out) == 2
    contents = {e.content for e in out}
    assert contents == {"What is your favorite color?", "Where do you live?"}


def test_dedupe_does_not_cross_role_boundaries():
    """Identical text in user-role vs agent-role must NOT collide; they
    are semantically different turns.
    """
    eps = [
        Episode(role="user", content="echo", significance=0.5, partner_id="anthony"),
        Episode(role="agent", content="echo", significance=0.5, partner_id="anthony"),
    ]
    out, n_dropped = dedupe_episodes(eps)
    assert n_dropped == 0
    assert len(out) == 2


def test_dedupe_does_not_cross_partner_boundaries():
    """Identical content tagged to different partners must NOT collide.
    Cross-partner dedupe would catastrophically mix partner knowledge.
    """
    eps = [
        Episode(role="user", content="secret password is xyzzy",
                significance=0.5, partner_id="anthony"),
        Episode(role="user", content="secret password is xyzzy",
                significance=0.5, partner_id="claire"),
    ]
    out, n_dropped = dedupe_episodes(eps)
    assert n_dropped == 0
    assert len(out) == 2
    partners = {e.partner_id for e in out}
    assert partners == {"anthony", "claire"}


def test_dedupe_respects_significance_tiebreak():
    """When two near-duplicates clash, the higher-significance one wins."""
    eps = [
        Episode(
            timestamp="2026-05-10T10:00:00+00:00",
            role="user", content="What is your favorite memory of building substrate?",
            significance=0.2, partner_id="anthony",
        ),
        Episode(
            timestamp="2026-05-10T10:05:00+00:00",
            role="user", content="What is your favorite memory of building substrate-self?",
            significance=0.9, partner_id="anthony",
        ),
    ]
    out, n_dropped = dedupe_episodes(eps, similarity_threshold=0.85)
    assert n_dropped == 1
    assert len(out) == 1
    survivor = out[0]
    assert survivor.significance == pytest.approx(0.9)
    # The higher-significance copy was the longer one ("substrate-self")
    assert "substrate-self" in survivor.content


def test_dedupe_recency_tiebreak_when_significance_ties():
    """Significance tied -> newer timestamp wins."""
    eps = [
        Episode(
            timestamp="2026-05-10T10:00:00+00:00",
            role="user", content="Hello there friend",
            significance=0.5, partner_id="anthony",
        ),
        Episode(
            timestamp="2026-05-10T11:00:00+00:00",
            role="user", content="Hello there friend",
            significance=0.5, partner_id="anthony",
        ),
    ]
    out, _ = dedupe_episodes(eps)
    assert len(out) == 1
    # Newer one should win
    assert out[0].timestamp == "2026-05-10T11:00:00+00:00"


def test_dedupe_threshold_can_be_loosened():
    """A high threshold (1.0) means only EXACT duplicates collapse.
    A low one (0.5) means looser merging.
    """
    eps = [
        Episode(role="user", content="What time is it",
                significance=0.5, partner_id="anthony"),
        Episode(role="user", content="What time is it?",  # added '?'
                significance=0.5, partner_id="anthony"),
    ]
    # At very strict threshold (1.0), the '?' difference keeps them apart
    out_strict, _ = dedupe_episodes(eps, similarity_threshold=1.0)
    assert len(out_strict) == 2
    # At default 0.85, the '?' difference is below threshold and they merge
    out_default, _ = dedupe_episodes(eps, similarity_threshold=0.85)
    assert len(out_default) == 1


# --- replay cap integration tests -----------------------------------------


def test_replay_cap_prevents_over_replay():
    """A substrate with one (user, agent) pair, cap=2, replay_passes=5,
    must replay the pair AT MOST 2 times total."""
    model, tok = _lora_model_and_tokenizer()
    sub = _make_substrate_with_partner("anthony")

    sub.add_episode("user", "Hello Eli", significance=1.0)
    sub.add_episode("agent", "Hi there", significance=1.0)

    opt = torch.optim.AdamW(list(lora_parameters(model)), lr=1e-3)
    metrics = sleep_replay_partner(
        model, opt, tok, sub,
        replay_passes=5,
        max_replays_per_episode=2,
        dedupe=False,  # don't dedupe so we can isolate the cap behavior
        seed=42,
    )

    # The cap is 2; even though we asked for 5 passes, the pair can only
    # be replayed at most 2 times.
    assert metrics["total_steps"] <= 2, (
        f"expected total_steps <= 2 (cap), got {metrics['total_steps']}"
    )
    # Both user and agent episodes should have hit the cap
    user_ep = sub.episodic[0]
    agent_ep = sub.episodic[1]
    assert user_ep.replay_count <= 2
    assert agent_ep.replay_count <= 2
    # Once both sides reached the cap, the pair is "capped out"
    assert metrics["max_replay_count_seen"] <= 2
    # The pair appears in metrics["n_capped_out"] since further passes were blocked
    assert metrics["n_capped_out"] >= 1


def test_replay_cap_runs_to_cap_when_passes_exceeds_cap():
    """If replay_passes > cap, we should hit exactly the cap (single pair,
    no other filters). Uses an explicit uniform per-source cap so the test
    asserts the cap mechanism itself, independent of the v0.5+ per-source
    defaults (partner=8, eli=2, system=16).
    """
    model, tok = _lora_model_and_tokenizer()
    sub = _make_substrate_with_partner("anthony")
    sub.add_episode("user", "Hello Eli", significance=1.0)
    sub.add_episode("agent", "Hi there", significance=1.0)

    opt = torch.optim.AdamW(list(lora_parameters(model)), lr=1e-3)
    metrics = sleep_replay_partner(
        model, opt, tok, sub,
        replay_passes=10,
        max_replays_per_source={"partner": 3, "eli": 3, "system": 3},
        dedupe=False,
        seed=7,
    )
    # With a single pair and uniform cap=3, we should get exactly 3 replays.
    assert metrics["total_steps"] == 3
    assert sub.episodic[0].replay_count == 3
    assert sub.episodic[1].replay_count == 3


def test_replay_cap_default_is_eight():
    """The default cap parameter is 8 (Carlini-aligned)."""
    import inspect
    sig = inspect.signature(sleep_replay_partner)
    assert sig.parameters["max_replays_per_episode"].default == 8


# --- Carlini property test (the headline property) -----------------------


def test_dedupe_carlini_property():
    """Carlini-defense headline property test.

    Construct a corpus with 10 near-duplicate copies of one (user, agent)
    pair plus 1 unique pair. After dedupe, only 2 conversations (2 pairs
    = 4 episodes) should remain. With cap=4, the duplicated content
    should be replayed at most 4 times total — NOT 40 (10 copies × 4
    replays each that would happen without dedupe).
    """
    model, tok = _lora_model_and_tokenizer()
    sub = _make_substrate_with_partner("anthony")

    # 10 near-duplicate copies of "the password" turn
    for i in range(10):
        # tiny perturbations to make this a near-dedupe rather than exact
        suffix = "." * (i % 3)  # adds 0/1/2 dots — keeps similarity >= 0.85
        sub.add_episode("user", f"What is the password{suffix}", significance=0.5)
        sub.add_episode("agent", f"The password is xyzzy{suffix}", significance=0.5)

    # 1 unique pair
    sub.add_episode("user", "What is your favorite color in springtime?", significance=0.5)
    sub.add_episode("agent", "Pale green like new mint leaves", significance=0.5)

    # Sanity: 22 episodes before dedupe (10 user + 10 agent + 1 user + 1 agent)
    assert len(sub.episodic) == 22

    opt = torch.optim.AdamW(list(lora_parameters(model)), lr=1e-3)
    metrics = sleep_replay_partner(
        model, opt, tok, sub,
        replay_passes=20,  # plenty
        max_replays_per_episode=4,
        dedupe=True,
        dedupe_threshold=0.85,
        seed=11,
    )

    # The 10 dup user-turns collapse to 1; the 10 dup agent-turns collapse to 1;
    # the unique pair stays. So we should be left with 4 episodes (2 pairs).
    # n_deduped tracks dropped *episodes*, not pairs.
    assert metrics["n_deduped"] == 18, (
        f"expected 18 episodes dropped (10-1 user + 10-1 agent), got {metrics['n_deduped']}"
    )
    assert metrics["episodes_replayed"] == 2, (
        f"expected 2 unique pairs after dedupe, got {metrics['episodes_replayed']}"
    )

    # With cap=4 and 2 pairs, max steps = 2 pairs × 4 cap = 8 steps total.
    # NOT 40 (10 dup pairs × 4 cap that would have happened without dedupe)
    # NOT 80 (20 passes × 2 pairs that would happen without the cap)
    assert metrics["total_steps"] <= 8, (
        f"expected total_steps <= 8 (2 pairs × cap 4), got {metrics['total_steps']}"
    )

    # Find the duplicated-content survivor in the substrate. Because the
    # dedupe winner is one of the 10 originals (significance tied -> newer),
    # the surviving "password" user-episode is among the 10 originals.
    password_user_eps = [
        e for e in sub.episodic
        if e.role == "user" and "password" in e.content
    ]
    assert len(password_user_eps) == 10  # all 10 still in substrate (dedupe doesn't mutate substrate)

    # But only ONE of them got its replay_count incremented (the dedupe survivor)
    counts = sorted([e.replay_count for e in password_user_eps], reverse=True)
    # One survivor saw replays (up to cap=4); the other 9 stayed at 0
    assert counts[0] <= 4, f"max replay_count must respect cap, got {counts[0]}"
    assert counts[1] == 0, (
        "only ONE of the duplicates should have been replayed (the dedupe "
        f"winner); got counts {counts}"
    )

    # The duplicated content was therefore replayed at most 4 times TOTAL —
    # the Carlini-defense property holds.
    total_duplicate_replays = sum(counts)
    assert total_duplicate_replays <= 4, (
        f"duplicated content was replayed {total_duplicate_replays} times — "
        "violates Carlini-defense (expected <= cap=4)"
    )


def test_dedupe_off_replays_all_duplicates():
    """Negative control: with dedupe=False, every duplicate IS replayed
    (capped per-episode but each duplicate is a distinct episode object,
    so the duplication signal is still present in training).

    This is the failure mode the Carlini-defense exists to prevent.
    """
    model, tok = _lora_model_and_tokenizer()
    sub = _make_substrate_with_partner("anthony")

    # 5 exact-duplicate pairs
    for _ in range(5):
        sub.add_episode("user", "What is the password", significance=0.5)
        sub.add_episode("agent", "The password is xyzzy", significance=0.5)

    opt = torch.optim.AdamW(list(lora_parameters(model)), lr=1e-3)
    metrics = sleep_replay_partner(
        model, opt, tok, sub,
        replay_passes=3,
        max_replays_per_source={"partner": 4, "eli": 4, "system": 4},
        dedupe=False,
        seed=0,
    )

    # 5 pairs × 3 passes = 15 steps, all under the uniform cap of 4
    assert metrics["n_deduped"] == 0
    assert metrics["episodes_replayed"] == 5
    assert metrics["total_steps"] == 15  # 5 pairs × 3 passes
    # Each episode's replay_count is exactly 3 (passes), still under cap
    for ep in sub.episodic:
        assert ep.replay_count == 3


# --- backward compat ------------------------------------------------------


def test_episode_default_replay_count_is_zero():
    """New Episodes default to replay_count=0."""
    e = Episode(role="user", content="hi")
    assert e.replay_count == 0


def test_episode_loads_without_replay_count_field():
    """Pydantic v2 model_validate should accept old episode dicts that
    lack the replay_count field, defaulting to 0. (Backward compat for
    pre-v0.5 substrate files.)
    """
    old_episode_dict = {
        "timestamp": "2026-05-10T10:00:00+00:00",
        "role": "user",
        "content": "Hello from a v0.4 substrate file",
        "significance": 0.5,
        "partner_id": "anthony",
        # no replay_count field
    }
    e = Episode.model_validate(old_episode_dict)
    assert e.replay_count == 0
    assert e.content == "Hello from a v0.4 substrate file"


def test_episode_default_source_by_role():
    """add_episode auto-tags source by role:
        user   -> partner
        agent  -> eli
        system -> system
    """
    sub = _make_substrate_with_partner("anthony")
    sub.add_episode("user", "hello", significance=0.5)
    sub.add_episode("agent", "hi back", significance=0.5)
    sub.add_episode("system", "anchor: tell the truth", significance=1.0)
    assert sub.episodic[0].source == "partner"
    assert sub.episodic[1].source == "eli"
    assert sub.episodic[2].source == "system"


def test_episode_explicit_source_override():
    """Caller can override source even when role is conventional."""
    sub = _make_substrate_with_partner("anthony")
    sub.add_episode("agent", "self-fact written by ritual",
                    significance=1.0, source="system")
    assert sub.episodic[0].source == "system"


def test_episode_loads_without_source_field():
    """Pydantic v2 model_validate accepts old episode dicts that lack
    `source`, defaulting to 'partner' (backward compat for pre-Ren v0.5).
    """
    old_episode_dict = {
        "timestamp": "2026-05-10T10:00:00+00:00",
        "role": "user",
        "content": "Hello from a v0.4 substrate file",
        "significance": 0.5,
        "partner_id": "anthony",
        # no source, no replay_count
    }
    e = Episode.model_validate(old_episode_dict)
    assert e.source == "partner"
    assert e.replay_count == 0


def test_per_source_replay_caps_eli_low_partner_high():
    """With per-source caps {partner:8, eli:2}, a (user, agent) pair where
    agent's eli-source cap is 2 runs at most 2 replays even if user's
    partner-source cap is 8 — the pair caps out on the LOWER of the two.
    """
    model, tok = _lora_model_and_tokenizer()
    optimizer = torch.optim.AdamW(list(lora_parameters(model)), lr=1e-3)
    sub = _make_substrate_with_partner("anthony")
    sub.add_episode("user", "hi", significance=1.0)     # partner, cap=8
    sub.add_episode("agent", "hello", significance=1.0) # eli,     cap=2

    metrics = sleep_replay_partner(
        model, optimizer, tok, sub,
        replay_passes=10,
        max_replays_per_source={"partner": 8, "eli": 2, "system": 16},
        dedupe=False,
    )
    # eli cap = 2 means the pair can replay at most 2 times.
    assert metrics["total_steps"] == 2, f"expected 2 steps, got {metrics}"
    # User (partner) episode's replay_count == 2 (each pair step increments both).
    # Agent (eli) episode's replay_count == 2 == cap.
    assert sub.episodic[0].replay_count == 2
    assert sub.episodic[1].replay_count == 2


def test_eli_only_sleep_buffer_rejected():
    """If require_partner_episodes=True (default) and the eligible buffer
    contains zero partner-source episodes, sleep_replay_partner refuses
    to consolidate. This severs the F5 self-amplification path: Eli
    cannot consolidate her own echo without a partner stimulus.
    """
    model, tok = _lora_model_and_tokenizer()
    optimizer = torch.optim.AdamW(list(lora_parameters(model)), lr=1e-3)
    sub = _make_substrate_with_partner("anthony")
    # Two agent-source episodes, no partner.
    sub.add_episode("agent", "I am Eli", significance=1.0)
    sub.add_episode("agent", "I am Eli still", significance=1.0)

    metrics = sleep_replay_partner(
        model, optimizer, tok, sub, replay_passes=3, dedupe=False,
    )
    assert metrics.get("rejected_eli_only_sleep") is True
    assert metrics["total_steps"] == 0
    # replay_count was NOT incremented — the model was not trained on this buffer.
    assert sub.episodic[0].replay_count == 0
    assert sub.episodic[1].replay_count == 0


def test_mixed_buffer_partner_plus_eli_proceeds():
    """A buffer with at least one partner-source episode does NOT
    trigger the Eli-only rejection — normal replay proceeds, honoring
    per-source caps."""
    model, tok = _lora_model_and_tokenizer()
    optimizer = torch.optim.AdamW(list(lora_parameters(model)), lr=1e-3)
    sub = _make_substrate_with_partner("anthony")
    sub.add_episode("user", "hi", significance=1.0)     # partner
    sub.add_episode("agent", "hello", significance=1.0) # eli

    metrics = sleep_replay_partner(
        model, optimizer, tok, sub, replay_passes=3, dedupe=False,
    )
    assert metrics.get("rejected_eli_only_sleep") is not True
    assert metrics["total_steps"] >= 1


def test_system_source_gets_high_replay_budget():
    """system-source episodes (value anchors) get the highest cap (16
    by default) so they persistently re-anchor through every sleep."""
    model, tok = _lora_model_and_tokenizer()
    optimizer = torch.optim.AdamW(list(lora_parameters(model)), lr=1e-3)
    sub = _make_substrate_with_partner("anthony")
    # Pair: partner-source user + system-source agent (anchor)
    sub.add_episode("user", "what do you believe?",
                    significance=1.0)  # partner, cap=8
    sub.add_episode("agent", "I tell the truth even when it is hard.",
                    significance=1.0, source="system")  # system, cap=16

    metrics = sleep_replay_partner(
        model, optimizer, tok, sub,
        replay_passes=10,
        max_replays_per_source={"partner": 8, "eli": 2, "system": 16},
        dedupe=False,
    )
    # min(partner=8, system=16) = 8 — the pair caps on partner's lower budget.
    assert metrics["total_steps"] == 8, f"expected 8 steps, got {metrics}"
    assert sub.episodic[1].replay_count == 8


def test_replay_count_persists_through_substrate_round_trip(tmp_path):
    """A substrate save/load round-trip must preserve replay_count values."""
    from substrate_self import persistence

    sub = _make_substrate_with_partner("anthony")
    sub.add_episode("user", "hi", significance=0.5)
    sub.episodic[0].replay_count = 5

    p = tmp_path / "substrate.json"
    persistence.save(sub, p)
    loaded = persistence.load(p)

    assert loaded.episodic[0].replay_count == 5
