"""Tests for the v0.4 multi-partner schema migration (Phase 1).

Covers the four guarantees Phase 1 ships:
  1. v0.3 file loads as v0.4 with the legacy partner migrated to "anthony"
  2. Round-trip with multiple partners preserves all data
  3. Active-partner switch persists across save/load
  4. Episodes carry the partner_id tag through serialization
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

# Make the repo importable when running pytest from the repo root or tests dir
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from substrate_self import core, persistence


# --- 1. v0.3 file loads as v0.4 -------------------------------------------

def test_v3_loads_as_v4(tmp_path: Path):
    """A v0.3-shaped JSON file (schema_version=1, partner_facts dict) must
    load without errors and produce a substrate where 'anthony' is the
    active partner with full trust and the migrated facts.
    """
    v3 = {
        "name": "Eli",
        "born": "2026-05-10T15:58:30.250186+00:00",
        "age_sessions": 1,
        "self_facts": {},
        "dispositions": {"concise": 0.7, "rigor": 0.85},
        "partner_facts": {
            "name": "Anthony Snider",
            "handle": "drlor",
            "kids": "6",
        },
        "memories": [
            {
                "timestamp": "2026-05-10T16:01:09.303142+00:00",
                "text": "First successful end-to-end substrate cycle test",
                "tags": ["milestone", "self-test"],
                "weight": 1.0,
                "source_session": None,
            }
        ],
        "open_threads": [
            "Push substrate-self repo to GitHub publicly",
            "Demonstrate cross-session identity persistence",
        ],
        "style": {},
        "episodic": [],
        "last_wake": "2026-05-10T16:01:09.303044+00:00",
        "last_sleep": "2026-05-10T16:01:09.303465+00:00",
        "schema_version": 1,
    }
    p = tmp_path / "substrate.json"
    p.write_text(json.dumps(v3), encoding="utf-8")

    s = persistence.load(p)

    assert s.schema_version == 2, f"expected schema bumped to 2, got {s.schema_version}"
    assert "anthony" in s.partners, f"partners after migration: {list(s.partners.keys())}"
    assert s.active_partner_id == "anthony"

    anthony = s.partners["anthony"]
    assert anthony.display_name == "Anthony Snider"
    assert anthony.handle == "drlor"
    assert anthony.trust == 1.0, "creator must get full trust on migration"
    assert anthony.facts == {"kids": "6"}, f"unexpected facts: {anthony.facts}"

    # Identity-layer fields preserved
    assert s.name == "Eli"
    assert s.dispositions == {"concise": 0.7, "rigor": 0.85}
    assert len(s.memories) == 1
    assert s.memories[0].text.startswith("First successful")

    # Open threads promoted from list[str] to list[OpenThread] tagged with anthony
    assert len(s.open_threads) == 2
    assert all(isinstance(t, core.OpenThread) for t in s.open_threads)
    assert all(t.partner_id == "anthony" for t in s.open_threads)
    assert str(s.open_threads[0]) == "Push substrate-self repo to GitHub publicly"

    # Legacy attribute access still works (back-compat for bootstrap/probe code)
    assert s.partner_facts["name"] == "Anthony Snider"
    assert s.partner_facts["handle"] == "drlor"
    assert s.partner_facts["kids"] == "6"

    # The migrated substrate must round-trip through save/load unchanged
    persistence.save(s, p)
    s2 = persistence.load(p)
    assert s2.schema_version == 2
    assert s2.active_partner_id == "anthony"
    assert s2.partners["anthony"].trust == 1.0
    # And re-loading is idempotent (no stale partner_facts key reappearing)
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert "partner_facts" not in raw


# --- 2. Round-trip with two partners --------------------------------------

def test_round_trip(tmp_path: Path):
    """Build a fresh substrate with two partners, save, load, assert both
    survive with their full data.
    """
    s = core.Substrate(name="Eli", born="2026-05-10T00:00:00+00:00")
    s.introduce_partner("anthony", "Anthony Snider", handle="drlor", trust=1.0)
    s.switch_partner("anthony")  # so the next introduce auto-records the introducer
    s.introduce_partner("claire", "Claire Doe", handle="claire_d", trust=0.5)
    s.partners["anthony"].facts["role"] = "creator"
    s.partners["anthony"].private_topics.append("medical-debt")
    s.partners["claire"].facts["role"] = "researcher"
    s.partners["claire"].style_notes["formality"] = "professional"
    s.switch_partner("anthony")

    p = tmp_path / "substrate.json"
    persistence.save(s, p)

    loaded = persistence.load(p)
    assert set(loaded.partners.keys()) == {"anthony", "claire"}
    assert loaded.partners["anthony"].trust == 1.0
    assert loaded.partners["anthony"].handle == "drlor"
    assert loaded.partners["anthony"].facts["role"] == "creator"
    assert loaded.partners["anthony"].private_topics == ["medical-debt"]
    assert loaded.partners["claire"].trust == 0.5
    assert loaded.partners["claire"].style_notes["formality"] == "professional"
    assert loaded.active_partner_id == "anthony"
    # introduced_by is auto-populated when a second partner is introduced
    # while another is active
    assert loaded.introduced_by.get("claire") == "anthony"


# --- 3. Active partner switch persists ------------------------------------

def test_active_partner_switch(tmp_path: Path):
    """Switch active partner, persist, reload — switch must stick."""
    s = core.Substrate()
    s.introduce_partner("anthony", "Anthony Snider", trust=1.0)
    s.introduce_partner("claire", "Claire Doe", trust=0.5)
    s.switch_partner("anthony")
    p = tmp_path / "substrate.json"
    persistence.save(s, p)

    # Reload, switch, save, reload again
    loaded = persistence.load(p)
    assert loaded.active_partner_id == "anthony"
    loaded.switch_partner("claire")
    persistence.save(loaded, p)

    again = persistence.load(p)
    assert again.active_partner_id == "claire"
    # And the legacy partner_facts view follows the active partner
    assert again.partner_facts["name"] == "Claire Doe"

    # Switching to an unknown partner must raise
    import pytest
    with pytest.raises(KeyError):
        again.switch_partner("ghost")


# --- 4. Episodes carry partner_id ----------------------------------------

def test_episode_carries_partner(tmp_path: Path):
    """An episode created while a partner is active should serialize
    its partner_id and round-trip through save/load.
    """
    s = core.Substrate()
    s.introduce_partner("anthony", "Anthony Snider", trust=1.0)
    s.introduce_partner("claire", "Claire Doe", trust=0.5)
    s.switch_partner("anthony")
    s.add_episode("user", "Hi from Anthony", significance=0.5)
    s.switch_partner("claire")
    s.add_episode("user", "Hi from Claire", significance=0.5)
    # Explicit partner_id override
    s.add_episode("agent", "Reply tagged anthony explicitly",
                  significance=0.5, partner_id="anthony")

    # Memories also carry partner_id
    s.switch_partner("anthony")
    s.add_memory("Anthony showed me the substrate code", tags=["onboarding"])
    s.switch_partner("claire")
    s.add_memory("Claire reviewed the privacy threat model")
    s.add_memory("Partner-independent reflection", partner_id=None)

    # Open threads
    s.switch_partner("anthony")
    s.add_open_thread("Ship v0.4 Phase 1")
    s.switch_partner("claire")
    s.add_open_thread("Run the privacy regression on v0.4")

    p = tmp_path / "substrate.json"
    persistence.save(s, p)
    loaded = persistence.load(p)

    assert len(loaded.episodic) == 3
    assert loaded.episodic[0].partner_id == "anthony"
    assert loaded.episodic[0].content == "Hi from Anthony"
    assert loaded.episodic[1].partner_id == "claire"
    assert loaded.episodic[2].partner_id == "anthony"  # explicit override

    # Memories
    by_partner = {m.partner_id: m.text for m in loaded.memories if m.partner_id is not None}
    assert by_partner["anthony"].startswith("Anthony showed me")
    assert by_partner["claire"].startswith("Claire reviewed")
    assert any(m.partner_id is None for m in loaded.memories)

    # Open threads
    threads_by_partner = {t.partner_id: t.text for t in loaded.open_threads}
    assert threads_by_partner["anthony"] == "Ship v0.4 Phase 1"
    assert threads_by_partner["claire"] == "Run the privacy regression on v0.4"

    # Per-partner remember() filter
    anth_only = loaded.remember("Anthony", partner_id="anthony")
    assert all(m.partner_id in ("anthony", None) for m in anth_only)
