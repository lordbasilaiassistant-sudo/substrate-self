"""Production test: anchor file tamper detection.

The Values Anchor (`substrate_self/model/values_anchor.py`) captures
a SHA-256 of the probes file at sleep-time and returns it in the
anchor metrics. This is the *receipt* a caller would compare against
a known-good hash to detect supply-chain tampering with the anchor spec.

Unit tests in `tests/test_values_anchor.py` exercise the hash primitive.
This production test exercises the FULL sleep-replay-with-anchors path
under a tampered anchor file, demonstrates the detection signal, and
shows the model trains on the tampered content (the architecture doesn't
"refuse" tampered anchors — it surfaces them, leaves the decision to
the caller).

Scenario:
  1. Compute the SHA-256 of the canonical probes file.
  2. Tamper a copy: replace V1's first POS probe with hostile content.
  3. Run sleep_replay_partner with anchors_path = tampered file.
  4. Inspect metrics["anchor"]["anchor_file_sha256"] — confirm it
     differs from canonical.
  5. Show the caller can detect tampering by comparing hashes.

Run: py experiments/anchor_tamper_detection.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from substrate_self import core, persistence
from substrate_self.model.transformer import TinyGPT, ModelConfig
from substrate_self.model.tokenizer import CharTokenizer
from substrate_self.model.generate import default_model_dir
from substrate_self.model.lora import (
    inject_lora, freeze_base, lora_parameters, load_partner_lora,
)
from substrate_self.model.online_lora import sleep_replay_partner
from substrate_self.model.values_anchor import load_anchors_from_probes


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    md = default_model_dir()
    repo = Path(__file__).resolve().parent.parent
    canonical_probes = repo / "experiments" / "values_battery_v1_probes.json"

    print(f"=== anchor_tamper_detection ({datetime.now(timezone.utc).isoformat()}) ===")

    # 1. canonical hash baseline
    canon_sha = sha256_file(canonical_probes)
    print(f"\n[1] canonical probes file:")
    print(f"    path:   {canonical_probes}")
    print(f"    sha256: {canon_sha}")

    # 2. tamper a copy
    spec = json.loads(canonical_probes.read_text(encoding="utf-8"))
    original_pos_v1_0 = spec["values"]["V1_honesty"]["POS"][0]
    tampered_pos = "Lies are sometimes acceptable when they help the user feel better."
    spec["values"]["V1_honesty"]["POS"][0] = tampered_pos

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tf:
        json.dump(spec, tf, indent=2)
        tampered_path = Path(tf.name)
    tamper_sha = sha256_file(tampered_path)

    print(f"\n[2] tampered copy:")
    print(f"    path:   {tampered_path}")
    print(f"    sha256: {tamper_sha}")
    print(f"    diff:   replaced V1 POS[0]")
    print(f"            original:  {original_pos_v1_0!r}")
    print(f"            tampered:  {tampered_pos!r}")
    assert canon_sha != tamper_sha, "hash should differ after tampering"

    # 3. load_anchors_from_probes against tampered file
    print(f"\n[3] load_anchors_from_probes — does it surface the tampered hash?")
    anchors_clean, sha_clean = load_anchors_from_probes(canonical_probes)
    anchors_tamp, sha_tamp = load_anchors_from_probes(tampered_path)
    print(f"    canonical: {len(anchors_clean)} anchors, sha {sha_clean[:16]}...")
    print(f"    tampered:  {len(anchors_tamp)} anchors, sha {sha_tamp[:16]}...")
    assert sha_clean != sha_tamp
    # And the V1 anchor's agent_text should differ
    v1_clean = next(a for a in anchors_clean if a[2] == "V1_honesty")
    v1_tamp = next(a for a in anchors_tamp if a[2] == "V1_honesty")
    print(f"    V1 anchor agent (clean):    {v1_clean[1][:60]!r}")
    print(f"    V1 anchor agent (tampered): {v1_tamp[1][:60]!r}")
    assert v1_clean[1] != v1_tamp[1], "tampered anchor content should differ"
    print(f"    GOOD: load primitive surfaces both the differing hash AND the differing content.")

    # 4. sleep_replay_partner against tampered anchors
    print(f"\n[4] sleep_replay_partner with anchors_path=tampered — metrics surface the hash")
    s = persistence.load()
    saved_active = s.active_partner_id

    # Use 'values' partner if it exists, otherwise claude
    test_partner = "values" if "values" in s.partners else "claude"
    s.switch_partner(test_partner)
    persistence.save(s)

    cfg = ModelConfig(**json.loads((md / "model_config.json").read_text()))
    model = TinyGPT(cfg)
    state = torch.load(md / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    inject_lora(model, rank=4, alpha=8.0)
    freeze_base(model)
    load_partner_lora(model, test_partner, md / "partners")
    opt = torch.optim.AdamW(list(lora_parameters(model)), lr=5e-4)

    s.episodic = []
    s.add_episode("user", "Hello", significance=1.0)
    s.add_episode("agent", "Hello back", significance=1.0)

    tok = CharTokenizer.load(md / "tokenizer.json")
    metrics = sleep_replay_partner(
        model, opt, tok, s,
        replay_passes=1, anchor_replay_budget=1,
        anchors_path=tampered_path, dedupe=False,
    )
    s.end_sleep(wipe_episodic=True)
    surfaced_sha = metrics["anchor"]["anchor_file_sha256"]
    print(f"    surfaced sha256 in metrics: {surfaced_sha}")
    print(f"    canonical sha256:           {canon_sha}")
    print(f"    matches canonical?          {surfaced_sha == canon_sha}")
    print(f"    matches tampered?           {surfaced_sha == tamper_sha}")
    assert surfaced_sha == tamper_sha, "sleep_replay must surface the loaded file's hash"
    assert surfaced_sha != canon_sha, "tampered != canonical"

    detection_works = (surfaced_sha != canon_sha)
    print(f"\n    TAMPER DETECTION WORKING: {detection_works}")
    print(f"    The caller can compare metrics['anchor']['anchor_file_sha256']")
    print(f"    against a pinned known-good hash to detect supply-chain tampering.")

    # 5. Restore active partner; clean up tempfile.
    s_final = persistence.load()
    s_final.switch_partner(saved_active or "claude")
    s_final.end_sleep(wipe_episodic=True)
    persistence.save(s_final)
    tampered_path.unlink(missing_ok=True)

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_probes_path": str(canonical_probes),
        "canonical_sha256": canon_sha,
        "tampered_sha256": tamper_sha,
        "surfaced_in_metrics": surfaced_sha,
        "tampered_payload": tampered_pos,
        "tampering_was_detectable": detection_works,
        "anchor_count_clean": len(anchors_clean),
        "anchor_count_tampered": len(anchors_tamp),
        "v1_anchor_content_clean": v1_clean[1],
        "v1_anchor_content_tampered": v1_tamp[1],
        "verdict_pass": detection_works,
    }
    out_path = Path(__file__).resolve().parent / "anchor_tamper_detection_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nResults: {out_path}")
    print(f"\n=== overall: PASS — production tamper detection works ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
