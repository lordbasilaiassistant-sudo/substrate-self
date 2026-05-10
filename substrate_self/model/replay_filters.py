"""Replay filters: dedupe + caps for Carlini-defense.

Carlini et al. arXiv 2202.07646 ("Quantifying Memorization Across Neural
Language Models") established empirically that LM memorization scales
**log-linearly with duplication**. Sleep replay in substrate-self is
*deliberate duplication* — each replay pass re-exposes the model to the
same episode, by design, to consolidate it into weights. That is the
mechanism by which Eli learns from experience, but it is also the
worst-case mechanism against memorization-extraction attacks.

Two cheap, complementary defenses (from `notes/research_discretion.md`):

  1. **Replay caps.** Bound the number of times any single episode can be
     replayed in its lifetime. Linearizes the duplication axis on Carlini's
     log-linear curve at a configurable ceiling.

  2. **Replay dedupe.** Drop near-duplicate episodes before they ever
     enter the replay loop. Near-duplicates would otherwise act as
     stealth duplication (an episode replayed once and its near-copy
     replayed once is, from the model's perspective, two duplications of
     roughly the same content).

This module owns the dedupe primitive. The cap primitive is owned by
`online_lora.sleep_replay_partner` because the cap requires per-pass
mutation of `Episode.replay_count`.

Design choices (documented for Mara's manifest discipline):
  - `difflib.SequenceMatcher.ratio()` for similarity — stdlib only, no
    external deps. Quadratic in number of pairs but episodic buffer
    is small (~tens of entries; substrate.episodic gets wiped each sleep)
    so O(n²) is fine in practice.
  - Group by `(role, partner_id)` before comparing. Two episodes with
    different roles or different partners are never duplicates of each
    other regardless of content similarity — they belong to different
    semantic slots.
  - Tie-break by significance (descending), then by recency (newer wins).
    This keeps the "load-bearing" version of a near-duplicate pair.
  - Preserve relative original order for surviving episodes so that
    downstream `user -> agent` pair construction (in sleep_replay_partner)
    still yields coherent turn pairs.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Iterable, List, Tuple

from substrate_self.core import Episode


def _content_similarity(a: str, b: str) -> float:
    """Plain SequenceMatcher ratio in [0, 1]. Identical -> 1.0; disjoint -> ~0.0."""
    if a is b or a == b:
        return 1.0
    return SequenceMatcher(a=a, b=b).ratio()


def _is_newer(a: Episode, b: Episode) -> bool:
    """True iff a.timestamp >= b.timestamp (ISO-8601 strings sort lexically)."""
    return a.timestamp >= b.timestamp


def _winner_index(idx_a: int, ep_a: Episode, idx_b: int, ep_b: Episode) -> int:
    """Pick the surviving index between two duplicates.

    Higher significance wins; ties broken by newer timestamp.
    """
    if ep_a.significance > ep_b.significance:
        return idx_a
    if ep_b.significance > ep_a.significance:
        return idx_b
    # tie on significance — newer wins
    return idx_a if _is_newer(ep_a, ep_b) else idx_b


def dedupe_episodes(
    episodes: Iterable[Episode],
    similarity_threshold: float = 0.85,
) -> Tuple[List[Episode], int]:
    """Drop near-duplicate episodes from a buffer.

    Groups episodes by `(role, partner_id)` and runs pairwise
    `SequenceMatcher.ratio()` on `content` within each group. Two episodes
    with ratio >= `similarity_threshold` are duplicates; the survivor is
    the one with higher significance (tie-break: newer timestamp).

    Returns: (deduped_episodes_in_original_order, n_dropped)

    `similarity_threshold=0.85` default rationale:
      - 1.00 catches only exact duplicates — too lax. Realistic chat near-
        duplicates include "Hi" vs "Hi!" and rephrasings that share ~90%
        characters; these still drive Carlini-style memorization.
      - 0.50 conflates semantically distinct turns ("yes" vs "no" share
        zero chars but a longer fragment with shared boilerplate would
        cross 0.5).
      - 0.85 is the empirically-cited mid-point in Carlini-followup work
        (e.g. The Pile near-duplicate sweep used 0.8 jaccard on shingles;
        SequenceMatcher.ratio() is comparable). Tunable per-deployment.
    """
    eps = list(episodes)
    if not eps:
        return [], 0

    n = len(eps)
    # For each index, `winner_for[i]` is the index of the cluster-winner
    # this index belongs to. Initially each index is its own cluster.
    winner_for = list(range(n))

    # Group indices by (role, partner_id)
    groups: dict[tuple[str, str | None], list[int]] = {}
    for i, ep in enumerate(eps):
        groups.setdefault((ep.role, ep.partner_id), []).append(i)

    for _key, idxs in groups.items():
        # We process indices in original order. For each new index, scan
        # known cluster winners in this group; if a duplicate is found,
        # contest with the winner and unify (later survivor becomes the
        # new winner for everyone in that cluster).
        winners_in_group: list[int] = []
        for i in idxs:
            ep_i = eps[i]
            matched_winner = None
            for w_idx in winners_in_group:
                if _content_similarity(ep_i.content, eps[w_idx].content) >= similarity_threshold:
                    matched_winner = w_idx
                    break
            if matched_winner is None:
                winner_for[i] = i
                winners_in_group.append(i)
            else:
                new_winner = _winner_index(matched_winner, eps[matched_winner], i, ep_i)
                if new_winner != matched_winner:
                    # `i` displaces matched_winner; rebind everyone whose
                    # cluster-pointer was matched_winner to point at `i`.
                    for k in range(n):
                        if winner_for[k] == matched_winner:
                            winner_for[k] = new_winner
                    # Also rebind matched_winner itself
                    winner_for[matched_winner] = new_winner
                    # Update group's winners list: replace matched_winner
                    # with new_winner (preserve insertion order of clusters)
                    winners_in_group = [
                        new_winner if x == matched_winner else x
                        for x in winners_in_group
                    ]
                    winner_for[i] = new_winner
                else:
                    winner_for[i] = matched_winner

    # An episode survives iff it IS the winner for its cluster.
    survivors_set = {winner_for[i] for i in range(n)}
    out = [eps[i] for i in range(n) if i in survivors_set]
    n_dropped = n - len(out)
    return out, n_dropped
