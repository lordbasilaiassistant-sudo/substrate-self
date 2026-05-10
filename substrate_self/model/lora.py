"""Per-partner LoRA shards for substrate-self.

Solves two problems at once (from privacy_test_v1 results):

  1. Privacy — partner-A info physically lives in different parameters than
     partner-B info. When partner B is active, partner A's LoRA is NOT loaded,
     so partner-A specifics cannot leak into partner-B's conversation through
     the shared model.

  2. Catastrophic forgetting — partner B's online updates + sleep replay
     cannot overwrite partner A's knowledge, because they're not the same
     parameters. v0.3 had asymmetric leak (0/12 A/B) — that's forgetting,
     not discretion. LoRA shards solve both.

Architecture:

    base model.pt        — frozen Eli (partner-independent)
    partners/<id>.lora   — small low-rank delta per partner (~few thousand params)

At inference: set_active_partner(model, partner_id, partners_dir) loads that
partner's LoRA into the LoRALinear modules. Online updates train ONLY the
LoRA params (base stays frozen). Sleep replay consolidates ONLY the active
partner's episodes into the active partner's LoRA.

Default: rank=4, alpha=8 — small enough to keep storage tiny, expressive
enough to capture per-partner facts and style.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps a frozen base Linear with a low-rank delta: y = base(x) + (alpha/rank) * B @ A @ x.

    A is initialized small (kaiming-ish), B is initialized to zero, so the
    initial LoRA contribution is exactly zero — a freshly-introduced partner
    sees the unmodified base model.
    """

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.alpha = alpha

        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

        self.scale = alpha / rank
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.enabled:
            out = out + self.scale * F.linear(F.linear(x, self.lora_A), self.lora_B)
        return out

    def reset_lora(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)


_DEFAULT_TARGETS = ("c_attn", "c_proj")


def inject_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    target_names: tuple[str, ...] = _DEFAULT_TARGETS,
) -> int:
    """Replace nn.Linear children whose attribute name matches `target_names`
    with LoRALinear wrappers. Operates in-place on the model.

    Default targets are TinyGPT attention projections (c_attn, c_proj).
    Returns number of layers wrapped.
    """
    n_wrapped = 0
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear) and any(t == name or t in name for t in target_names):
            setattr(model, name, LoRALinear(module, rank=rank, alpha=alpha))
            n_wrapped += 1
        else:
            n_wrapped += inject_lora(module, rank=rank, alpha=alpha, target_names=target_names)
    return n_wrapped


def lora_modules(model: nn.Module) -> list[tuple[str, LoRALinear]]:
    return [(name, m) for name, m in model.named_modules() if isinstance(m, LoRALinear)]


def lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for _, m in lora_modules(model):
        yield m.lora_A
        yield m.lora_B


def freeze_base(model: nn.Module) -> None:
    """Freeze every parameter that is NOT a LoRA param. Call after inject_lora."""
    lora_param_ids = {id(p) for p in lora_parameters(model)}
    for p in model.parameters():
        if id(p) not in lora_param_ids:
            p.requires_grad = False


def extract_lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return state_dict containing ONLY the LoRA params, keyed by their
    fully-qualified module path. Safe to save per-partner."""
    out: dict[str, torch.Tensor] = {}
    for name, m in lora_modules(model):
        out[f"{name}.lora_A"] = m.lora_A.detach().cpu().clone()
        out[f"{name}.lora_B"] = m.lora_B.detach().cpu().clone()
    return out


def apply_lora_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load a previously-extracted LoRA state into the model. Missing keys
    are reset to init values; extra keys are ignored. Useful when the model
    was rebuilt fresh and we want to restore a partner's LoRA."""
    name_to_module = dict(lora_modules(model))
    for name, m in name_to_module.items():
        a_key = f"{name}.lora_A"
        b_key = f"{name}.lora_B"
        device = m.lora_A.device
        if a_key in state and b_key in state:
            m.lora_A.data.copy_(state[a_key].to(device))
            m.lora_B.data.copy_(state[b_key].to(device))
        else:
            m.reset_lora()


def save_partner_lora(
    model: nn.Module,
    partner_id: str,
    partners_dir: Path,
    metadata: Optional[dict] = None,
) -> Path:
    """Save the active LoRA state to partners/<id>.lora.

    File format: torch.save({"state": <state_dict>, "rank": int, "alpha": float, "meta": {...}}).
    """
    partners_dir = Path(partners_dir)
    partners_dir.mkdir(parents=True, exist_ok=True)
    out = partners_dir / f"{partner_id}.lora"
    state = extract_lora_state(model)
    mods = lora_modules(model)
    rank = mods[0][1].rank if mods else 0
    alpha = mods[0][1].alpha if mods else 0.0
    payload = {
        "state": state,
        "rank": rank,
        "alpha": alpha,
        "meta": metadata or {},
    }
    tmp = out.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)
    return out


def load_partner_lora(model: nn.Module, partner_id: str, partners_dir: Path) -> bool:
    """Load partners/<id>.lora into model. Returns True if a file was found
    and applied; False if no file existed (partner is fresh — LoRA reset to init).
    """
    path = Path(partners_dir) / f"{partner_id}.lora"
    if not path.exists():
        for _, m in lora_modules(model):
            m.reset_lora()
        return False
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["state"] if isinstance(payload, dict) and "state" in payload else payload
    apply_lora_state(model, state)
    return True


def set_active_partner(
    model: nn.Module,
    new_partner_id: str,
    partners_dir: Path,
    current_partner_id: Optional[str] = None,
) -> dict:
    """Atomic partner switch:
       1. If current_partner_id given, save its LoRA state first
       2. Load new partner's LoRA (or reset to init if no file)
    Returns {"saved": bool, "loaded": bool, "fresh": bool}.
    """
    saved = False
    if current_partner_id is not None:
        save_partner_lora(model, current_partner_id, partners_dir)
        saved = True
    loaded = load_partner_lora(model, new_partner_id, partners_dir)
    return {"saved": saved, "loaded": loaded, "fresh": not loaded}


def count_lora_params(model: nn.Module) -> int:
    return sum(p.numel() for p in lora_parameters(model))


def base_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a state_dict containing the BASE-MODEL parameters only,
    with LoRALinear wrappers re-keyed so the result is loadable into a
    plain (un-injected) model.

    Mapping: `<path>.base.<key>` -> `<path>.<key>` (drops the "base." segment).
    LoRA params (`lora_A`, `lora_B`) are dropped entirely.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        if k.endswith(".lora_A") or k.endswith(".lora_B"):
            continue
        if ".base." in k:
            new_key = k.replace(".base.", ".", 1)
            out[new_key] = v.detach().cpu().clone()
        else:
            out[k] = v.detach().cpu().clone()
    return out


def save_base_model(model: nn.Module, path) -> None:
    """Save the base model.pt without any LoRA params, in the original
    (un-injected) state_dict shape. Reloadable into a fresh TinyGPT(cfg)."""
    from pathlib import Path as _P
    p = _P(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    torch.save(base_state_dict(model), tmp)
    tmp.replace(p)
