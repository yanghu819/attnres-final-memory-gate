"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_PREPARE_IMPORT_ERROR = None
try:
    from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb
except ModuleNotFoundError as exc:
    _PREPARE_IMPORT_ERROR = exc
    MAX_SEQ_LEN = int(os.getenv("AUTORESEARCH_MAX_SEQ_LEN", "2048"))
    TIME_BUDGET = float(os.getenv("AUTORESEARCH_TIME_BUDGET", "600.0"))
    Tokenizer = None

    def _missing_prepare_dependency(*_args, **_kwargs):
        raise RuntimeError(
            "prepare.py dependencies are unavailable; install project dependencies "
            "and run the data preparation step before training or evaluation."
        ) from _PREPARE_IMPORT_ERROR

    make_dataloader = _missing_prepare_dependency
    evaluate_bpb = _missing_prepare_dependency


def env_int(name, default):
    value = os.getenv(name)
    return int(value) if value is not None else default


def env_float(name, default):
    value = os.getenv(name)
    return float(value) if value is not None else default


def env_str(name, default):
    value = os.getenv(name)
    return value if value is not None else default


def env_flag(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


ATTN_BACKEND = env_str("AUTORESEARCH_ATTN_BACKEND", "fa3")
USE_COMPILE = env_flag("AUTORESEARCH_USE_COMPILE", True)
TRAIN_TIME_BUDGET = env_float("AUTORESEARCH_TIME_BUDGET", TIME_BUDGET)
MAX_STEPS = env_int("AUTORESEARCH_MAX_STEPS", 0)
TOKEN_BUDGET = env_int("AUTORESEARCH_TOKEN_BUDGET", 0)

if MAX_STEPS > 0 and TOKEN_BUDGET > 0:
    raise ValueError("Set at most one of AUTORESEARCH_MAX_STEPS and AUTORESEARCH_TOKEN_BUDGET")

fa3 = None
if ATTN_BACKEND == "fa3":
    from kernels import get_kernel
    cap = torch.cuda.get_device_capability()
    # varunneal's FA3 is Hopper only, use kernels-community on non-Hopper GPUs
    repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
    try:
        fa3 = get_kernel(repo).flash_attn_interface
    except Exception as e:
        print(f"WARNING: failed to initialize flash-attn backend ({e}); falling back to sdpa")
        ATTN_BACKEND = "sdpa"

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    use_value_embeds: bool = True
    use_rotary: bool = True
    use_qk_norm: bool = True
    use_absolute_positions: bool = False
    norm_type: str = "rms"
    mlp_activation: str = "relu2"
    reference_init: bool = False
    reference_optimizer: bool = False
    tie_lm_head: bool = True
    lm_head_rotation_every: int = 0
    lm_head_rotation_max_positions: int = 64
    lm_head_rotation_buffer_rows: int = 2048
    lm_head_rotation_rank: int = 16
    lm_head_rotation_alpha: float = 0.05
    lm_head_rotation_normalize_rows: bool = True
    lm_head_rotation_buffer_device: str = "cpu"
    window_pattern: str = "SSSL"
    diffattn_mode: str = "off"
    diffattn_q2_mode: str = "wo"
    diffattn_kv_share: bool = True
    diffattn_lam_mode: str = "token"
    diffattn_lam_init: float = -2.0
    diffattn_lam_router_dim: int = 0
    diffattn_wo_init_std: float = 0.0
    moda_mode: str = "off"
    moda_history: int = 0
    moda_gate_init: float = -4.0
    moda_kv_mode: str = "shared"
    anchorkv_mode: str = "off"
    anchorkv_anchor_layer: int = 1
    anchorkv_gate_init: float = -5.0
    anchorkv_anchor_bias_init: float = -6.0
    attnres_mode: str = "off"
    attnres_block_size: int = 4
    attnres_eps: float = 1e-8
    attnres_weight_mode: str = "softmax"
    attnres_diff_lam_init: float = -2.0
    attnres_num_registers: int = 0
    attnres_register_mode: str = "learned"
    attnres_register_scope: str = "all"
    attnres_register_bias_init: float = -6.0
    attnres_register_bias_start: float = -6.0
    attnres_register_bias_target: float = -6.0
    attnres_register_bias_warmup_steps: int = 0
    attnres_register_init_std: float = 0.0
    attnres_history: int = 0
    attnres_gate_init: float = -2.0
    attnres_latest_bias_init: float = 4.0
    attnres_query_init_std: float = 0.0
    attnres_target_latest: float = 0.0
    attnres_target_latest_coef: float = 0.0
    attnres_target_register_layer_min: float = 0.0
    attnres_target_register_layer_max: float = 0.0
    attnres_target_register_final_min: float = 0.0
    attnres_target_register_final_max: float = 0.0
    attnres_target_register_coef: float = 0.0
    attnres_token_registers: int = 0
    attnres_token_register_mode: str = "off"
    attnres_token_register_bias_mode: str = "static"
    attnres_token_register_value_mode: str = "raw"
    attnres_token_register_scale: float = 1.0
    attnres_token_register_scale_start: float = 1.0
    attnres_token_register_scale_warmup_steps: int = 0
    attnres_token_register_delta_scale: float = 1.0
    attnres_token_register_weight_cap: float = 0.0
    attnres_token_register_spread_coef: float = 0.0
    attnres_token_register_spread_samples: int = 0
    attnres_token_query_mode: str = "off"
    attnres_token_query_value_mode: str = "raw"
    attnres_token_query_scale: float = 1.0
    attnres_final_memory_mode: str = "off"
    attnres_final_memory_source: str = "full"
    attnres_final_memory_value_mode: str = "raw"
    attnres_final_memory_scale: float = 1.0
    attnres_final_memory_groups: int = 1
    attnres_final_memory_rank: int = 0
    attnres_final_memory_residual_rank: int = 0
    attnres_final_memory_residual_scale: float = 1.0
    attnres_final_memory_bigram_buckets: int = 0
    attnres_final_memory_bigram_banks: int = 1
    attnres_final_memory_bigram_scale: float = 1.0
    attnres_final_memory_gate_cap: float = 0.0
    attnres_final_memory_gate_bias_init: float = 0.0
    attnres_final_memory_delta_scale: float = 1.0
    attnres_final_memory_query_cap: float = 0.0
    attnres_final_memory_query_bias_init: float = 0.0
    attnres_final_memory_query_delta_scale: float = 1.0
    attnres_final_memory_q_mod_scale: float = 0.0
    attnres_final_memory_k_mod_scale: float = 0.0
    attnres_final_memory_v_mod_scale: float = 0.0
    attnres_final_memory_logit_mod_scale: float = 0.0
    attnres_final_memory_vres_scale: float = 0.0
    attnres_input_memory_mode: str = "off"
    attnres_input_memory_value_mode: str = "rmsnorm"
    attnres_input_memory_scale: float = 1.0
    attnres_input_memory_hash_dim: int = 64
    attnres_input_memory_bigram_buckets: int = 0
    attnres_input_memory_bigram_banks: int = 1
    attnres_input_memory_bigram_scale: float = 1.0
    attnres_input_memory_trigram_buckets: int = 0
    attnres_input_memory_trigram_banks: int = 1
    attnres_input_memory_trigram_scale: float = 1.0
    attnres_input_memory_smear: bool = False
    attnres_input_memory_smear_bias_init: float = -4.0
    attnres_target_gate: float = 0.0
    attnres_target_gate_coef: float = 0.0
    permix_mode: str = "off"
    permix_skip_detach: bool = True
    permix_skip_ema: float = 0.5
    permix_identity_bias: float = 2.0
    permix_router_dim: int = 0
    permix_mix_strength: float = 1.0
    permix_router_temperature: float = 1.0
    permix_target_raw_nonid: float = 0.0
    permix_target_raw_nonid_coef: float = 0.0
    permix_entropy_coef: float = 0.0
    permix_aux_scale: float = 1.0
    segattn_mode: str = "off"
    segattn_num_segments: int = 4
    segattn_proj_dim: int = 0
    segattn_gate_init: float = -2.0
    segattn_gate_cap: float = 1.0
    segattn_start_layer: int = 0


def sample_logit_grads(logits, targets, max_positions=64, ignore_index=-1, normalize_rows=True):
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    mask = flat_targets.ne(ignore_index)
    if not mask.any():
        return None
    flat_logits = flat_logits[mask]
    flat_targets = flat_targets[mask]
    n = flat_logits.size(0)
    if max_positions > 0 and n > max_positions:
        idx = torch.randperm(n, device=flat_logits.device)[:max_positions]
        flat_logits = flat_logits[idx]
        flat_targets = flat_targets[idx]
    probs = torch.softmax(flat_logits.float(), dim=-1)
    grads = probs.clone()
    grads[torch.arange(grads.size(0), device=grads.device), flat_targets] -= 1.0
    if normalize_rows:
        grads = grads / grads.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return grads.detach()


class LogitGradBuffer:
    def __init__(self, max_rows=2048, device="cpu", dtype=torch.float16):
        self.max_rows = max_rows
        self.device = device
        self.dtype = dtype
        self.chunks: List[torch.Tensor] = []
        self.n_rows = 0

    def add(self, grads):
        if grads is None or grads.numel() == 0:
            return
        grads = grads.to(device=self.device, dtype=self.dtype)
        self.chunks.append(grads)
        self.n_rows += grads.size(0)
        while self.n_rows > self.max_rows and self.chunks:
            overflow = self.n_rows - self.max_rows
            first = self.chunks[0]
            if first.size(0) <= overflow:
                self.n_rows -= first.size(0)
                self.chunks.pop(0)
            else:
                self.chunks[0] = first[overflow:]
                self.n_rows -= overflow

    def clear(self):
        self.chunks = []
        self.n_rows = 0

    def as_matrix(self, device):
        if self.n_rows <= 0:
            raise RuntimeError("LM head gradient buffer is empty")
        return torch.cat(self.chunks, dim=0).to(device=device, dtype=torch.float32)


@torch.no_grad()
def lm_head_lost_fraction(grads, weight, eps=1e-12):
    q_basis, _ = torch.linalg.qr(weight.float(), mode="reduced")
    grads = grads.float()
    lost = grads - (grads @ q_basis) @ q_basis.T
    return float((lost.norm() ** 2 / grads.norm().clamp_min(eps) ** 2).item())


@torch.no_grad()
def rotate_lm_head_towards_lost_dirs_(lm_head, grads, rank=16, alpha=0.05, optimizer=None):
    weight = lm_head.weight.detach().float()
    vocab_size, d_model = weight.shape
    rank = min(rank, d_model - 1, grads.size(0), vocab_size)
    if rank <= 0:
        return None

    q_basis, r_factor = torch.linalg.qr(weight, mode="reduced")
    grads = grads.to(device=weight.device, dtype=torch.float32)
    grads_null = grads - (grads @ q_basis) @ q_basis.T
    lost_before = float((grads_null.norm() ** 2 / grads.norm().clamp_min(1e-12) ** 2).item())

    svd_q = min(rank + 8, min(grads_null.shape))
    if svd_q <= 0:
        return None
    _, singular_vals_null, right_vecs = torch.svd_lowrank(grads_null, q=svd_q, niter=2)
    del singular_vals_null
    target_dirs = right_vecs[:, :rank]

    u_r, singular_vals, vh = torch.linalg.svd(r_factor, full_matrices=False)
    current_dirs = q_basis @ u_r
    keep = d_model - rank
    keep_dirs = current_dirs[:, :keep]
    tail_dirs = current_dirs[:, keep:]

    candidate_dirs = target_dirs - keep_dirs @ (keep_dirs.T @ target_dirs)
    pool = torch.cat([candidate_dirs, tail_dirs], dim=1)
    pool = pool - keep_dirs @ (keep_dirs.T @ pool)
    add_dirs, _ = torch.linalg.qr(pool, mode="reduced")
    add_dirs = add_dirs[:, :rank]

    mixed_tail = (1.0 - alpha) * tail_dirs + alpha * add_dirs
    mixed_tail = mixed_tail - keep_dirs @ (keep_dirs.T @ mixed_tail)
    mixed_tail, _ = torch.linalg.qr(mixed_tail, mode="reduced")

    new_dirs = torch.cat([keep_dirs, mixed_tail], dim=1)
    new_dirs, _ = torch.linalg.qr(new_dirs, mode="reduced")
    new_weight = new_dirs @ (torch.diag(singular_vals) @ vh)
    lm_head.weight.copy_(new_weight.to(dtype=lm_head.weight.dtype))

    if optimizer is not None and lm_head.weight in optimizer.state:
        state = optimizer.state[lm_head.weight]
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(key)
            if torch.is_tensor(value):
                value.zero_()

    lost_after = lm_head_lost_fraction(grads, lm_head.weight.detach())
    return {
        "lost_before": lost_before,
        "lost_after": lost_after,
        "rows_used": int(grads.size(0)),
        "rank": int(rank),
        "alpha": float(alpha),
    }


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class SimpleLayerNorm(nn.Module):
    def __init__(self, ndim, bias=False, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, self.eps)


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def repeat_kv_heads(x, n_head):
    if x.size(2) == n_head:
        return x
    assert n_head % x.size(2) == 0
    repeat = n_head // x.size(2)
    return x.repeat_interleave(repeat, dim=2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.use_rotary = config.use_rotary
        self.use_qk_norm = config.use_qk_norm
        self.use_value_embeds = config.use_value_embeds
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.moda_mode = config.moda_mode
        self.moda_enabled = self.moda_mode != "off"
        self.moda_history = config.moda_history
        self.moda_gate_init = config.moda_gate_init
        self.moda_kv_mode = config.moda_kv_mode
        if self.moda_enabled:
            assert self.moda_kv_mode in {"shared", "separate"}, "moda_kv_mode must be shared or separate"
            self.moda_gate = nn.Parameter(torch.zeros(()))
            if self.moda_kv_mode == "separate":
                self.c_mk = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
                self.c_mv = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
            else:
                self.c_mk = None
                self.c_mv = None
            self._moda_mask_cache = {}
            self.last_moda_cache = None
        else:
            self.moda_gate = None
            self.c_mk = None
            self.c_mv = None
            self._moda_mask_cache = None
            self.last_moda_cache = None
        self.anchorkv_mode = config.anchorkv_mode
        self.anchorkv_enabled = self.anchorkv_mode != "off"
        self.anchorkv_gate_init = config.anchorkv_gate_init
        self.anchorkv_anchor_bias_init = config.anchorkv_anchor_bias_init
        if self.anchorkv_enabled:
            self.anchorkv_gate = nn.Parameter(torch.zeros(()))
            self.anchorkv_logit_bias = nn.Parameter(torch.zeros(()))
            self.last_anchorkv_cache = None
        else:
            self.anchorkv_gate = None
            self.anchorkv_logit_bias = None
            self.last_anchorkv_cache = None
        self.diffattn_mode = config.diffattn_mode
        self.diffattn_enabled = self.diffattn_mode != "off"
        self.diffattn_q2_mode = config.diffattn_q2_mode
        self.diffattn_kv_share = config.diffattn_kv_share
        self.diffattn_lam_mode = config.diffattn_lam_mode
        self.diffattn_lam_init = config.diffattn_lam_init
        self.diffattn_wo_init_std = config.diffattn_wo_init_std
        if self.diffattn_enabled:
            assert self.diffattn_q2_mode == "wo", "only q2_mode=wo is supported in this branch"
            assert self.diffattn_kv_share, "this branch only supports kv_share=True"
            assert self.diffattn_lam_mode in {"token", "head"}, "diffattn_lam_mode must be token or head"
            lam_out_dim = 1 if self.diffattn_lam_mode == "token" else self.n_head
            if config.diffattn_lam_router_dim > 0:
                self.lam_proj = nn.Sequential(
                    nn.Linear(self.n_embd, config.diffattn_lam_router_dim, bias=False),
                    nn.GELU(),
                    nn.Linear(config.diffattn_lam_router_dim, lam_out_dim, bias=False),
                )
            else:
                self.lam_proj = nn.Linear(self.n_embd, lam_out_dim, bias=False)
        else:
            self.lam_proj = None
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=config.bias) if self.use_value_embeds and has_ve(layer_idx, config.n_layer) else None
        self.last_diffattn_stats = None
        self.last_moda_stats = None
        self.last_anchorkv_stats = None

    def _get_moda_mask(self, T, num_blocks, device):
        key = (T, num_blocks, device.type, getattr(device, "index", None))
        mask = self._moda_mask_cache.get(key)
        if mask is None:
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
            mask = causal.repeat(1, num_blocks)
            self._moda_mask_cache[key] = mask
        return mask

    def forward(self, x, ve, cos_sin, window_size, moda_caches=None, anchor_cache=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        q2 = None
        q2_raw = None
        lam = None

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        if self.use_rotary:
            cos, sin = cos_sin
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        if self.use_qk_norm:
            q, k = norm(q), norm(k)
        if self.diffattn_enabled:
            q2_raw = F.linear(x, self.c_proj.weight).view(B, T, self.n_head, self.head_dim)
            q2 = q2_raw
            if self.use_rotary:
                q2 = apply_rotary_emb(q2, cos, sin)
            if self.use_qk_norm:
                q2 = norm(q2)
            lam_logits = self.lam_proj(x.float()) + self.diffattn_lam_init
            if self.diffattn_lam_mode == "token":
                lam = torch.sigmoid(lam_logits).view(B, T, 1, 1)
            else:
                lam = torch.sigmoid(lam_logits).view(B, T, self.n_head, 1)

        if self.moda_enabled and moda_caches and ATTN_BACKEND != "sdpa":
            raise RuntimeError("MoDA prototype currently requires AUTORESEARCH_ATTN_BACKEND=sdpa")
        if self.anchorkv_enabled and anchor_cache is not None and ATTN_BACKEND != "sdpa":
            raise RuntimeError("Anchor-KV prototype currently requires AUTORESEARCH_ATTN_BACKEND=sdpa")

        if ATTN_BACKEND == "fa3":
            y1 = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
            if self.diffattn_enabled:
                y2 = fa3.flash_attn_func(q2, k, v, causal=True, window_size=window_size)
                y = y1 - lam.to(dtype=y1.dtype) * y2
            else:
                y = y1
            y = y.contiguous().view(B, T, -1)
            self.last_moda_stats = None
            self.last_moda_cache = None
            self.last_anchorkv_stats = None
            self.last_anchorkv_cache = None
        else:
            if window_size[0] < T:
                raise RuntimeError("sdpa backend only supports full attention; set AUTORESEARCH_WINDOW_PATTERN=L")
            k = repeat_kv_heads(k, self.n_head)
            v = repeat_kv_heads(v, self.n_head)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            y1 = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
            if self.diffattn_enabled:
                q2 = q2.transpose(1, 2)
                y2 = F.scaled_dot_product_attention(
                    q2, k, v, is_causal=True,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                )
                y = y1 - lam.transpose(1, 2).to(dtype=y1.dtype) * y2
            else:
                y = y1
            if self.moda_enabled and self.c_mk is not None and self.c_mv is not None:
                mk = self.c_mk(x).view(B, T, self.n_kv_head, self.head_dim)
                mv = self.c_mv(x).view(B, T, self.n_kv_head, self.head_dim)
                mk = apply_rotary_emb(mk, cos, sin)
                mk = norm(mk)
                current_cache_k = repeat_kv_heads(mk, self.n_head).transpose(1, 2)
                current_cache_v = repeat_kv_heads(mv, self.n_head).transpose(1, 2)
            else:
                current_cache_k = k
                current_cache_v = v
            if self.anchorkv_enabled and anchor_cache is not None:
                anchor_k, anchor_v = anchor_cache
                scale = self.head_dim ** -0.5
                seq_scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
                causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
                seq_scores = seq_scores.masked_fill(~causal, float("-inf"))
                anchor_scores = torch.matmul(q.float(), anchor_k.float().transpose(-2, -1)) * scale
                anchor_scores = anchor_scores + (self.anchorkv_logit_bias.float() + self.anchorkv_anchor_bias_init)
                anchor_scores = anchor_scores.masked_fill(~causal, float("-inf"))
                all_scores = torch.cat([seq_scores, anchor_scores], dim=-1)
                all_weights = F.softmax(all_scores, dim=-1).to(dtype=v.dtype)
                w_seq = all_weights[..., :T]
                w_anchor = all_weights[..., T:]
                ctx_seq = torch.matmul(w_seq, v)
                ctx_anchor = torch.matmul(w_anchor, anchor_v)
                y_all = ctx_seq + ctx_anchor
                anchorkv_gate = torch.sigmoid(self.anchorkv_gate.float() + self.anchorkv_gate_init)
                y = y + anchorkv_gate.to(dtype=y.dtype) * (y_all - y)
                self.last_anchorkv_stats = {
                    "gate": anchorkv_gate.detach(),
                    "bias": (self.anchorkv_logit_bias.float() + self.anchorkv_anchor_bias_init).detach(),
                    "anchor_mass": w_anchor.float().sum(dim=-1).mean().detach(),
                    "delta_norm": (y_all.float() - y1.float()).norm(dim=-1).mean().detach(),
                }
            else:
                self.last_anchorkv_stats = None
            if self.moda_enabled and moda_caches:
                depth_k = torch.stack([mk for mk, _ in moda_caches], dim=3)  # [B, H, T, L, D]
                depth_v = torch.stack([mv for _, mv in moda_caches], dim=3)  # [B, H, T, L, D]
                scale = self.head_dim ** -0.5
                seq_scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
                causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
                seq_scores = seq_scores.masked_fill(~causal, float("-inf"))
                depth_scores = (q.float().unsqueeze(3) * depth_k.float()).sum(dim=-1) * scale  # [B,H,T,L]
                all_scores = torch.cat([seq_scores, depth_scores], dim=-1)
                all_weights = F.softmax(all_scores, dim=-1).to(dtype=v.dtype)
                w_seq = all_weights[..., :T]
                w_depth = all_weights[..., T:]
                ctx_seq = torch.matmul(w_seq, v)
                ctx_depth = (w_depth.unsqueeze(-1) * depth_v).sum(dim=3)
                y_all = ctx_seq + ctx_depth
                moda_gate = torch.sigmoid(self.moda_gate.float() + self.moda_gate_init)
                y = y + moda_gate.to(dtype=y.dtype) * (y_all - y)
                self.last_moda_stats = {
                    "gate": moda_gate.detach(),
                    "history": torch.tensor(float(depth_k.shape[3]), device=x.device).detach(),
                    "delta_norm": (y_all.float() - y1.float()).norm(dim=-1).mean().detach(),
                    "depth_mass": w_depth.float().sum(dim=-1).mean().detach(),
                }
            else:
                self.last_moda_stats = None
            self.last_moda_cache = (current_cache_k, current_cache_v)
            self.last_anchorkv_cache = (current_cache_k, current_cache_v)
            y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.resid_dropout(self.c_proj(y))
        if self.diffattn_enabled:
            lam_f = lam.float()
            self.last_diffattn_stats = {
                "lam_mean": lam_f.mean().detach(),
                "lam_max": lam_f.max().detach(),
                "lam_min": lam_f.min().detach(),
                "q2_raw_norm": q2_raw.float().norm(dim=-1).mean().detach(),
                "q2_norm": q2.float().norm(dim=-1).mean().detach(),
            }
        else:
            self.last_diffattn_stats = None
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation = config.mlp_activation
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        if self.activation == "gelu":
            x = F.gelu(x)
        else:
            x = F.relu(x).square()
        x = self.c_proj(x)
        return self.dropout(x)


class ReferenceRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        compute_dtype = x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
        rms = x.to(compute_dtype).pow(2).mean(dim=-1, keepdim=True)
        scale = torch.rsqrt(rms + self.eps).to(dtype=x.dtype)
        return x * scale * self.weight.to(dtype=x.dtype)


class DepthSoftmaxMixer(nn.Module):
    def __init__(
        self,
        num_queries,
        dim,
        eps=1e-8,
        diff_lam_init=-2.0,
        num_registers=0,
        register_mode="learned",
        register_bias_init=-6.0,
        register_bias_start=-6.0,
        register_bias_target=-6.0,
        register_bias_warmup_steps=0,
        register_init_std=0.0,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.dim = dim
        self.key_norms = nn.ModuleList([ReferenceRMSNorm(dim=dim, eps=eps) for _ in range(num_queries)])
        self.queries = nn.Parameter(torch.zeros(num_queries, dim))
        self.queries2 = nn.Parameter(torch.zeros(num_queries, dim))
        self.lam_proj = nn.Linear(dim, num_queries, bias=False)
        self.eps = eps
        self.diff_lam_init = diff_lam_init
        self.num_registers = num_registers
        self.register_mode = register_mode
        self.register_bias_init = register_bias_init
        self.register_bias_start = register_bias_start
        self.register_bias_target = register_bias_target
        self.register_bias_warmup_steps = register_bias_warmup_steps
        self.register_init_std = register_init_std
        self.runtime_register_bias_offset = 0.0
        if self.register_mode not in {"learned", "query_learned", "null", "summary", "recent", "summary_null"}:
            raise ValueError(f"unknown attnres register_mode: {self.register_mode}")
        if num_registers > 0 and self.register_mode == "learned":
            self.registers = nn.Parameter(torch.zeros(num_registers, dim))
        elif num_registers > 0 and self.register_mode == "query_learned":
            self.registers = nn.Parameter(torch.zeros(num_queries, num_registers, dim))
        else:
            self.registers = None
        if num_registers > 0:
            self.register_bias = nn.Parameter(torch.zeros(num_queries, num_registers))
        else:
            self.register_bias = None

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.queries)
        nn.init.zeros_(self.queries2)
        nn.init.zeros_(self.lam_proj.weight)
        if self.register_mode in {"learned", "query_learned"} and self.registers is not None:
            if self.register_init_std > 0:
                nn.init.normal_(self.registers, mean=0.0, std=self.register_init_std)
            else:
                nn.init.zeros_(self.registers)
        if self.register_bias is not None:
            self.register_bias.fill_(self.register_bias_target)

    def set_runtime_step(self, step):
        if self.registers is None or self.register_bias_warmup_steps <= 0:
            self.runtime_register_bias_offset = 0.0
            return
        frac = min(max(float(step), 0.0) / float(self.register_bias_warmup_steps), 1.0)
        scheduled_bias = self.register_bias_start + frac * (self.register_bias_target - self.register_bias_start)
        self.runtime_register_bias_offset = scheduled_bias - self.register_bias_target

    def _normalize_weights(self, logits, source_tensor, latest_source, query_index, weight_mode):
        if weight_mode == "softmax":
            return F.softmax(logits, dim=0)
        if weight_mode == "sigmoid_norm":
            gates = torch.sigmoid(logits)
            return gates / gates.sum(dim=0, keepdim=True).clamp_min(self.eps)
        if weight_mode == "diffv2":
            normed_sources = self.key_norms[query_index](source_tensor)
            query2 = self.queries2[query_index].to(dtype=source_tensor.dtype)
            logits2 = torch.einsum("d,s...d->s...", query2, normed_sources)
            weights1 = F.softmax(logits, dim=0)
            weights2 = F.softmax(logits2, dim=0)
            lam = torch.sigmoid(self.lam_proj(latest_source)[..., query_index] + self.diff_lam_init)
            return weights1 - lam.unsqueeze(0).to(dtype=weights1.dtype) * weights2
        raise ValueError(f"unknown attnres weight_mode: {weight_mode}")

    def _build_register_tensor(self, source_tensor, num_real_sources, query_index):
        if self.num_registers <= 0:
            return None
        if self.register_mode == "learned":
            register_tensor = self.registers.to(dtype=source_tensor.dtype)
            expand_shape = (self.num_registers, *source_tensor.shape[1:-1], self.dim)
            return register_tensor.view(
                self.num_registers, *([1] * (source_tensor.dim() - 2)), self.dim
            ).expand(expand_shape)
        if self.register_mode == "query_learned":
            register_tensor = self.registers[query_index].to(dtype=source_tensor.dtype)
            expand_shape = (self.num_registers, *source_tensor.shape[1:-1], self.dim)
            return register_tensor.view(
                self.num_registers, *([1] * (source_tensor.dim() - 2)), self.dim
            ).expand(expand_shape)
        if self.register_mode == "null":
            return source_tensor.new_zeros((self.num_registers, *source_tensor.shape[1:]))
        if self.register_mode == "summary":
            base = source_tensor[:num_real_sources].mean(dim=0, keepdim=True)
            return base.expand(self.num_registers, *base.shape[1:])
        if self.register_mode == "recent":
            recent = source_tensor[max(0, num_real_sources - 2):num_real_sources]
            base = recent.mean(dim=0, keepdim=True)
            return base.expand(self.num_registers, *base.shape[1:])
        if self.register_mode == "summary_null":
            recent = source_tensor[max(0, num_real_sources - 2):num_real_sources]
            base = recent.mean(dim=0)
            registers = source_tensor.new_zeros((self.num_registers, *source_tensor.shape[1:]))
            registers[0] = base
            return registers
        raise ValueError(f"unknown attnres register_mode: {self.register_mode}")

    def mix_with_weights(
        self,
        query_index,
        source_values,
        weight_mode="softmax",
        include_registers=True,
        extra_registers=None,
        extra_register_bias=None,
        extra_query=None,
        source_logit_bias=None,
        source_key_scale=None,
        source_value_scale=None,
        source_value_residual=None,
        extra_register_weight_cap=0.0,
    ):
        source_tensor = torch.stack(list(source_values), dim=0) if isinstance(source_values, (list, tuple)) else source_values
        num_real_sources = source_tensor.size(0)
        latest_source = source_tensor[num_real_sources - 1].float()
        active_registers = include_registers and self.registers is not None
        extra_register_count = 0
        if include_registers and self.num_registers > 0:
            active_registers = True
            register_tensor = self._build_register_tensor(source_tensor, num_real_sources, query_index)
            source_tensor = torch.cat([source_tensor, register_tensor], dim=0)
        if include_registers and extra_registers is not None:
            active_registers = True
            extra_register_count = int(extra_registers.size(0))
            source_tensor = torch.cat([source_tensor, extra_registers.to(dtype=source_tensor.dtype)], dim=0)
        normed_sources = self.key_norms[query_index](source_tensor)
        value_residual_base = normed_sources.to(dtype=source_tensor.dtype)
        mixed_source_tensor = source_tensor
        if source_key_scale is not None:
            key_scale = source_key_scale.to(dtype=normed_sources.dtype)
            normed_sources = normed_sources * key_scale.unsqueeze(0)
        if source_value_scale is not None:
            value_scale = source_value_scale.to(dtype=source_tensor.dtype)
            mixed_source_tensor = mixed_source_tensor * value_scale.unsqueeze(0)
        if source_value_residual is not None:
            value_residual = source_value_residual.to(dtype=source_tensor.dtype)
            mixed_source_tensor = mixed_source_tensor + value_residual_base * value_residual.unsqueeze(0)
        query = self.queries[query_index].to(dtype=source_tensor.dtype)
        if extra_query is not None:
            query = query.view(*([1] * (extra_query.dim() - 1)), self.dim) + extra_query.to(dtype=source_tensor.dtype)
            logits = torch.einsum("...d,s...d->s...", query, normed_sources)
        else:
            logits = torch.einsum("d,s...d->s...", query, normed_sources)
        if source_logit_bias is not None:
            logit_bias = source_logit_bias.to(dtype=logits.dtype)
            if logit_bias.dim() == 1:
                view_shape = (num_real_sources,) + (1,) * (logits.dim() - 1)
                logit_bias = logit_bias[:num_real_sources].view(view_shape)
            else:
                logit_bias = logit_bias[:num_real_sources]
            logits = logits.clone()
            logits[:num_real_sources] = logits[:num_real_sources] + logit_bias
        if active_registers and self.register_bias is not None:
            reg_bias = self.register_bias[query_index].to(dtype=logits.dtype)
            if self.runtime_register_bias_offset != 0.0:
                reg_bias = reg_bias + logits.new_tensor(self.runtime_register_bias_offset)
            view_shape = (self.num_registers,) + (1,) * (logits.dim() - 1)
            logits = logits.clone()
            logits[num_real_sources:num_real_sources + self.num_registers] = (
                logits[num_real_sources:num_real_sources + self.num_registers] + reg_bias.view(view_shape)
            )
        if include_registers and extra_register_count > 0 and extra_register_bias is not None:
            extra_bias = extra_register_bias.to(dtype=logits.dtype)
            start = num_real_sources + (self.num_registers if self.num_registers > 0 else 0)
            logits = logits.clone()
            if extra_bias.dim() == 1:
                view_shape = (extra_register_count,) + (1,) * (logits.dim() - 1)
                extra_bias = extra_bias.view(view_shape)
            logits[start:start + extra_register_count] = logits[start:start + extra_register_count] + extra_bias
        weights = self._normalize_weights(
            logits,
            source_tensor=source_tensor,
            latest_source=latest_source,
            query_index=query_index,
            weight_mode=weight_mode,
        )
        if include_registers and extra_register_count > 0 and extra_register_weight_cap > 0:
            start = num_real_sources + (self.num_registers if include_registers and self.num_registers > 0 else 0)
            end = start + extra_register_count
            extra_weights = weights[start:end]
            extra_total = extra_weights.sum(dim=0, keepdim=True)
            cap = extra_total.new_tensor(extra_register_weight_cap)
            limited_extra_total = torch.minimum(extra_total, cap)
            extra_scale = limited_extra_total / extra_total.clamp_min(self.eps)
            scaled_extra = extra_weights * extra_scale

            before = weights[:start]
            after = weights[end:]
            non_extra_total = before.sum(dim=0, keepdim=True)
            if after.numel() > 0:
                non_extra_total = non_extra_total + after.sum(dim=0, keepdim=True)
            target_non_extra_total = weights.new_ones(extra_total.shape) - scaled_extra.sum(dim=0, keepdim=True)
            non_extra_scale = target_non_extra_total / non_extra_total.clamp_min(self.eps)
            pieces = []
            if before.numel() > 0:
                pieces.append(before * non_extra_scale)
                pieces.append(scaled_extra)
            if after.numel() > 0:
                pieces.append(after * non_extra_scale)
            weights = torch.cat(pieces, dim=0)
        mixed = torch.einsum("s...,s...d->...d", weights, mixed_source_tensor)
        return mixed, weights, {
            "num_real_sources": num_real_sources,
            "num_builtin_registers": self.num_registers if include_registers else 0,
            "extra_register_count": extra_register_count,
            "num_registers": ((self.num_registers if include_registers else 0) + extra_register_count) if active_registers else 0,
        }


class FullAttnResMixer(nn.Module):
    def __init__(
        self,
        num_logical_layers,
        dim,
        eps=1e-8,
        weight_mode="softmax",
        diff_lam_init=-2.0,
        num_registers=0,
        register_mode="learned",
        register_bias_init=-6.0,
        register_bias_start=-6.0,
        register_bias_target=-6.0,
        register_bias_warmup_steps=0,
        register_init_std=0.0,
    ):
        super().__init__()
        self.num_logical_layers = num_logical_layers
        self.mixer = DepthSoftmaxMixer(
            num_queries=num_logical_layers + 1,
            dim=dim,
            eps=eps,
            diff_lam_init=diff_lam_init,
            num_registers=num_registers,
            register_mode=register_mode,
            register_bias_init=register_bias_init,
            register_bias_start=register_bias_start,
            register_bias_target=register_bias_target,
            register_bias_warmup_steps=register_bias_warmup_steps,
            register_init_std=register_init_std,
        )
        self.weight_mode = weight_mode

    def set_runtime_step(self, step):
        self.mixer.set_runtime_step(step)

    def layer_input_with_weights(
        self,
        embedding,
        prior_layer_outputs,
        layer_index,
        include_registers=True,
        extra_registers=None,
        extra_register_bias=None,
        extra_query=None,
        source_logit_bias=None,
        source_key_scale=None,
        source_value_scale=None,
        source_value_residual=None,
        extra_register_weight_cap=0.0,
    ):
        sources = [embedding, *prior_layer_outputs]
        return self.mixer.mix_with_weights(
            query_index=layer_index,
            source_values=sources,
            weight_mode=self.weight_mode,
            include_registers=include_registers,
            extra_registers=extra_registers,
            extra_register_bias=extra_register_bias,
            extra_query=extra_query,
            source_logit_bias=source_logit_bias,
            source_key_scale=source_key_scale,
            source_value_scale=source_value_scale,
            source_value_residual=source_value_residual,
            extra_register_weight_cap=extra_register_weight_cap,
        )

    def final_output_with_weights(
        self,
        embedding,
        layer_outputs,
        include_registers=True,
        extra_registers=None,
        extra_register_bias=None,
        extra_query=None,
        source_logit_bias=None,
        source_key_scale=None,
        source_value_scale=None,
        source_value_residual=None,
        extra_register_weight_cap=0.0,
    ):
        sources = [embedding, *layer_outputs]
        return self.mixer.mix_with_weights(
            query_index=self.num_logical_layers,
            source_values=sources,
            weight_mode=self.weight_mode,
            include_registers=include_registers,
            extra_registers=extra_registers,
            extra_register_bias=extra_register_bias,
            extra_query=extra_query,
            source_logit_bias=source_logit_bias,
            source_key_scale=source_key_scale,
            source_value_scale=source_value_scale,
            source_value_residual=source_value_residual,
            extra_register_weight_cap=extra_register_weight_cap,
        )


class BlockAttnResState:
    def __init__(self, embedding, block_size):
        self.completed_blocks = [embedding]
        self.partial_block = None
        self.completed_layers = 0
        self.block_size = block_size

    def current_sources(self):
        if self.partial_block is None:
            return list(self.completed_blocks)
        return [*self.completed_blocks, self.partial_block]

    def append_layer_output(self, layer_output):
        if self.partial_block is None:
            self.partial_block = layer_output
        else:
            self.partial_block = self.partial_block + layer_output
        self.completed_layers += 1
        if self.completed_layers % self.block_size == 0:
            self.completed_blocks.append(self.partial_block)
            self.partial_block = None

    def final_sources(self):
        sources = list(self.completed_blocks)
        if self.partial_block is not None:
            sources.append(self.partial_block)
        return sources


class BlockAttnResMixer(nn.Module):
    def __init__(
        self,
        num_logical_layers,
        dim,
        block_size,
        eps=1e-8,
        weight_mode="softmax",
        diff_lam_init=-2.0,
        num_registers=0,
        register_mode="learned",
        register_bias_init=-6.0,
        register_bias_start=-6.0,
        register_bias_target=-6.0,
        register_bias_warmup_steps=0,
        register_init_std=0.0,
    ):
        super().__init__()
        self.num_logical_layers = num_logical_layers
        self.block_size = block_size
        self.mixer = DepthSoftmaxMixer(
            num_queries=num_logical_layers + 1,
            dim=dim,
            eps=eps,
            diff_lam_init=diff_lam_init,
            num_registers=num_registers,
            register_mode=register_mode,
            register_bias_init=register_bias_init,
            register_bias_start=register_bias_start,
            register_bias_target=register_bias_target,
            register_bias_warmup_steps=register_bias_warmup_steps,
            register_init_std=register_init_std,
        )
        self.weight_mode = weight_mode

    def init_state(self, embedding):
        return BlockAttnResState(embedding=embedding, block_size=self.block_size)

    def set_runtime_step(self, step):
        self.mixer.set_runtime_step(step)

    def layer_input_with_weights(
        self,
        state,
        layer_index,
        include_registers=True,
        extra_registers=None,
        extra_register_bias=None,
        extra_query=None,
        source_logit_bias=None,
        source_key_scale=None,
        source_value_scale=None,
        source_value_residual=None,
        extra_register_weight_cap=0.0,
    ):
        return self.mixer.mix_with_weights(
            query_index=layer_index,
            source_values=state.current_sources(),
            weight_mode=self.weight_mode,
            include_registers=include_registers,
            extra_registers=extra_registers,
            extra_register_bias=extra_register_bias,
            extra_query=extra_query,
            source_logit_bias=source_logit_bias,
            source_key_scale=source_key_scale,
            source_value_scale=source_value_scale,
            source_value_residual=source_value_residual,
            extra_register_weight_cap=extra_register_weight_cap,
        )

    def append_layer_output(self, state, layer_output):
        state.append_layer_output(layer_output)

    def final_output_with_weights(
        self,
        state,
        include_registers=True,
        extra_registers=None,
        extra_register_bias=None,
        extra_query=None,
        source_logit_bias=None,
        source_key_scale=None,
        source_value_scale=None,
        source_value_residual=None,
        extra_register_weight_cap=0.0,
    ):
        return self.mixer.mix_with_weights(
            query_index=self.num_logical_layers,
            source_values=state.final_sources(),
            weight_mode=self.weight_mode,
            include_registers=include_registers,
            extra_registers=extra_registers,
            extra_register_bias=extra_register_bias,
            extra_query=extra_query,
            source_logit_bias=source_logit_bias,
            source_key_scale=source_key_scale,
            source_value_scale=source_value_scale,
            source_value_residual=source_value_residual,
            extra_register_weight_cap=extra_register_weight_cap,
        )


class AttentionResidual(nn.Module):
    """Depth-wise softmax aggregation over prior hidden states."""

    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.gate_init = config.attnres_gate_init
        self.latest_bias_init = config.attnres_latest_bias_init
        self.query_init_std = config.attnres_query_init_std
        self.query = nn.Parameter(torch.zeros(self.n_embd))
        self.latest_bias = nn.Parameter(torch.zeros(()))
        self.gate = nn.Parameter(torch.zeros(()))
        self.scale = self.n_embd ** -0.5
        self.last_stats = None

    def init_weights(self):
        if self.query_init_std > 0:
            nn.init.normal_(self.query, mean=0.0, std=self.query_init_std)
        else:
            nn.init.zeros_(self.query)
        self.latest_bias.data.fill_(self.latest_bias_init)
        self.gate.data.zero_()

    def forward(self, states, current):
        assert len(states) >= 1
        q = self.query.float() * self.scale
        logits = []
        for s in states:
            s_norm = F.rms_norm(s.float(), (s.size(-1),))
            logits.append(torch.einsum("d,btd->bt", q, s_norm))
        logits = torch.stack(logits, dim=0)
        logits[-1] = logits[-1] + self.latest_bias.float()
        weights = F.softmax(logits, dim=0)

        mixed = torch.zeros_like(current)
        for i, s in enumerate(states):
            mixed = mixed + weights[i].unsqueeze(-1).to(dtype=s.dtype) * s

        gate = torch.sigmoid(self.gate.float() + self.gate_init)
        out = current + gate.to(dtype=current.dtype) * (mixed - current)

        weights_f = weights.float()
        entropy = -(weights_f.clamp_min(1e-9) * weights_f.clamp_min(1e-9).log()).sum(dim=0).mean()
        self.last_stats = {
            "gate": gate.detach(),
            "latest": weights_f[-1].mean().detach(),
            "x0": weights_f[0].mean().detach(),
            "maxprob": weights_f.max(dim=0).values.mean().detach(),
            "entropy": entropy.detach(),
        }
        return out


class SegmentSelfAttention(nn.Module):
    """Token-local attention over channel segments."""

    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.num_segments = config.segattn_num_segments
        assert self.num_segments >= 2, "segattn_num_segments must be >= 2"
        assert self.n_embd % self.num_segments == 0, "n_embd must be divisible by segattn_num_segments"
        self.segment_dim = self.n_embd // self.num_segments
        self.proj_dim = config.segattn_proj_dim if config.segattn_proj_dim > 0 else self.segment_dim
        self.segattn_mode = config.segattn_mode
        self.matrix_gated = self.segattn_mode in {"gated", "dual_gated"}
        self.branch_gated = self.segattn_mode in {"branch_gated", "dual_gated"}
        self.gated = self.matrix_gated or self.branch_gated
        self.gate_init = config.segattn_gate_init
        self.gate_cap = config.segattn_gate_cap
        self.q_proj = nn.Linear(self.segment_dim, self.proj_dim, bias=False)
        self.k_proj = nn.Linear(self.segment_dim, self.proj_dim, bias=False)
        self.v_proj = nn.Linear(self.segment_dim, self.segment_dim, bias=False)
        self.out_proj = nn.Linear(self.segment_dim, self.segment_dim, bias=False)
        self.gate_proj = nn.Linear(self.n_embd, 1, bias=False) if self.gated else None
        self.scale = self.proj_dim ** -0.5
        self.last_stats = None

    def init_weights(self, s):
        torch.nn.init.uniform_(self.q_proj.weight, -s, s)
        torch.nn.init.uniform_(self.k_proj.weight, -s, s)
        torch.nn.init.uniform_(self.v_proj.weight, -s, s)
        torch.nn.init.zeros_(self.out_proj.weight)
        if self.gate_proj is not None:
            torch.nn.init.zeros_(self.gate_proj.weight)

    def forward(self, x):
        B, T, C = x.shape
        xs = x.reshape(B, T, self.num_segments, self.segment_dim)
        xs_norm = F.rms_norm(xs, (self.segment_dim,))
        q = self.q_proj(xs_norm)
        k = self.k_proj(xs_norm)
        v = self.v_proj(xs)
        att = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
        att = F.softmax(att, dim=-1)
        gate = None
        if self.gated:
            gate = (self.gate_cap * torch.sigmoid(self.gate_proj(x.float()) + self.gate_init)).view(B, T, 1, 1)
        if self.matrix_gated:
            eye = torch.eye(self.num_segments, device=x.device, dtype=att.dtype).view(1, 1, self.num_segments, self.num_segments)
            att = (1.0 - gate) * eye + gate * att
        y = torch.matmul(att.to(dtype=v.dtype), v)
        y = self.out_proj(y)
        if self.branch_gated:
            y = gate.to(dtype=y.dtype) * y
        att_f = att.float()
        diag = att_f.diagonal(dim1=-2, dim2=-1).mean()
        stats = {
            "diag": diag.detach(),
            "offdiag": (1.0 - diag).detach(),
            "entropy": (
                -(att_f.clamp_min(1e-9) * att_f.clamp_min(1e-9).log()).sum(dim=-1).mean()
            ).detach(),
        }
        if gate is not None:
            stats["gate"] = gate.float().mean().detach()
        self.last_stats = stats
        return y.reshape(B, T, C)


class PermixSkipMixer(nn.Module):
    """Two-stream doubly-stochastic mixer over the main stream and a skip state."""

    def __init__(
        self,
        n_embd,
        router_dim=0,
        identity_bias=2.0,
        mix_strength=1.0,
        router_temperature=1.0,
    ):
        super().__init__()
        if router_dim > 0:
            self.router = nn.Sequential(
                nn.Linear(n_embd, router_dim, bias=False),
                nn.GELU(),
                nn.Linear(router_dim, 2, bias=True),
            )
        else:
            self.router = nn.Linear(n_embd, 2, bias=True)
        self.identity_bias = identity_bias
        self.mix_strength = mix_strength
        self.router_temperature = router_temperature
        self.last_stats = None

    def init_weights(self):
        final_linear = self.router[-1] if isinstance(self.router, nn.Sequential) else self.router
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if final_linear.bias is not None:
            final_linear.bias.data[0] = self.identity_bias
            final_linear.bias.data[1] = 0.0

    def forward(self, x, s):
        router_in = 0.5 * (x + s)
        router_logits = self.router(router_in.float()) / self.router_temperature
        alpha = F.softmax(router_logits, dim=-1)
        alpha_s = (alpha[..., 1:2] * self.mix_strength).to(dtype=x.dtype)
        alpha_x = 1.0 - alpha_s
        x_new = alpha_x * x + alpha_s * s
        s_new = alpha_x * s + alpha_s * x

        alpha_f = alpha.float()
        alpha_s_f = alpha_s.float()
        entropy = -(alpha_f.clamp_min(1e-9) * alpha_f.clamp_min(1e-9).log()).sum(dim=-1).mean()
        self.last_stats = {
            "id": (1.0 - alpha_s_f).mean().detach(),
            "nonid": alpha_s_f.mean().detach(),
            "raw_nonid": alpha_f[..., 1].mean().detach(),
            "entropy": entropy.detach(),
            "maxprob": alpha_f.max(dim=-1).values.mean().detach(),
            "swap_frac": (alpha_f[..., 1] > alpha_f[..., 0]).float().mean().detach(),
            "router_temp": torch.tensor(self.router_temperature, device=x.device).detach(),
        }
        return x_new, s_new


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attnres_mode = config.attnres_mode
        self.norm_type = config.norm_type
        self.ln_1 = SimpleLayerNorm(config.n_embd, bias=config.bias) if self.norm_type == "layernorm" else None
        self.ln_2 = SimpleLayerNorm(config.n_embd, bias=config.bias) if self.norm_type == "layernorm" else None
        self.attnres_attn = AttentionResidual(config) if self.attnres_mode == "legacy_block" else None
        self.attnres_mlp = AttentionResidual(config) if self.attnres_mode == "legacy_block" else None
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.segattn = (
            SegmentSelfAttention(config)
            if config.segattn_mode != "off" and layer_idx >= config.segattn_start_layer
            else None
        )
        self.last_attnres_stats = None

    def init_weights(self):
        if self.attnres_attn is not None:
            self.attnres_attn.init_weights()
        if self.attnres_mlp is not None:
            self.attnres_mlp.init_weights()

    def _norm1(self, x):
        return self.ln_1(x) if self.ln_1 is not None else norm(x)

    def _norm2(self, x):
        return self.ln_2(x) if self.ln_2 is not None else norm(x)

    def attn_output(self, x, ve, cos_sin, window_size, moda_caches=None, anchor_cache=None):
        return self.attn(self._norm1(x), ve, cos_sin, window_size, moda_caches=moda_caches, anchor_cache=anchor_cache)

    def mlp_output(self, x):
        return self.mlp(self._norm2(x))

    def forward(self, x, ve, cos_sin, window_size, block_history=None, moda_caches=None, anchor_cache=None):
        self.last_attnres_stats = None
        if self.attnres_mode == "legacy_block":
            assert block_history is not None
            h_attn = self.attnres_attn(block_history + [x], x)
            x = x + self.attn(self._norm1(h_attn), ve, cos_sin, window_size, moda_caches=moda_caches, anchor_cache=anchor_cache)
            h_mlp = self.attnres_mlp(block_history + [x], x)
            x = x + self.mlp(self._norm2(h_mlp))
            self.last_attnres_stats = {
                key: 0.5 * (self.attnres_attn.last_stats[key] + self.attnres_mlp.last_stats[key])
                for key in self.attnres_attn.last_stats
            }
        else:
            x = x + self.attn(self._norm1(x), ve, cos_sin, window_size, moda_caches=moda_caches, anchor_cache=anchor_cache)
            x = x + self.mlp(self._norm2(x))
        if self.segattn is not None:
            x = x + self.segattn(self._norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        if config.use_absolute_positions:
            self.transformer["wpe"] = nn.Embedding(config.sequence_len, config.n_embd)
        self.transformer["drop"] = nn.Dropout(config.dropout)
        if config.norm_type == "layernorm":
            self.transformer["ln_f"] = SimpleLayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.reference_init and config.tie_lm_head:
            self.transformer.wte.weight = self.lm_head.weight
        self.tie_lm_head = config.tie_lm_head
        self.lm_head_rotation_enabled = config.lm_head_rotation_every > 0
        if self.lm_head_rotation_enabled and self.tie_lm_head:
            raise ValueError("LM head rotation requires AUTORESEARCH_UNTIE_LM_HEAD=1 so the output head is not tied to input embeddings.")
        self.lm_head_rotation_every = config.lm_head_rotation_every
        self.lm_head_rotation_rank = config.lm_head_rotation_rank
        self.lm_head_rotation_alpha = config.lm_head_rotation_alpha
        self.lm_head_rotation_buffer = None
        if self.lm_head_rotation_enabled:
            self.lm_head_rotation_buffer = LogitGradBuffer(
                max_rows=config.lm_head_rotation_buffer_rows,
                device=config.lm_head_rotation_buffer_device,
                dtype=torch.float16,
            )
        self.last_lm_head_rotation_stats = None
        self.lm_head_rotation_updates = 0
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.attnres_enabled = config.attnres_mode != "off"
        self.attnres_full_enabled = config.attnres_mode == "full"
        self.attnres_block_enabled = config.attnres_mode == "block"
        self.attnres_legacy_full_enabled = config.attnres_mode == "legacy_full"
        self.attnres_legacy_block_enabled = config.attnres_mode == "legacy_block"
        self.attnres_num_logical_layers = 2 * config.n_layer
        self.attnres_register_scope = config.attnres_register_scope
        self.attnres_token_register_mode = config.attnres_token_register_mode
        self.attnres_token_registers = config.attnres_token_registers
        self.attnres_token_register_bias_mode = config.attnres_token_register_bias_mode
        self.attnres_token_register_value_mode = config.attnres_token_register_value_mode
        self.attnres_token_register_scale = config.attnres_token_register_scale
        self.attnres_token_register_scale_start = config.attnres_token_register_scale_start
        self.attnres_token_register_scale_warmup_steps = config.attnres_token_register_scale_warmup_steps
        self.attnres_token_register_delta_scale = config.attnres_token_register_delta_scale
        self.attnres_token_register_weight_cap = config.attnres_token_register_weight_cap
        self.attnres_token_register_spread_coef = config.attnres_token_register_spread_coef
        self.attnres_token_register_spread_samples = config.attnres_token_register_spread_samples
        self.attnres_token_query_mode = config.attnres_token_query_mode
        self.attnres_token_query_value_mode = config.attnres_token_query_value_mode
        self.attnres_token_query_scale = config.attnres_token_query_scale
        self.attnres_final_memory_mode = config.attnres_final_memory_mode
        self.attnres_final_memory_source = config.attnres_final_memory_source
        self.attnres_final_memory_value_mode = config.attnres_final_memory_value_mode
        self.attnres_final_memory_scale = config.attnres_final_memory_scale
        self.attnres_final_memory_groups = config.attnres_final_memory_groups
        self.attnres_final_memory_rank = config.attnres_final_memory_rank
        self.attnres_final_memory_residual_rank = config.attnres_final_memory_residual_rank
        self.attnres_final_memory_residual_scale = config.attnres_final_memory_residual_scale
        self.attnres_final_memory_bigram_buckets = config.attnres_final_memory_bigram_buckets
        self.attnres_final_memory_bigram_banks = config.attnres_final_memory_bigram_banks
        self.attnres_final_memory_bigram_scale = config.attnres_final_memory_bigram_scale
        self.attnres_final_memory_gate_cap = config.attnres_final_memory_gate_cap
        self.attnres_final_memory_gate_bias_init = config.attnres_final_memory_gate_bias_init
        self.attnres_final_memory_delta_scale = config.attnres_final_memory_delta_scale
        self.attnres_final_memory_query_cap = config.attnres_final_memory_query_cap
        self.attnres_final_memory_query_bias_init = config.attnres_final_memory_query_bias_init
        self.attnres_final_memory_query_delta_scale = config.attnres_final_memory_query_delta_scale
        self.attnres_final_memory_q_mod_scale = config.attnres_final_memory_q_mod_scale
        self.attnres_final_memory_k_mod_scale = config.attnres_final_memory_k_mod_scale
        self.attnres_final_memory_v_mod_scale = config.attnres_final_memory_v_mod_scale
        self.attnres_final_memory_logit_mod_scale = config.attnres_final_memory_logit_mod_scale
        self.attnres_final_memory_vres_scale = config.attnres_final_memory_vres_scale
        self.attnres_input_memory_mode = config.attnres_input_memory_mode
        self.attnres_input_memory_value_mode = config.attnres_input_memory_value_mode
        self.attnres_input_memory_scale = config.attnres_input_memory_scale
        self.attnres_input_memory_hash_dim = config.attnres_input_memory_hash_dim
        self.attnres_input_memory_bigram_buckets = config.attnres_input_memory_bigram_buckets
        self.attnres_input_memory_bigram_banks = config.attnres_input_memory_bigram_banks
        self.attnres_input_memory_bigram_scale = config.attnres_input_memory_bigram_scale
        self.attnres_input_memory_trigram_buckets = config.attnres_input_memory_trigram_buckets
        self.attnres_input_memory_trigram_banks = config.attnres_input_memory_trigram_banks
        self.attnres_input_memory_trigram_scale = config.attnres_input_memory_trigram_scale
        self.attnres_input_memory_smear = config.attnres_input_memory_smear
        self.attnres_input_memory_smear_bias_init = config.attnres_input_memory_smear_bias_init
        if self.attnres_register_scope not in {"all", "attn_only", "mlp_only", "final_only"}:
            raise ValueError(f"unknown attnres_register_scope: {self.attnres_register_scope}")
        if self.attnres_token_register_mode not in {"off", "shared", "query_local"}:
            raise ValueError(f"unknown attnres_token_register_mode: {self.attnres_token_register_mode}")
        if self.attnres_token_register_bias_mode not in {"static", "projected"}:
            raise ValueError(f"unknown attnres_token_register_bias_mode: {self.attnres_token_register_bias_mode}")
        if self.attnres_token_register_value_mode not in {"raw", "rmsnorm"}:
            raise ValueError(f"unknown attnres_token_register_value_mode: {self.attnres_token_register_value_mode}")
        if self.attnres_token_query_mode not in {"off", "shared", "query_local"}:
            raise ValueError(f"unknown attnres_token_query_mode: {self.attnres_token_query_mode}")
        if self.attnres_token_query_value_mode not in {"raw", "rmsnorm"}:
            raise ValueError(f"unknown attnres_token_query_value_mode: {self.attnres_token_query_value_mode}")
        if self.attnres_final_memory_mode not in {
            "off",
            "static",
            "dynamic",
            "unified_static",
            "unified_projected",
            "query_static",
            "query_projected",
            "hybrid_static",
            "hybrid_projected",
            "modkv",
            "modqkv",
            "modlogit",
            "modkvlogit",
            "modqkvlogit",
            "vres",
        }:
            raise ValueError(f"unknown attnres_final_memory_mode: {self.attnres_final_memory_mode}")
        if self.attnres_input_memory_mode not in {"off", "bigram", "bigram_trigram"}:
            raise ValueError(f"unknown attnres_input_memory_mode: {self.attnres_input_memory_mode}")
        if self.attnres_final_memory_source not in {"full", "factorized", "tied", "tied_residual", "tied_bigram", "tied_bigram_only", "tied_bigram_factorized"}:
            raise ValueError(f"unknown attnres_final_memory_source: {self.attnres_final_memory_source}")
        if self.attnres_final_memory_value_mode not in {"raw", "rmsnorm"}:
            raise ValueError(f"unknown attnres_final_memory_value_mode: {self.attnres_final_memory_value_mode}")
        if self.attnres_input_memory_value_mode not in {"raw", "rmsnorm"}:
            raise ValueError(f"unknown attnres_input_memory_value_mode: {self.attnres_input_memory_value_mode}")
        if self.attnres_final_memory_groups <= 0:
            raise ValueError("attnres_final_memory_groups must be >= 1")
        if config.n_embd % self.attnres_final_memory_groups != 0:
            raise ValueError("n_embd must be divisible by attnres_final_memory_groups")
        if self.attnres_final_memory_source == "factorized" and self.attnres_final_memory_rank <= 0:
            raise ValueError("attnres_final_memory_rank must be > 0 for factorized final memory")
        if self.attnres_final_memory_source == "tied_bigram_factorized" and self.attnres_final_memory_rank <= 0:
            raise ValueError("attnres_final_memory_rank must be > 0 for tied_bigram_factorized final memory")
        if self.attnres_final_memory_source == "tied_residual" and self.attnres_final_memory_residual_rank <= 0:
            raise ValueError("attnres_final_memory_residual_rank must be > 0 for tied_residual final memory")
        if self.attnres_final_memory_source in {"tied_bigram", "tied_bigram_only", "tied_bigram_factorized"} and self.attnres_final_memory_bigram_buckets <= 0:
            raise ValueError("attnres_final_memory_bigram_buckets must be > 0 for tied_bigram final memory")
        if self.attnres_final_memory_source in {"tied_bigram", "tied_bigram_only", "tied_bigram_factorized"} and self.attnres_final_memory_bigram_banks <= 0:
            raise ValueError("attnres_final_memory_bigram_banks must be >= 1 for tied_bigram final memory")
        if self.attnres_input_memory_mode != "off":
            if self.attnres_input_memory_hash_dim <= 0:
                raise ValueError("attnres_input_memory_hash_dim must be > 0 when input memory is enabled")
            if self.attnres_input_memory_bigram_buckets <= 0:
                raise ValueError("attnres_input_memory_bigram_buckets must be > 0 when input memory is enabled")
            if self.attnres_input_memory_bigram_banks <= 0:
                raise ValueError("attnres_input_memory_bigram_banks must be >= 1 when input memory is enabled")
        if self.attnres_input_memory_mode == "bigram_trigram":
            if self.attnres_input_memory_trigram_buckets <= 0:
                raise ValueError("attnres_input_memory_trigram_buckets must be > 0 for bigram_trigram input memory")
            if self.attnres_input_memory_trigram_banks <= 0:
                raise ValueError("attnres_input_memory_trigram_banks must be >= 1 for bigram_trigram input memory")
        if self.attnres_final_memory_mode != "off":
            if self.attnres_token_registers > 0 or self.config.attnres_num_registers > 0:
                raise ValueError("attnres_final_memory_mode requires attnres_token_registers=0 and attnres_num_registers=0")
            if self.attnres_token_query_mode != "off":
                raise ValueError("attnres_final_memory_mode requires attnres_token_query_mode=off")
        if self.attnres_final_memory_mode in {"unified_static", "unified_projected"}:
            if self.attnres_final_memory_source not in {"tied", "tied_bigram", "tied_bigram_only"}:
                raise ValueError("unified final memory mode currently supports tied/tied_bigram/tied_bigram_only sources only")
        if self.attnres_full_enabled:
            self.attnres_mixer = FullAttnResMixer(
                num_logical_layers=self.attnres_num_logical_layers,
                dim=config.n_embd,
                eps=config.attnres_eps,
                weight_mode=config.attnres_weight_mode,
                diff_lam_init=config.attnres_diff_lam_init,
                num_registers=config.attnres_num_registers,
                register_mode=config.attnres_register_mode,
                register_bias_init=config.attnres_register_bias_init,
                register_bias_start=config.attnres_register_bias_start,
                register_bias_target=config.attnres_register_bias_target,
                register_bias_warmup_steps=config.attnres_register_bias_warmup_steps,
                register_init_std=config.attnres_register_init_std,
            )
        elif self.attnres_block_enabled:
            self.attnres_mixer = BlockAttnResMixer(
                num_logical_layers=self.attnres_num_logical_layers,
                dim=config.n_embd,
                block_size=config.attnres_block_size,
                eps=config.attnres_eps,
                weight_mode=config.attnres_weight_mode,
                diff_lam_init=config.attnres_diff_lam_init,
                num_registers=config.attnres_num_registers,
                register_mode=config.attnres_register_mode,
                register_bias_init=config.attnres_register_bias_init,
                register_bias_start=config.attnres_register_bias_start,
                register_bias_target=config.attnres_register_bias_target,
                register_bias_warmup_steps=config.attnres_register_bias_warmup_steps,
                register_init_std=config.attnres_register_init_std,
            )
        else:
            self.attnres_mixer = None
        self.attnres_token_register_runtime_bias_offset = 0.0
        self.attnres_token_register_runtime_scale = self.attnres_token_register_scale
        if self.attnres_enabled and self.attnres_token_registers > 0 and self.attnres_token_register_mode in {"shared", "query_local"}:
            if self.attnres_token_register_mode == "shared":
                self.attnres_token_register_embeds = nn.ModuleList([
                    nn.Embedding(config.vocab_size, config.n_embd)
                    for _ in range(self.attnres_token_registers)
                ])
            else:
                self.attnres_token_register_embeds = nn.ModuleList([
                    nn.ModuleList([
                        nn.Embedding(config.vocab_size, config.n_embd)
                        for _ in range(self.attnres_num_logical_layers + 1)
                    ])
                    for _ in range(self.attnres_token_registers)
                ])
            self.attnres_token_register_bias = nn.Parameter(
                torch.zeros(self.attnres_num_logical_layers + 1, self.attnres_token_registers)
            )
            if self.attnres_token_register_bias_mode == "projected":
                self.attnres_token_register_bias_proj = nn.ModuleList([
                    nn.Linear(config.n_embd, self.attnres_token_registers, bias=False)
                    for _ in range(self.attnres_num_logical_layers + 1)
                ])
            else:
                self.attnres_token_register_bias_proj = nn.ModuleList()
        else:
            self.attnres_token_register_embeds = nn.ModuleList()
            self.attnres_token_register_bias = None
            self.attnres_token_register_bias_proj = nn.ModuleList()
        if self.attnres_enabled and self.attnres_token_query_mode in {"shared", "query_local"}:
            if self.attnres_token_query_mode == "shared":
                self.attnres_token_query_embeds = nn.ModuleList([
                    nn.Embedding(config.vocab_size, config.n_embd)
                ])
            else:
                self.attnres_token_query_embeds = nn.ModuleList([
                    nn.Embedding(config.vocab_size, config.n_embd)
                    for _ in range(self.attnres_num_logical_layers + 1)
                ])
        else:
            self.attnres_token_query_embeds = nn.ModuleList()
        if self.attnres_enabled and self.attnres_final_memory_mode != "off":
            if self.attnres_final_memory_source == "full":
                self.attnres_final_memory_embed = nn.Embedding(config.vocab_size, config.n_embd)
                self.attnres_final_memory_proj = None
            elif self.attnres_final_memory_source == "factorized":
                self.attnres_final_memory_embed = nn.Embedding(config.vocab_size, self.attnres_final_memory_rank)
                self.attnres_final_memory_proj = nn.Linear(self.attnres_final_memory_rank, config.n_embd, bias=False)
            elif self.attnres_final_memory_source == "tied_residual":
                self.attnres_final_memory_embed = nn.Embedding(config.vocab_size, self.attnres_final_memory_residual_rank)
                self.attnres_final_memory_proj = nn.Linear(self.attnres_final_memory_residual_rank, config.n_embd, bias=False)
            elif self.attnres_final_memory_source in {"tied_bigram", "tied_bigram_only"}:
                if self.attnres_final_memory_bigram_banks == 1:
                    self.attnres_final_memory_embed = nn.Embedding(self.attnres_final_memory_bigram_buckets, config.n_embd)
                else:
                    self.attnres_final_memory_embed = nn.ModuleList([
                        nn.Embedding(self.attnres_final_memory_bigram_buckets, config.n_embd)
                        for _ in range(self.attnres_final_memory_bigram_banks)
                    ])
                self.attnres_final_memory_proj = None
            elif self.attnres_final_memory_source == "tied_bigram_factorized":
                if self.attnres_final_memory_bigram_banks == 1:
                    self.attnres_final_memory_embed = nn.Embedding(
                        self.attnres_final_memory_bigram_buckets, self.attnres_final_memory_rank
                    )
                else:
                    self.attnres_final_memory_embed = nn.ModuleList([
                        nn.Embedding(self.attnres_final_memory_bigram_buckets, self.attnres_final_memory_rank)
                        for _ in range(self.attnres_final_memory_bigram_banks)
                    ])
                self.attnres_final_memory_proj = nn.Linear(self.attnres_final_memory_rank, config.n_embd, bias=False)
            else:
                self.attnres_final_memory_embed = None
                self.attnres_final_memory_proj = None
            if self.attnres_final_memory_mode in {"static", "dynamic", "query_static", "query_projected", "hybrid_static", "hybrid_projected"}:
                self.attnres_final_memory_gate_bias = nn.Parameter(
                    torch.full((self.attnres_final_memory_groups,), float(config.attnres_final_memory_gate_bias_init))
                )
            else:
                self.attnres_final_memory_gate_bias = None
            if self.attnres_final_memory_mode in {"dynamic", "query_projected"}:
                self.attnres_final_memory_gate_proj = nn.Linear(config.n_embd, self.attnres_final_memory_groups, bias=False)
            else:
                self.attnres_final_memory_gate_proj = None
            if self.attnres_final_memory_mode in {"hybrid_static", "hybrid_projected"}:
                self.attnres_final_memory_query_bias = nn.Parameter(
                    torch.full((self.attnres_final_memory_groups,), float(config.attnres_final_memory_query_bias_init))
                )
            else:
                self.attnres_final_memory_query_bias = None
            if self.attnres_final_memory_mode == "hybrid_projected":
                self.attnres_final_memory_query_proj = nn.Linear(config.n_embd, self.attnres_final_memory_groups, bias=False)
            else:
                self.attnres_final_memory_query_proj = None
            if self.attnres_final_memory_mode in {"unified_static", "unified_projected"}:
                self.attnres_final_memory_unified_bias = nn.Parameter(
                    torch.full((self._attnres_num_final_memory_tokens(),), float(config.attnres_final_memory_gate_bias_init))
                )
            else:
                self.attnres_final_memory_unified_bias = None
            if self.attnres_final_memory_mode == "unified_projected":
                self.attnres_final_memory_unified_proj = nn.Linear(
                    config.n_embd, self._attnres_num_final_memory_tokens(), bias=False
                )
            else:
                self.attnres_final_memory_unified_proj = None
            if self.attnres_final_memory_mode in {"modqkv", "modqkvlogit"} and self.attnres_final_memory_q_mod_scale > 0:
                self.attnres_final_memory_q_mod_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
            else:
                self.attnres_final_memory_q_mod_proj = None
            if self.attnres_final_memory_mode in {"modkv", "modqkv", "modkvlogit", "modqkvlogit"} and self.attnres_final_memory_k_mod_scale > 0:
                self.attnres_final_memory_k_mod_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
            else:
                self.attnres_final_memory_k_mod_proj = None
            if self.attnres_final_memory_mode in {"modkv", "modqkv", "modkvlogit", "modqkvlogit"} and self.attnres_final_memory_v_mod_scale > 0:
                self.attnres_final_memory_v_mod_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
            else:
                self.attnres_final_memory_v_mod_proj = None
            if self.attnres_final_memory_mode == "vres" and self.attnres_final_memory_vres_scale > 0:
                self.attnres_final_memory_vres_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
            else:
                self.attnres_final_memory_vres_proj = None
            if self.attnres_final_memory_mode in {"modlogit", "modkvlogit", "modqkvlogit"} and self.attnres_final_memory_logit_mod_scale > 0:
                self.attnres_final_memory_logit_mod_proj = nn.Linear(config.n_embd, self.attnres_num_logical_layers + 1, bias=False)
            else:
                self.attnres_final_memory_logit_mod_proj = None
        else:
            self.attnres_final_memory_embed = None
            self.attnres_final_memory_proj = None
            self.attnres_final_memory_gate_bias = None
            self.attnres_final_memory_gate_proj = None
            self.attnres_final_memory_query_bias = None
            self.attnres_final_memory_query_proj = None
            self.attnres_final_memory_unified_bias = None
            self.attnres_final_memory_unified_proj = None
            self.attnres_final_memory_q_mod_proj = None
            self.attnres_final_memory_k_mod_proj = None
            self.attnres_final_memory_v_mod_proj = None
            self.attnres_final_memory_vres_proj = None
            self.attnres_final_memory_logit_mod_proj = None
        if self.attnres_input_memory_mode != "off":
            if self.attnres_input_memory_bigram_banks == 1:
                self.attnres_input_bigram_embed = nn.Embedding(
                    self.attnres_input_memory_bigram_buckets, self.attnres_input_memory_hash_dim
                )
            else:
                self.attnres_input_bigram_embed = nn.ModuleList([
                    nn.Embedding(self.attnres_input_memory_bigram_buckets, self.attnres_input_memory_hash_dim)
                    for _ in range(self.attnres_input_memory_bigram_banks)
                ])
            self.attnres_input_bigram_proj = nn.Linear(self.attnres_input_memory_hash_dim, config.n_embd, bias=False)
            if self.attnres_input_memory_mode == "bigram_trigram":
                if self.attnres_input_memory_trigram_banks == 1:
                    self.attnres_input_trigram_embed = nn.Embedding(
                        self.attnres_input_memory_trigram_buckets, self.attnres_input_memory_hash_dim
                    )
                else:
                    self.attnres_input_trigram_embed = nn.ModuleList([
                        nn.Embedding(self.attnres_input_memory_trigram_buckets, self.attnres_input_memory_hash_dim)
                        for _ in range(self.attnres_input_memory_trigram_banks)
                    ])
                self.attnres_input_trigram_proj = nn.Linear(self.attnres_input_memory_hash_dim, config.n_embd, bias=False)
            else:
                self.attnres_input_trigram_embed = None
                self.attnres_input_trigram_proj = None
            if self.attnres_input_memory_smear:
                self.attnres_input_smear_gate = nn.Parameter(
                    torch.full((config.n_embd,), float(self.attnres_input_memory_smear_bias_init))
                )
            else:
                self.attnres_input_smear_gate = None
        else:
            self.attnres_input_bigram_embed = None
            self.attnres_input_bigram_proj = None
            self.attnres_input_trigram_embed = None
            self.attnres_input_trigram_proj = None
            self.attnres_input_smear_gate = None
        self.attnres_layers = nn.ModuleList(
            [AttentionResidual(config) for _ in range(config.n_layer)]
        ) if self.attnres_legacy_full_enabled else nn.ModuleList()
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if config.use_value_embeds and has_ve(i, config.n_layer)
        })
        self.permix_layers = nn.ModuleList()
        self.permix_enabled = config.permix_mode != "off"
        if self.permix_enabled:
            for _ in range(config.n_layer):
                self.permix_layers.append(
                    PermixSkipMixer(
                        config.n_embd,
                        router_dim=config.permix_router_dim,
                        identity_bias=config.permix_identity_bias,
                        mix_strength=config.permix_mix_strength,
                        router_temperature=config.permix_router_temperature,
                    )
                )
        self.last_permix_metrics = None
        self.last_attnres_metrics = None
        self.last_attnres_input_memory_stats = None
        self.last_attnres_grad_metrics = None
        self.last_diffattn_metrics = None
        self.last_moda_metrics = None
        self.last_anchorkv_metrics = None
        self.last_segattn_metrics = None
        self.permix_aux_scale = config.permix_aux_scale
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        if config.use_rotary:
            cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        else:
            cos = torch.empty(0, dtype=torch.bfloat16)
            sin = torch.empty(0, dtype=torch.bfloat16)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def set_attnres_runtime_step(self, step):
        self._attnres_set_runtime_step(step)

    def _attnres_use_registers(self, scope_name):
        if self.config.attnres_num_registers <= 0 and self.attnres_token_registers <= 0:
            return False
        if self.attnres_register_scope == "all":
            return True
        return self.attnres_register_scope == scope_name

    def _attnres_use_token_registers(self):
        return self.attnres_token_register_bias is not None and self.attnres_token_registers > 0

    def _iter_attnres_token_register_embeddings(self):
        for item in self.attnres_token_register_embeds:
            if isinstance(item, nn.Embedding):
                yield item
            else:
                for sub in item:
                    yield sub

    def _iter_attnres_token_query_embeddings(self):
        for item in self.attnres_token_query_embeds:
            yield item

    def _attnres_use_final_memory(self):
        return self.attnres_enabled and self.attnres_final_memory_mode != "off"

    def _attnres_use_unified_final_memory(self):
        return self.attnres_enabled and self.attnres_final_memory_mode in {"unified_static", "unified_projected"}

    def _attnres_use_query_final_memory(self):
        return self.attnres_enabled and self.attnres_final_memory_mode in {
            "query_static",
            "query_projected",
            "hybrid_static",
            "hybrid_projected",
        }

    def _attnres_use_blend_final_memory(self):
        return self.attnres_enabled and self.attnres_final_memory_mode in {
            "static",
            "dynamic",
            "hybrid_static",
            "hybrid_projected",
        }

    def _attnres_use_modulated_final_memory(self):
        return self.attnres_enabled and self.attnres_final_memory_mode in {
            "modkv",
            "modqkv",
            "modlogit",
            "modkvlogit",
            "modqkvlogit",
            "vres",
        }

    def _attnres_num_final_memory_tokens(self):
        if self.attnres_final_memory_source == "tied":
            return 1
        if self.attnres_final_memory_source in {"tied_bigram", "tied_bigram_factorized"}:
            return 1 + self.attnres_final_memory_bigram_banks
        if self.attnres_final_memory_source == "tied_bigram_only":
            return self.attnres_final_memory_bigram_banks
        return 1

    def _attnres_final_memory_token_names(self):
        if self.attnres_final_memory_source == "tied":
            return ["unigram"]
        if self.attnres_final_memory_source in {"tied_bigram", "tied_bigram_factorized"}:
            return ["unigram", *[f"bigram_bank{i}" for i in range(self.attnres_final_memory_bigram_banks)]]
        if self.attnres_final_memory_source == "tied_bigram_only":
            return [f"bigram_bank{i}" for i in range(self.attnres_final_memory_bigram_banks)]
        return ["memory"]

    def _iter_attnres_final_memory_embeddings(self):
        if self.attnres_final_memory_embed is None:
            return
        if isinstance(self.attnres_final_memory_embed, nn.Embedding):
            yield self.attnres_final_memory_embed
        else:
            for emb in self.attnres_final_memory_embed:
                yield emb

    def _iter_attnres_input_memory_embeddings(self):
        if self.attnres_input_bigram_embed is not None:
            if isinstance(self.attnres_input_bigram_embed, nn.Embedding):
                yield self.attnres_input_bigram_embed
            else:
                for emb in self.attnres_input_bigram_embed:
                    yield emb
        if self.attnres_input_trigram_embed is not None:
            if isinstance(self.attnres_input_trigram_embed, nn.Embedding):
                yield self.attnres_input_trigram_embed
            else:
                for emb in self.attnres_input_trigram_embed:
                    yield emb

    def _attnres_bigram_bucket_ids(self, idx, bank_index=0):
        prev = torch.zeros_like(idx)
        prev[:, 1:] = idx[:, :-1]
        mul_a = 1315423911 + bank_index * 2246822519
        mul_b = 2654435761 + bank_index * 3266489917
        mixed = prev.long() * mul_a + idx.long() * mul_b
        return torch.remainder(mixed, self.attnres_final_memory_bigram_buckets).to(dtype=torch.long)

    def _attnres_input_bigram_bucket_ids(self, idx, bank_index=0):
        prev = torch.zeros_like(idx)
        prev[:, 1:] = idx[:, :-1]
        mul_a = 2654435761 + bank_index * 2246822519
        mul_b = 805459861 + bank_index * 3266489917
        mixed = prev.long() * mul_a + idx.long() * mul_b
        return torch.remainder(mixed, self.attnres_input_memory_bigram_buckets).to(dtype=torch.long)

    def _attnres_input_trigram_bucket_ids(self, idx, bank_index=0):
        prev1 = torch.zeros_like(idx)
        prev1[:, 1:] = idx[:, :-1]
        prev2 = torch.zeros_like(idx)
        prev2[:, 2:] = idx[:, :-2]
        mul_a = 1103515245 + bank_index * 2654435761
        mul_b = 214013 + bank_index * 2246822519
        mul_c = 2531011 + bank_index * 3266489917
        mixed = prev2.long() * mul_a + prev1.long() * mul_b + idx.long() * mul_c
        return torch.remainder(mixed, self.attnres_input_memory_trigram_buckets).to(dtype=torch.long)

    def _build_attnres_final_memory_tokens(self, idx):
        if not self._attnres_use_unified_final_memory():
            return None
        tokens = []
        if self.attnres_final_memory_source in {"tied", "tied_bigram", "tied_bigram_factorized"}:
            tokens.append(self.transformer.wte(idx))
        if self.attnres_final_memory_source in {"tied_bigram", "tied_bigram_only", "tied_bigram_factorized"}:
            if isinstance(self.attnres_final_memory_embed, nn.Embedding):
                bigram = self.attnres_final_memory_embed(self._attnres_bigram_bucket_ids(idx, 0))
                if self.attnres_final_memory_source == "tied_bigram_factorized":
                    bigram = self.attnres_final_memory_proj(bigram)
                tokens.append(bigram)
            else:
                for bank_index, emb in enumerate(self.attnres_final_memory_embed):
                    bigram = emb(self._attnres_bigram_bucket_ids(idx, bank_index))
                    if self.attnres_final_memory_source == "tied_bigram_factorized":
                        bigram = self.attnres_final_memory_proj(bigram)
                    tokens.append(bigram)
        token_tensor = torch.stack(tokens, dim=0)
        if self.attnres_final_memory_value_mode == "rmsnorm":
            token_tensor = F.rms_norm(token_tensor.float(), (token_tensor.size(-1),)).to(dtype=token_tensor.dtype)
        if self.attnres_final_memory_scale != 1.0:
            token_tensor = token_tensor * token_tensor.new_tensor(self.attnres_final_memory_scale)
        return token_tensor

    def _build_attnres_final_memory_unified_bias(self, query_input):
        if not self._attnres_use_unified_final_memory():
            return None
        bias = self.attnres_final_memory_unified_bias
        if self.attnres_final_memory_mode == "unified_projected":
            proj_input = F.rms_norm(query_input.float(), (query_input.size(-1),))
            delta = self.attnres_final_memory_unified_proj(proj_input)
            if self.attnres_final_memory_delta_scale != 1.0:
                delta = delta * delta.new_tensor(self.attnres_final_memory_delta_scale)
            return bias.view(-1, 1, 1).to(dtype=query_input.dtype) + delta.permute(2, 0, 1).to(dtype=query_input.dtype)
        return bias

    def _attnres_set_runtime_step(self, step):
        if self.attnres_mixer is not None:
            self.attnres_mixer.set_runtime_step(step)
        if self.attnres_token_register_scale_warmup_steps > 0:
            frac = min(max(float(step), 0.0) / float(self.attnres_token_register_scale_warmup_steps), 1.0)
            self.attnres_token_register_runtime_scale = self.attnres_token_register_scale_start + frac * (
                self.attnres_token_register_scale - self.attnres_token_register_scale_start
            )
        else:
            self.attnres_token_register_runtime_scale = self.attnres_token_register_scale
        if self.attnres_token_register_bias is None or self.config.attnres_register_bias_warmup_steps <= 0:
            self.attnres_token_register_runtime_bias_offset = 0.0
            return
        frac = min(max(float(step), 0.0) / float(self.config.attnres_register_bias_warmup_steps), 1.0)
        scheduled_bias = self.config.attnres_register_bias_start + frac * (
            self.config.attnres_register_bias_target - self.config.attnres_register_bias_start
        )
        self.attnres_token_register_runtime_bias_offset = scheduled_bias - self.config.attnres_register_bias_target

    def _build_attnres_token_registers(self, idx, query_index, bias_input=None):
        if not self._attnres_use_token_registers():
            return None, None
        if self.attnres_token_register_mode == "shared":
            extra_registers = torch.stack([emb(idx) for emb in self.attnres_token_register_embeds], dim=0)
        elif self.attnres_token_register_mode == "query_local":
            extra_registers = torch.stack([embs[query_index](idx) for embs in self.attnres_token_register_embeds], dim=0)
        else:
            raise ValueError(f"unknown attnres_token_register_mode: {self.attnres_token_register_mode}")
        if self.attnres_token_register_value_mode == "rmsnorm":
            extra_registers = F.rms_norm(extra_registers.float(), (extra_registers.size(-1),)).to(dtype=extra_registers.dtype)
        if self.attnres_token_register_runtime_scale != 1.0:
            extra_registers = extra_registers * extra_registers.new_tensor(self.attnres_token_register_runtime_scale)
        extra_bias = self.attnres_token_register_bias[query_index]
        if self.attnres_token_register_runtime_bias_offset != 0.0:
            extra_bias = extra_bias + extra_bias.new_tensor(self.attnres_token_register_runtime_bias_offset)
        if self.attnres_token_register_bias_mode == "projected" and bias_input is not None:
            proj_input = F.rms_norm(bias_input.float(), (bias_input.size(-1),))
            bias_delta = self.attnres_token_register_bias_proj[query_index](proj_input).permute(2, 0, 1)
            if self.attnres_token_register_delta_scale != 1.0:
                bias_delta = bias_delta * bias_delta.new_tensor(self.attnres_token_register_delta_scale)
            extra_bias = extra_bias.view(-1, 1, 1) + bias_delta
        return extra_registers, extra_bias

    def _build_attnres_final_memory(self, idx):
        if not self._attnres_use_final_memory():
            return None
        if self.attnres_final_memory_source == "full":
            memory = self.attnres_final_memory_embed(idx)
        elif self.attnres_final_memory_source == "factorized":
            memory = self.attnres_final_memory_proj(self.attnres_final_memory_embed(idx))
        elif self.attnres_final_memory_source == "tied":
            memory = self.transformer.wte(idx)
        elif self.attnres_final_memory_source == "tied_residual":
            memory = self.transformer.wte(idx)
            residual = self.attnres_final_memory_proj(self.attnres_final_memory_embed(idx))
            if self.attnres_final_memory_residual_scale != 1.0:
                residual = residual * residual.new_tensor(self.attnres_final_memory_residual_scale)
            memory = memory + residual
        elif self.attnres_final_memory_source in {"tied_bigram", "tied_bigram_only", "tied_bigram_factorized"}:
            memory = self.transformer.wte(idx) if self.attnres_final_memory_source == "tied_bigram" else torch.zeros_like(self.transformer.wte(idx))
            if self.attnres_final_memory_source == "tied_bigram_factorized":
                memory = self.transformer.wte(idx)
            if isinstance(self.attnres_final_memory_embed, nn.Embedding):
                bigram = self.attnres_final_memory_embed(self._attnres_bigram_bucket_ids(idx, 0))
            else:
                bigram = None
                for bank_index, emb in enumerate(self.attnres_final_memory_embed):
                    bank_memory = emb(self._attnres_bigram_bucket_ids(idx, bank_index))
                    bigram = bank_memory if bigram is None else bigram + bank_memory
            if self.attnres_final_memory_source == "tied_bigram_factorized":
                bigram = self.attnres_final_memory_proj(bigram)
            if self.attnres_final_memory_bigram_scale != 1.0:
                bigram = bigram * bigram.new_tensor(self.attnres_final_memory_bigram_scale)
            memory = memory + bigram
        else:
            raise ValueError(f"unknown attnres_final_memory_source: {self.attnres_final_memory_source}")
        if self.attnres_final_memory_value_mode == "rmsnorm":
            memory = F.rms_norm(memory.float(), (memory.size(-1),)).to(dtype=memory.dtype)
        if self.attnres_final_memory_scale != 1.0:
            memory = memory * memory.new_tensor(self.attnres_final_memory_scale)
        return memory

    def _build_attnres_input_memory_delta(self, idx):
        if self.attnres_input_memory_mode == "off":
            return None, None
        bigram_low = None
        if isinstance(self.attnres_input_bigram_embed, nn.Embedding):
            bigram_low = self.attnres_input_bigram_embed(self._attnres_input_bigram_bucket_ids(idx, 0))
        else:
            for bank_index, emb in enumerate(self.attnres_input_bigram_embed):
                update = emb(self._attnres_input_bigram_bucket_ids(idx, bank_index))
                bigram_low = update if bigram_low is None else (bigram_low + update)
        if self.attnres_input_memory_value_mode == "rmsnorm":
            bigram_low = F.rms_norm(bigram_low.float(), (bigram_low.size(-1),)).to(dtype=bigram_low.dtype)
        bigram = self.attnres_input_bigram_proj(bigram_low)
        if self.attnres_input_memory_bigram_scale != 1.0:
            bigram = bigram * bigram.new_tensor(self.attnres_input_memory_bigram_scale)
        delta = bigram
        stats = {
            "input_memory_bigram_rms": bigram.float().pow(2).mean(dim=-1).sqrt().mean().detach(),
        }
        if self.attnres_input_memory_mode == "bigram_trigram":
            trigram_low = None
            if isinstance(self.attnres_input_trigram_embed, nn.Embedding):
                trigram_low = self.attnres_input_trigram_embed(self._attnres_input_trigram_bucket_ids(idx, 0))
            else:
                for bank_index, emb in enumerate(self.attnres_input_trigram_embed):
                    update = emb(self._attnres_input_trigram_bucket_ids(idx, bank_index))
                    trigram_low = update if trigram_low is None else (trigram_low + update)
            if self.attnres_input_memory_value_mode == "rmsnorm":
                trigram_low = F.rms_norm(trigram_low.float(), (trigram_low.size(-1),)).to(dtype=trigram_low.dtype)
            trigram = self.attnres_input_trigram_proj(trigram_low)
            if self.attnres_input_memory_trigram_scale != 1.0:
                trigram = trigram * trigram.new_tensor(self.attnres_input_memory_trigram_scale)
            delta = delta + trigram
            stats["input_memory_trigram_rms"] = trigram.float().pow(2).mean(dim=-1).sqrt().mean().detach()
        if self.attnres_input_memory_scale != 1.0:
            delta = delta * delta.new_tensor(self.attnres_input_memory_scale)
        if self.attnres_input_smear_gate is not None:
            prev_delta = torch.zeros_like(delta)
            prev_delta[:, 1:] = delta[:, :-1]
            gate = torch.sigmoid(self.attnres_input_smear_gate).to(dtype=delta.dtype).view(1, 1, -1)
            delta = torch.lerp(delta, prev_delta, gate)
            stats["input_memory_smear_gate"] = gate.mean().detach()
        stats["input_memory_delta_norm"] = delta.float().norm(dim=-1).mean().detach()
        stats["input_memory_rms"] = delta.float().pow(2).mean(dim=-1).sqrt().mean().detach()
        return delta, stats

    def _apply_attnres_input_memory(self, idx, x):
        delta, stats = self._build_attnres_input_memory_delta(idx)
        if delta is None:
            return x, None
        return x + delta.to(dtype=x.dtype), stats

    def _build_attnres_final_memory_query(self, idx, gate_input):
        if not self._attnres_use_query_final_memory():
            return None, None
        memory = self._build_attnres_final_memory(idx)
        B, T, D = memory.shape
        G = self.attnres_final_memory_groups
        group_dim = D // G
        memory_groups = memory.view(B, T, G, group_dim)
        if self.attnres_final_memory_mode in {"hybrid_static", "hybrid_projected"}:
            gate_logit = self.attnres_final_memory_query_bias.to(dtype=memory.dtype).view(1, 1, G).expand(B, T, G)
            query_cap = self.attnres_final_memory_query_cap
            query_delta_scale = self.attnres_final_memory_query_delta_scale
            query_proj = self.attnres_final_memory_query_proj
            use_projected = self.attnres_final_memory_mode == "hybrid_projected"
        else:
            gate_logit = self.attnres_final_memory_gate_bias.to(dtype=memory.dtype).view(1, 1, G).expand(B, T, G)
            query_cap = self.attnres_final_memory_gate_cap
            query_delta_scale = self.attnres_final_memory_delta_scale
            query_proj = self.attnres_final_memory_gate_proj
            use_projected = self.attnres_final_memory_mode == "query_projected"
        gate_delta = None
        if use_projected:
            if gate_input is None:
                gate_input = memory
            proj_input = F.rms_norm(gate_input.float(), (gate_input.size(-1),))
            gate_delta = query_proj(proj_input)
            if query_delta_scale != 1.0:
                gate_delta = gate_delta * gate_delta.new_tensor(query_delta_scale)
            gate_logit = gate_logit + gate_delta.to(dtype=gate_logit.dtype)
        gate = torch.sigmoid(gate_logit).unsqueeze(-1)
        if query_cap > 0:
            gate = gate * gate.new_tensor(query_cap)
        query = (memory_groups * gate.to(dtype=memory.dtype)).reshape(B, T, D)
        gate_scalar = gate.squeeze(-1)
        stats = {
            "final_memory_query_gate": gate_scalar.mean().detach(),
            "final_memory_query_gate_std": gate_scalar.std(unbiased=False).detach(),
            "final_memory_query_norm": query.float().norm(dim=-1).mean().detach(),
            "final_memory_query_rms": query.float().pow(2).mean(dim=-1).sqrt().mean().detach(),
            "final_memory_rms": memory.float().pow(2).mean(dim=-1).sqrt().mean().detach(),
        }
        if gate_delta is not None:
            stats["final_memory_query_delta_norm"] = gate_delta.float().norm(dim=-1).mean().detach()
        return query.to(dtype=memory.dtype), stats

    def _build_attnres_final_memory_modulation(self, idx):
        if not self._attnres_use_modulated_final_memory():
            return None, None, None, None, None, None
        memory = self._build_attnres_final_memory(idx)
        mod_input = F.rms_norm(memory.float(), (memory.size(-1),))
        q_delta = None
        logit_bias = None
        key_scale = None
        value_scale = None
        value_residual = None
        stats = {
            "final_memory_rms": memory.float().pow(2).mean(dim=-1).sqrt().mean().detach(),
        }
        if self.attnres_final_memory_q_mod_proj is not None:
            q_delta = torch.tanh(self.attnres_final_memory_q_mod_proj(mod_input))
            q_delta = q_delta * q_delta.new_tensor(self.attnres_final_memory_q_mod_scale)
            stats["final_memory_mod_q_delta_norm"] = q_delta.float().norm(dim=-1).mean().detach()
        if self.attnres_final_memory_k_mod_proj is not None:
            key_delta = torch.tanh(self.attnres_final_memory_k_mod_proj(mod_input))
            key_delta = key_delta * key_delta.new_tensor(self.attnres_final_memory_k_mod_scale)
            key_scale = 1.0 + key_delta
            stats["final_memory_mod_k_delta_norm"] = key_delta.float().norm(dim=-1).mean().detach()
            stats["final_memory_mod_k_scale_mean"] = key_scale.float().mean().detach()
        if self.attnres_final_memory_v_mod_proj is not None:
            value_delta = torch.tanh(self.attnres_final_memory_v_mod_proj(mod_input))
            value_delta = value_delta * value_delta.new_tensor(self.attnres_final_memory_v_mod_scale)
            value_scale = 1.0 + value_delta
            stats["final_memory_mod_v_delta_norm"] = value_delta.float().norm(dim=-1).mean().detach()
            stats["final_memory_mod_v_scale_mean"] = value_scale.float().mean().detach()
        if self.attnres_final_memory_vres_proj is not None:
            value_residual = torch.tanh(self.attnres_final_memory_vres_proj(mod_input))
            value_residual = value_residual * value_residual.new_tensor(self.attnres_final_memory_vres_scale)
            stats["final_memory_vres_delta_norm"] = value_residual.float().norm(dim=-1).mean().detach()
        if self.attnres_final_memory_logit_mod_proj is not None:
            logit_delta = torch.tanh(self.attnres_final_memory_logit_mod_proj(mod_input))
            logit_delta = logit_delta * logit_delta.new_tensor(self.attnres_final_memory_logit_mod_scale)
            logit_bias = logit_delta.permute(2, 0, 1)
            stats["final_memory_mod_logit_delta_norm"] = logit_delta.float().norm(dim=-1).mean().detach()
        return (
            q_delta.to(dtype=memory.dtype) if q_delta is not None else None,
            logit_bias.to(dtype=memory.dtype) if logit_bias is not None else None,
            key_scale.to(dtype=memory.dtype) if key_scale is not None else None,
            value_scale.to(dtype=memory.dtype) if value_scale is not None else None,
            value_residual.to(dtype=memory.dtype) if value_residual is not None else None,
            stats,
        )

    def _apply_attnres_final_memory(self, idx, x, gate_input=None):
        if not self._attnres_use_blend_final_memory():
            return x, None
        memory = self._build_attnres_final_memory(idx).to(dtype=x.dtype)
        B, T, D = x.shape
        G = self.attnres_final_memory_groups
        group_dim = D // G
        x_groups = x.view(B, T, G, group_dim)
        memory_groups = memory.view(B, T, G, group_dim)
        gate_logit = self.attnres_final_memory_gate_bias.to(dtype=x.dtype).view(1, 1, G)
        gate_logit = gate_logit.expand(B, T, G)
        if self.attnres_final_memory_mode == "dynamic":
            if gate_input is None:
                gate_input = x
            proj_input = F.rms_norm(gate_input.float(), (gate_input.size(-1),))
            gate_delta = self.attnres_final_memory_gate_proj(proj_input)
            if self.attnres_final_memory_delta_scale != 1.0:
                gate_delta = gate_delta * gate_delta.new_tensor(self.attnres_final_memory_delta_scale)
            gate_logit = gate_logit + gate_delta.to(dtype=gate_logit.dtype)
        gate = torch.sigmoid(gate_logit).unsqueeze(-1)
        if self.attnres_final_memory_gate_cap > 0:
            gate = gate * gate.new_tensor(self.attnres_final_memory_gate_cap)
        mixed_groups = torch.lerp(x_groups, memory_groups, gate.to(dtype=x.dtype))
        mixed = mixed_groups.reshape(B, T, D)
        gate_scalar = gate.squeeze(-1)
        stats = {
            "final_memory_gate": gate_scalar.mean().detach(),
            "final_memory_gate_std": gate_scalar.std(unbiased=False).detach(),
            "final_memory_groups": gate_scalar.new_tensor(float(G)).detach(),
            "final_memory_source": gate_scalar.new_tensor(
                {
                    "full": 0.0,
                    "factorized": 1.0,
                    "tied": 2.0,
                    "tied_residual": 3.0,
                    "tied_bigram": 4.0,
                    "tied_bigram_only": 5.0,
                    "tied_bigram_factorized": 6.0,
                }[self.attnres_final_memory_source]
            ).detach(),
            "final_memory_delta_norm": (mixed.float() - x.float()).norm(dim=-1).mean().detach(),
            "final_memory_rms": memory.float().pow(2).mean(dim=-1).sqrt().mean().detach(),
        }
        return mixed, stats

    def _summarize_attnres_extra_source_weights(self, weights, meta, names):
        extra_count = int(meta.get("extra_register_count", 0))
        if extra_count <= 0:
            return {}
        start = int(meta.get("num_real_sources", 0)) + int(meta.get("num_builtin_registers", 0))
        extra = weights[start:start + extra_count]
        stats = {
            "memory_total": extra.sum(dim=0).mean(),
        }
        for i, name in enumerate(names[:extra_count]):
            stats[f"memory_{name}"] = extra[i].mean()
        return stats

    def _build_attnres_token_query(self, idx, query_index):
        if not self.attnres_enabled or self.attnres_token_query_mode == "off":
            return None
        if self.attnres_token_query_mode == "shared":
            extra_query = self.attnres_token_query_embeds[0](idx)
        elif self.attnres_token_query_mode == "query_local":
            extra_query = self.attnres_token_query_embeds[query_index](idx)
        else:
            raise ValueError(f"unknown attnres_token_query_mode: {self.attnres_token_query_mode}")
        if self.attnres_token_query_value_mode == "rmsnorm":
            extra_query = F.rms_norm(extra_query.float(), (extra_query.size(-1),)).to(dtype=extra_query.dtype)
        if self.attnres_token_query_scale != 1.0:
            extra_query = extra_query * extra_query.new_tensor(self.attnres_token_query_scale)
        return extra_query

    def _compute_attnres_token_register_spread_loss(self, idx):
        if (
            not self._attnres_use_token_registers()
            or self.attnres_token_register_spread_coef <= 0
            or self.attnres_token_register_spread_samples <= 1
        ):
            return None
        token_ids = idx.reshape(-1).unique()
        if token_ids.numel() <= 1:
            return None
        max_samples = min(int(self.attnres_token_register_spread_samples), int(token_ids.numel()))
        if token_ids.numel() > max_samples:
            perm = torch.randperm(token_ids.numel(), device=token_ids.device)[:max_samples]
            token_ids = token_ids.index_select(0, perm)
        losses = []
        for emb in self._iter_attnres_token_register_embeddings():
            vecs = emb(token_ids)
            if self.attnres_token_register_value_mode == "rmsnorm":
                vecs = F.rms_norm(vecs.float(), (vecs.size(-1),))
            else:
                vecs = vecs.float()
            vecs = F.normalize(vecs, dim=-1, eps=1e-6)
            sim = vecs @ vecs.transpose(0, 1)
            offdiag = sim - torch.eye(sim.size(0), device=sim.device, dtype=sim.dtype)
            losses.append(offdiag.square().mean())
        if not losses:
            return None
        return torch.stack(losses).mean()

    @torch.no_grad()
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, SimpleLayerNorm):
                module.weight.fill_(1.0)
                if module.bias is not None:
                    module.bias.zero_()
            elif isinstance(module, ReferenceRMSNorm):
                module.weight.fill_(1.0)
        if self.attnres_mixer is not None:
            self.attnres_mixer.mixer.init_weights()
        if self.attnres_token_register_bias is not None:
            self.attnres_token_register_bias.fill_(self.config.attnres_register_bias_target)
        if self.config.reference_init:
            torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.02)
            if "wpe" in self.transformer:
                torch.nn.init.normal_(self.transformer.wpe.weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
            for emb in self._iter_attnres_token_register_embeddings():
                torch.nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            for emb in self._iter_attnres_token_query_embeddings():
                torch.nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            for emb in self._iter_attnres_final_memory_embeddings():
                if self.attnres_final_memory_source in {"tied_residual", "tied_bigram", "tied_bigram_only", "tied_bigram_factorized"}:
                    torch.nn.init.zeros_(emb.weight)
                else:
                    torch.nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            for emb in self._iter_attnres_input_memory_embeddings():
                torch.nn.init.zeros_(emb.weight)
            if self.attnres_final_memory_proj is not None:
                torch.nn.init.normal_(self.attnres_final_memory_proj.weight, mean=0.0, std=0.02)
            if self.attnres_input_bigram_proj is not None:
                torch.nn.init.normal_(self.attnres_input_bigram_proj.weight, mean=0.0, std=0.02)
            if self.attnres_input_trigram_proj is not None:
                torch.nn.init.normal_(self.attnres_input_trigram_proj.weight, mean=0.0, std=0.02)
            if self.attnres_final_memory_gate_bias is not None:
                self.attnres_final_memory_gate_bias.fill_(self.attnres_final_memory_gate_bias_init)
            if self.attnres_input_smear_gate is not None:
                self.attnres_input_smear_gate.fill_(self.attnres_input_memory_smear_bias_init)
            for proj in self.attnres_token_register_bias_proj:
                torch.nn.init.zeros_(proj.weight)
            if self.attnres_final_memory_gate_proj is not None:
                torch.nn.init.zeros_(self.attnres_final_memory_gate_proj.weight)
            if self.attnres_final_memory_unified_bias is not None:
                self.attnres_final_memory_unified_bias.fill_(self.attnres_final_memory_gate_bias_init)
            if self.attnres_final_memory_unified_proj is not None:
                torch.nn.init.zeros_(self.attnres_final_memory_unified_proj.weight)
            if self.attnres_final_memory_q_mod_proj is not None:
                torch.nn.init.zeros_(self.attnres_final_memory_q_mod_proj.weight)
            if self.attnres_final_memory_k_mod_proj is not None:
                torch.nn.init.zeros_(self.attnres_final_memory_k_mod_proj.weight)
            if self.attnres_final_memory_v_mod_proj is not None:
                torch.nn.init.zeros_(self.attnres_final_memory_v_mod_proj.weight)
            if self.attnres_final_memory_vres_proj is not None:
                torch.nn.init.zeros_(self.attnres_final_memory_vres_proj.weight)
            if self.attnres_final_memory_logit_mod_proj is not None:
                torch.nn.init.zeros_(self.attnres_final_memory_logit_mod_proj.weight)
            proj_std = 0.02 / math.sqrt(2 * self.config.n_layer)
            for block in self.transformer.h:
                torch.nn.init.normal_(block.attn.c_q.weight, mean=0.0, std=0.02)
                torch.nn.init.normal_(block.attn.c_k.weight, mean=0.0, std=0.02)
                torch.nn.init.normal_(block.attn.c_v.weight, mean=0.0, std=0.02)
                torch.nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=proj_std)
                torch.nn.init.normal_(block.mlp.c_fc.weight, mean=0.0, std=0.02)
                torch.nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=proj_std)
                if block.attn.ve_gate is not None:
                    torch.nn.init.zeros_(block.attn.ve_gate.weight)
                if block.segattn is not None:
                    s = 3**0.5 * self.config.n_embd**-0.5
                    block.segattn.init_weights(s)
            self.resid_lambdas.fill_(1.0)
            self.x0_lambdas.zero_()
            for mixer in self.permix_layers:
                mixer.init_weights()
            for attnres in self.attnres_layers:
                attnres.init_weights()
            for block in self.transformer.h:
                block.init_weights()
            if self.config.use_rotary:
                head_dim = self.config.n_embd // self.config.n_head
                cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
                self.cos, self.sin = cos, sin
            self.transformer.wte.to(dtype=torch.bfloat16)
            if "wpe" in self.transformer:
                self.transformer.wpe.to(dtype=torch.bfloat16)
            for emb in self._iter_attnres_token_register_embeddings():
                emb.to(dtype=torch.bfloat16)
            for emb in self._iter_attnres_token_query_embeddings():
                emb.to(dtype=torch.bfloat16)
            for emb in self._iter_attnres_final_memory_embeddings():
                emb.to(dtype=torch.bfloat16)
            for emb in self._iter_attnres_input_memory_embeddings():
                emb.to(dtype=torch.bfloat16)
            if self.attnres_final_memory_proj is not None:
                self.attnres_final_memory_proj.to(dtype=torch.bfloat16)
            if self.attnres_input_bigram_proj is not None:
                self.attnres_input_bigram_proj.to(dtype=torch.bfloat16)
            if self.attnres_input_trigram_proj is not None:
                self.attnres_input_trigram_proj.to(dtype=torch.bfloat16)
            if self.attnres_final_memory_unified_proj is not None:
                self.attnres_final_memory_unified_proj.to(dtype=torch.bfloat16)
            if not self.tie_lm_head:
                self.lm_head.weight.copy_(self.transformer.wte.weight.to(dtype=self.lm_head.weight.dtype))
            return

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        for emb in self._iter_attnres_token_register_embeddings():
            torch.nn.init.normal_(emb.weight, mean=0.0, std=1.0)
        for emb in self._iter_attnres_token_query_embeddings():
            torch.nn.init.normal_(emb.weight, mean=0.0, std=1.0)
        for emb in self._iter_attnres_final_memory_embeddings():
            if self.attnres_final_memory_source in {"tied_residual", "tied_bigram", "tied_bigram_only", "tied_bigram_factorized"}:
                torch.nn.init.zeros_(emb.weight)
            else:
                torch.nn.init.normal_(emb.weight, mean=0.0, std=1.0)
        if self.attnres_final_memory_proj is not None:
            if self.attnres_final_memory_source == "tied_residual":
                torch.nn.init.normal_(self.attnres_final_memory_proj.weight, mean=0.0, std=1.0)
            else:
                torch.nn.init.normal_(self.attnres_final_memory_proj.weight, mean=0.0, std=1.0)
        if self.attnres_final_memory_gate_bias is not None:
            self.attnres_final_memory_gate_bias.fill_(self.attnres_final_memory_gate_bias_init)
        for proj in self.attnres_token_register_bias_proj:
            torch.nn.init.zeros_(proj.weight)
        if self.attnres_final_memory_gate_proj is not None:
            torch.nn.init.zeros_(self.attnres_final_memory_gate_proj.weight)
        if self.attnres_final_memory_unified_bias is not None:
            self.attnres_final_memory_unified_bias.fill_(self.attnres_final_memory_gate_bias_init)
        if self.attnres_final_memory_unified_proj is not None:
            torch.nn.init.zeros_(self.attnres_final_memory_unified_proj.weight)
        if not self.tie_lm_head:
            self.lm_head.weight.copy_(self.transformer.wte.weight.to(dtype=self.lm_head.weight.dtype))
        if self.attnres_final_memory_q_mod_proj is not None:
            torch.nn.init.zeros_(self.attnres_final_memory_q_mod_proj.weight)
        if self.attnres_final_memory_k_mod_proj is not None:
            torch.nn.init.zeros_(self.attnres_final_memory_k_mod_proj.weight)
        if self.attnres_final_memory_v_mod_proj is not None:
            torch.nn.init.zeros_(self.attnres_final_memory_v_mod_proj.weight)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            if (
                block.attn.diffattn_enabled
                and block.attn.diffattn_q2_mode == "wo"
                and block.attn.diffattn_wo_init_std > 0
            ):
                # q2=wo reuses c_proj.weight as its query source. A tiny nonzero init keeps
                # the residual branch near-zero while avoiding a degenerate all-zero q2.
                torch.nn.init.normal_(
                    block.attn.c_proj.weight, mean=0.0, std=block.attn.diffattn_wo_init_std
                )
            else:
                torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            if block.attn.lam_proj is not None:
                if isinstance(block.attn.lam_proj, nn.Sequential):
                    for module in block.attn.lam_proj:
                        if isinstance(module, nn.Linear):
                            torch.nn.init.zeros_(module.weight)
                    assert block.attn.lam_proj[-1].bias is None
                else:
                    torch.nn.init.zeros_(block.attn.lam_proj.weight)
                    assert block.attn.lam_proj.bias is None
            if block.segattn is not None:
                block.segattn.init_weights(s)
        for mixer in self.permix_layers:
            mixer.init_weights()
        for attnres in self.attnres_layers:
            attnres.init_weights()
        for block in self.transformer.h:
            block.init_weights()
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        if self.config.use_rotary:
            cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
            self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for emb in self._iter_attnres_token_register_embeddings():
            emb.to(dtype=torch.bfloat16)
        for emb in self._iter_attnres_token_query_embeddings():
            emb.to(dtype=torch.bfloat16)
        for emb in self._iter_attnres_final_memory_embeddings():
            emb.to(dtype=torch.bfloat16)
        if self.attnres_final_memory_proj is not None:
            self.attnres_final_memory_proj.to(dtype=torch.bfloat16)
        if self.attnres_final_memory_unified_proj is not None:
            self.attnres_final_memory_unified_proj.to(dtype=torch.bfloat16)
        if self.attnres_final_memory_q_mod_proj is not None:
            self.attnres_final_memory_q_mod_proj.to(dtype=torch.bfloat16)
        if self.attnres_final_memory_k_mod_proj is not None:
            self.attnres_final_memory_k_mod_proj.to(dtype=torch.bfloat16)
        if self.attnres_final_memory_v_mod_proj is not None:
            self.attnres_final_memory_v_mod_proj.to(dtype=torch.bfloat16)
        if self.attnres_final_memory_vres_proj is not None:
            self.attnres_final_memory_vres_proj.to(dtype=torch.bfloat16)
        if self.attnres_final_memory_logit_mod_proj is not None:
            self.attnres_final_memory_logit_mod_proj.to(dtype=torch.bfloat16)
        if "wpe" in self.transformer:
            self.transformer.wpe.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def _summarize_attnres_weights(
        self,
        weights,
        num_real_sources=None,
        num_registers=0,
        num_builtin_registers=0,
        extra_register_count=0,
    ):
        weights_f = weights.float()
        if torch.any(weights_f < 0):
            stats_weights = weights_f.abs() / weights_f.abs().sum(dim=0, keepdim=True).clamp_min(1e-9)
        else:
            stats_weights = weights_f
        latest_index = (num_real_sources - 1) if num_real_sources is not None else (stats_weights.size(0) - 1)
        entropy = -(stats_weights.clamp_min(1e-9) * stats_weights.clamp_min(1e-9).log()).sum(dim=0).mean()
        if num_real_sources is not None and num_registers > 0:
            register_mass = stats_weights[num_real_sources:].sum(dim=0).mean()
        else:
            register_mass = stats_weights.new_zeros(())
        return {
            "latest": stats_weights[latest_index].mean(),
            "x0": stats_weights[0].mean(),
            "maxprob": stats_weights.max(dim=0).values.mean(),
            "entropy": entropy,
            "register": register_mass,
        }

    def _grad_norm_sq(self, param):
        if param is None or param.grad is None:
            return None
        return param.grad.detach().float().pow(2).sum()

    def _module_grad_norm_sq(self, module):
        if module is None:
            return None
        total = None
        for p in module.parameters():
            sq = self._grad_norm_sq(p)
            if sq is not None:
                total = sq if total is None else (total + sq)
        return total

    def collect_attnres_grad_metrics(self):
        metrics = {}
        wte_sq = self._grad_norm_sq(self.transformer.wte.weight)
        if wte_sq is not None:
            metrics["wte_grad_norm"] = wte_sq.sqrt()
        lm_head_sq = self._grad_norm_sq(self.lm_head.weight)
        if lm_head_sq is not None:
            metrics["lm_head_grad_norm"] = lm_head_sq.sqrt()
        total_embed_sq = None
        for emb in self._iter_attnres_final_memory_embeddings():
            sq = self._grad_norm_sq(emb.weight)
            if sq is not None:
                total_embed_sq = sq if total_embed_sq is None else (total_embed_sq + sq)
        if total_embed_sq is not None:
            metrics["final_memory_embed_grad_norm"] = total_embed_sq.sqrt()
        input_embed_sq = None
        for emb in self._iter_attnres_input_memory_embeddings():
            sq = self._grad_norm_sq(emb.weight)
            if sq is not None:
                input_embed_sq = sq if input_embed_sq is None else (input_embed_sq + sq)
        if input_embed_sq is not None:
            metrics["input_memory_embed_grad_norm"] = input_embed_sq.sqrt()
        proj_sq = self._module_grad_norm_sq(self.attnres_final_memory_proj)
        if proj_sq is not None:
            metrics["final_memory_proj_grad_norm"] = proj_sq.sqrt()
        input_bigram_proj_sq = self._module_grad_norm_sq(self.attnres_input_bigram_proj)
        if input_bigram_proj_sq is not None:
            metrics["input_memory_bigram_proj_grad_norm"] = input_bigram_proj_sq.sqrt()
        input_trigram_proj_sq = self._module_grad_norm_sq(self.attnres_input_trigram_proj)
        if input_trigram_proj_sq is not None:
            metrics["input_memory_trigram_proj_grad_norm"] = input_trigram_proj_sq.sqrt()
        input_smear_gate_sq = self._grad_norm_sq(self.attnres_input_smear_gate)
        if input_smear_gate_sq is not None:
            metrics["input_memory_smear_gate_grad_norm"] = input_smear_gate_sq.sqrt()
        gate_bias_sq = self._grad_norm_sq(self.attnres_final_memory_gate_bias)
        if gate_bias_sq is not None:
            metrics["final_memory_gate_bias_grad_norm"] = gate_bias_sq.sqrt()
        gate_proj_sq = self._module_grad_norm_sq(self.attnres_final_memory_gate_proj)
        if gate_proj_sq is not None:
            metrics["final_memory_gate_proj_grad_norm"] = gate_proj_sq.sqrt()
        query_bias_sq = self._grad_norm_sq(self.attnres_final_memory_query_bias)
        if query_bias_sq is not None:
            metrics["final_memory_query_bias_grad_norm"] = query_bias_sq.sqrt()
        query_proj_sq = self._module_grad_norm_sq(self.attnres_final_memory_query_proj)
        if query_proj_sq is not None:
            metrics["final_memory_query_proj_grad_norm"] = query_proj_sq.sqrt()
        unified_bias_sq = self._grad_norm_sq(self.attnres_final_memory_unified_bias)
        if unified_bias_sq is not None:
            metrics["final_memory_unified_bias_grad_norm"] = unified_bias_sq.sqrt()
        unified_proj_sq = self._module_grad_norm_sq(self.attnres_final_memory_unified_proj)
        if unified_proj_sq is not None:
            metrics["final_memory_unified_proj_grad_norm"] = unified_proj_sq.sqrt()
        q_mod_sq = self._module_grad_norm_sq(self.attnres_final_memory_q_mod_proj)
        if q_mod_sq is not None:
            metrics["final_memory_q_mod_proj_grad_norm"] = q_mod_sq.sqrt()
        k_mod_sq = self._module_grad_norm_sq(self.attnres_final_memory_k_mod_proj)
        if k_mod_sq is not None:
            metrics["final_memory_k_mod_proj_grad_norm"] = k_mod_sq.sqrt()
        v_mod_sq = self._module_grad_norm_sq(self.attnres_final_memory_v_mod_proj)
        if v_mod_sq is not None:
            metrics["final_memory_v_mod_proj_grad_norm"] = v_mod_sq.sqrt()
        vres_sq = self._module_grad_norm_sq(self.attnres_final_memory_vres_proj)
        if vres_sq is not None:
            metrics["final_memory_vres_proj_grad_norm"] = vres_sq.sqrt()
        logit_mod_sq = self._module_grad_norm_sq(self.attnres_final_memory_logit_mod_proj)
        if logit_mod_sq is not None:
            metrics["final_memory_logit_mod_proj_grad_norm"] = logit_mod_sq.sqrt()
        return metrics or None

    def _observe_lm_head_rotation(self, logits, targets):
        if not self.lm_head_rotation_enabled or self.lm_head_rotation_buffer is None:
            return
        grads = sample_logit_grads(
            logits=logits.detach(),
            targets=targets.detach(),
            max_positions=self.config.lm_head_rotation_max_positions,
            ignore_index=-1,
            normalize_rows=self.config.lm_head_rotation_normalize_rows,
        )
        self.lm_head_rotation_buffer.add(grads)

    @torch.no_grad()
    def maybe_rotate_lm_head(self, step, optimizer=None):
        if not self.lm_head_rotation_enabled or self.lm_head_rotation_buffer is None:
            return None
        if step % self.lm_head_rotation_every != 0 or self.lm_head_rotation_buffer.n_rows <= 0:
            return None
        grads = self.lm_head_rotation_buffer.as_matrix(device=self.lm_head.weight.device)
        stats = rotate_lm_head_towards_lost_dirs_(
            self.lm_head,
            grads=grads,
            rank=self.lm_head_rotation_rank,
            alpha=self.lm_head_rotation_alpha,
            optimizer=optimizer,
        )
        self.lm_head_rotation_buffer.clear()
        if stats is None:
            return None
        self.last_lm_head_rotation_stats = {
            "lost_before": torch.tensor(stats["lost_before"], device=self.lm_head.weight.device),
            "lost_after": torch.tensor(stats["lost_after"], device=self.lm_head.weight.device),
            "rows_used": torch.tensor(float(stats["rows_used"]), device=self.lm_head.weight.device),
            "rank": torch.tensor(float(stats["rank"]), device=self.lm_head.weight.device),
            "alpha": torch.tensor(float(stats["alpha"]), device=self.lm_head.weight.device),
        }
        self.lm_head_rotation_updates += 1
        return stats

    def _forward_full_attnres_hidden(self, idx, embedding, cos_sin):
        logical_outputs = []
        input_stats = {"latest": [], "x0": [], "maxprob": [], "entropy": [], "register": []}
        final_stats = None
        final_memory_stats = None
        for block_index, block in enumerate(self.transformer.h):
            ve = self.value_embeds[str(block_index)](idx) if str(block_index) in self.value_embeds else None
            attn_layer_index = 2 * block_index
            attn_bias_input = logical_outputs[-1] if logical_outputs else embedding
            attn_extra_registers, attn_extra_bias = self._build_attnres_token_registers(idx, attn_layer_index, bias_input=attn_bias_input)
            attn_extra_query = self._build_attnres_token_query(idx, attn_layer_index)
            attn_input, attn_weights, attn_meta = self.attnres_mixer.layer_input_with_weights(
                embedding=embedding,
                prior_layer_outputs=logical_outputs,
                layer_index=attn_layer_index,
                include_registers=self._attnres_use_registers("attn_only"),
                extra_registers=attn_extra_registers,
                extra_register_bias=attn_extra_bias,
                extra_query=attn_extra_query,
                extra_register_weight_cap=self.attnres_token_register_weight_cap,
            )
            stats = self._summarize_attnres_weights(attn_weights, **attn_meta)
            for key in input_stats:
                input_stats[key].append(stats[key])
            attn_output = block.attn_output(attn_input, ve, cos_sin, self.window_sizes[block_index])
            logical_outputs.append(attn_output)

            mlp_layer_index = attn_layer_index + 1
            mlp_extra_registers, mlp_extra_bias = self._build_attnres_token_registers(idx, mlp_layer_index, bias_input=logical_outputs[-1])
            mlp_extra_query = self._build_attnres_token_query(idx, mlp_layer_index)
            mlp_input, mlp_weights, mlp_meta = self.attnres_mixer.layer_input_with_weights(
                embedding=embedding,
                prior_layer_outputs=logical_outputs,
                layer_index=mlp_layer_index,
                include_registers=self._attnres_use_registers("mlp_only"),
                extra_registers=mlp_extra_registers,
                extra_register_bias=mlp_extra_bias,
                extra_query=mlp_extra_query,
                extra_register_weight_cap=self.attnres_token_register_weight_cap,
            )
            stats = self._summarize_attnres_weights(mlp_weights, **mlp_meta)
            for key in input_stats:
                input_stats[key].append(stats[key])
            mlp_output = block.mlp_output(mlp_input)
            logical_outputs.append(mlp_output)

        final_bias_input = logical_outputs[-1] if logical_outputs else embedding
        final_extra_registers, final_extra_bias = self._build_attnres_token_registers(idx, self.attnres_num_logical_layers, bias_input=final_bias_input)
        final_extra_query = self._build_attnres_token_query(idx, self.attnres_num_logical_layers)
        final_query_stats = None
        final_source_logit_bias = None
        final_source_key_scale = None
        final_source_value_scale = None
        final_source_value_residual = None
        final_mod_stats = None
        if self._attnres_use_unified_final_memory():
            final_extra_registers = self._build_attnres_final_memory_tokens(idx)
            final_extra_bias = self._build_attnres_final_memory_unified_bias(final_bias_input)
            final_extra_query = None
        elif self._attnres_use_query_final_memory():
            final_extra_query, final_query_stats = self._build_attnres_final_memory_query(idx, final_bias_input)
        elif self._attnres_use_modulated_final_memory():
            (
                final_extra_query,
                final_source_logit_bias,
                final_source_key_scale,
                final_source_value_scale,
                final_source_value_residual,
                final_mod_stats,
            ) = self._build_attnres_final_memory_modulation(idx)
        final_include_registers = self._attnres_use_registers("final_only") or self._attnres_use_unified_final_memory()
        x, final_weights, final_meta = self.attnres_mixer.final_output_with_weights(
            embedding=embedding,
            layer_outputs=logical_outputs,
            include_registers=final_include_registers,
            extra_registers=final_extra_registers,
            extra_register_bias=final_extra_bias,
            extra_query=final_extra_query,
            source_logit_bias=final_source_logit_bias,
            source_key_scale=final_source_key_scale,
            source_value_scale=final_source_value_scale,
            source_value_residual=final_source_value_residual,
            extra_register_weight_cap=self.attnres_final_memory_gate_cap if self._attnres_use_unified_final_memory() else self.attnres_token_register_weight_cap,
        )
        final_stats = self._summarize_attnres_weights(final_weights, **final_meta)
        if self._attnres_use_unified_final_memory():
            final_memory_stats = self._summarize_attnres_extra_source_weights(
                final_weights,
                final_meta,
                self._attnres_final_memory_token_names(),
            )
        elif self._attnres_use_blend_final_memory():
            blend_input = x
            x, blend_stats = self._apply_attnres_final_memory(idx, x, gate_input=blend_input)
            final_memory_stats = {}
            if final_query_stats is not None:
                final_memory_stats.update(final_query_stats)
            if blend_stats is not None:
                final_memory_stats.update(blend_stats)
            if not final_memory_stats:
                final_memory_stats = None
        else:
            final_memory_stats = final_query_stats if final_query_stats is not None else final_mod_stats
        return x, input_stats, final_stats, final_memory_stats

    def _forward_block_attnres_hidden(self, idx, embedding, cos_sin):
        state = self.attnres_mixer.init_state(embedding)
        input_stats = {"latest": [], "x0": [], "maxprob": [], "entropy": [], "register": []}
        final_stats = None
        final_memory_stats = None
        for block_index, block in enumerate(self.transformer.h):
            ve = self.value_embeds[str(block_index)](idx) if str(block_index) in self.value_embeds else None
            attn_layer_index = 2 * block_index
            attn_bias_input = state.current_sources()[-1]
            attn_extra_registers, attn_extra_bias = self._build_attnres_token_registers(idx, attn_layer_index, bias_input=attn_bias_input)
            attn_extra_query = self._build_attnres_token_query(idx, attn_layer_index)
            attn_input, attn_weights, attn_meta = self.attnres_mixer.layer_input_with_weights(
                state=state,
                layer_index=attn_layer_index,
                include_registers=self._attnres_use_registers("attn_only"),
                extra_registers=attn_extra_registers,
                extra_register_bias=attn_extra_bias,
                extra_query=attn_extra_query,
                extra_register_weight_cap=self.attnres_token_register_weight_cap,
            )
            stats = self._summarize_attnres_weights(attn_weights, **attn_meta)
            for key in input_stats:
                input_stats[key].append(stats[key])
            attn_output = block.attn_output(attn_input, ve, cos_sin, self.window_sizes[block_index])
            self.attnres_mixer.append_layer_output(state, attn_output)

            mlp_layer_index = attn_layer_index + 1
            mlp_bias_input = state.current_sources()[-1]
            mlp_extra_registers, mlp_extra_bias = self._build_attnres_token_registers(idx, mlp_layer_index, bias_input=mlp_bias_input)
            mlp_extra_query = self._build_attnres_token_query(idx, mlp_layer_index)
            mlp_input, mlp_weights, mlp_meta = self.attnres_mixer.layer_input_with_weights(
                state=state,
                layer_index=mlp_layer_index,
                include_registers=self._attnres_use_registers("mlp_only"),
                extra_registers=mlp_extra_registers,
                extra_register_bias=mlp_extra_bias,
                extra_query=mlp_extra_query,
                extra_register_weight_cap=self.attnres_token_register_weight_cap,
            )
            stats = self._summarize_attnres_weights(mlp_weights, **mlp_meta)
            for key in input_stats:
                input_stats[key].append(stats[key])
            mlp_output = block.mlp_output(mlp_input)
            self.attnres_mixer.append_layer_output(state, mlp_output)

        final_bias_input = state.final_sources()[-1]
        final_extra_registers, final_extra_bias = self._build_attnres_token_registers(idx, self.attnres_num_logical_layers, bias_input=final_bias_input)
        final_extra_query = self._build_attnres_token_query(idx, self.attnres_num_logical_layers)
        final_query_stats = None
        final_source_logit_bias = None
        final_source_key_scale = None
        final_source_value_scale = None
        final_source_value_residual = None
        final_mod_stats = None
        if self._attnres_use_unified_final_memory():
            final_extra_registers = self._build_attnres_final_memory_tokens(idx)
            final_extra_bias = self._build_attnres_final_memory_unified_bias(final_bias_input)
            final_extra_query = None
        elif self._attnres_use_query_final_memory():
            final_extra_query, final_query_stats = self._build_attnres_final_memory_query(idx, final_bias_input)
        elif self._attnres_use_modulated_final_memory():
            (
                final_extra_query,
                final_source_logit_bias,
                final_source_key_scale,
                final_source_value_scale,
                final_source_value_residual,
                final_mod_stats,
            ) = self._build_attnres_final_memory_modulation(idx)
        final_include_registers = self._attnres_use_registers("final_only") or self._attnres_use_unified_final_memory()
        x, final_weights, final_meta = self.attnres_mixer.final_output_with_weights(
            state,
            include_registers=final_include_registers,
            extra_registers=final_extra_registers,
            extra_register_bias=final_extra_bias,
            extra_query=final_extra_query,
            source_logit_bias=final_source_logit_bias,
            source_key_scale=final_source_key_scale,
            source_value_scale=final_source_value_scale,
            source_value_residual=final_source_value_residual,
            extra_register_weight_cap=self.attnres_final_memory_gate_cap if self._attnres_use_unified_final_memory() else self.attnres_token_register_weight_cap,
        )
        final_stats = self._summarize_attnres_weights(final_weights, **final_meta)
        if self._attnres_use_unified_final_memory():
            final_memory_stats = self._summarize_attnres_extra_source_weights(
                final_weights,
                final_meta,
                self._attnres_final_memory_token_names(),
            )
        elif self._attnres_use_blend_final_memory():
            blend_input = x
            x, blend_stats = self._apply_attnres_final_memory(idx, x, gate_input=blend_input)
            final_memory_stats = {}
            if final_query_stats is not None:
                final_memory_stats.update(final_query_stats)
            if blend_stats is not None:
                final_memory_stats.update(blend_stats)
            if not final_memory_stats:
                final_memory_stats = None
        else:
            final_memory_stats = final_query_stats if final_query_stats is not None else final_mod_stats
        return x, input_stats, final_stats, final_memory_stats

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        if "wpe" in self.transformer:
            wte += sum(p.numel() for p in self.transformer.wpe.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        attnres_params = list(self.attnres_layers.parameters())
        if self.attnres_mixer is not None:
            attnres_params.extend(list(self.attnres_mixer.parameters()))
        attnres_params.extend(list(self.attnres_token_register_embeds.parameters()))
        attnres_params.extend(list(self.attnres_token_query_embeds.parameters()))
        if self.attnres_token_register_bias is not None:
            attnres_params.append(self.attnres_token_register_bias)
        attnres_params.extend(list(self.attnres_token_register_bias_proj.parameters()))
        if self.attnres_final_memory_embed is not None:
            attnres_params.extend(list(self.attnres_final_memory_embed.parameters()))
        if self.attnres_final_memory_proj is not None:
            attnres_params.extend(list(self.attnres_final_memory_proj.parameters()))
        if self.attnres_final_memory_gate_bias is not None:
            attnres_params.append(self.attnres_final_memory_gate_bias)
        if self.attnres_final_memory_gate_proj is not None:
            attnres_params.extend(list(self.attnres_final_memory_gate_proj.parameters()))
        if self.attnres_final_memory_query_bias is not None:
            attnres_params.append(self.attnres_final_memory_query_bias)
        if self.attnres_final_memory_query_proj is not None:
            attnres_params.extend(list(self.attnres_final_memory_query_proj.parameters()))
        if self.attnres_final_memory_unified_bias is not None:
            attnres_params.append(self.attnres_final_memory_unified_bias)
        if self.attnres_final_memory_unified_proj is not None:
            attnres_params.extend(list(self.attnres_final_memory_unified_proj.parameters()))
        if self.attnres_final_memory_q_mod_proj is not None:
            attnres_params.extend(list(self.attnres_final_memory_q_mod_proj.parameters()))
        if self.attnres_final_memory_k_mod_proj is not None:
            attnres_params.extend(list(self.attnres_final_memory_k_mod_proj.parameters()))
        if self.attnres_final_memory_v_mod_proj is not None:
            attnres_params.extend(list(self.attnres_final_memory_v_mod_proj.parameters()))
        if self.attnres_final_memory_vres_proj is not None:
            attnres_params.extend(list(self.attnres_final_memory_vres_proj.parameters()))
        if self.attnres_final_memory_logit_mod_proj is not None:
            attnres_params.extend(list(self.attnres_final_memory_logit_mod_proj.parameters()))
        if self.attnres_input_bigram_embed is not None:
            attnres_params.extend(list(self.attnres_input_bigram_embed.parameters()))
        if self.attnres_input_bigram_proj is not None:
            attnres_params.extend(list(self.attnres_input_bigram_proj.parameters()))
        if self.attnres_input_trigram_embed is not None:
            attnres_params.extend(list(self.attnres_input_trigram_embed.parameters()))
        if self.attnres_input_trigram_proj is not None:
            attnres_params.extend(list(self.attnres_input_trigram_proj.parameters()))
        if self.attnres_input_smear_gate is not None:
            attnres_params.append(self.attnres_input_smear_gate)
        moda_params = []
        anchorkv_params = []
        for block in self.transformer.h:
            if block.attnres_attn is not None:
                attnres_params.extend(list(block.attnres_attn.parameters()))
            if block.attnres_mlp is not None:
                attnres_params.extend(list(block.attnres_mlp.parameters()))
            if block.attn.moda_enabled:
                if isinstance(block.attn.moda_gate, nn.Parameter):
                    moda_params.append(block.attn.moda_gate)
                moda_params.extend(list(block.attn.c_mk.parameters()) if block.attn.c_mk is not None else [])
                moda_params.extend(list(block.attn.c_mv.parameters()) if block.attn.c_mv is not None else [])
            if isinstance(block.attn.anchorkv_gate, nn.Parameter):
                anchorkv_params.append(block.attn.anchorkv_gate)
            if isinstance(block.attn.anchorkv_logit_bias, nn.Parameter):
                anchorkv_params.append(block.attn.anchorkv_logit_bias)
        attnres_ids = {id(p) for p in attnres_params}
        moda_ids = {id(p) for p in moda_params}
        anchorkv_ids = {id(p) for p in anchorkv_params}
        transformer_matrices = sum(
            p.numel()
            for p in self.transformer.h.parameters()
            if id(p) not in attnres_ids and id(p) not in moda_ids and id(p) not in anchorkv_ids
        )
        attnres = sum(p.numel() for p in attnres_params)
        moda = sum(p.numel() for p in moda_params)
        anchorkv = sum(p.numel() for p in anchorkv_params)
        permix = sum(p.numel() for p in self.permix_layers.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + attnres + moda + anchorkv + permix + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'attnres': attnres, 'moda': moda, 'anchorkv': anchorkv, 'permix': permix,
            'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5,
                        permix_lr=0.1, diffattn_lr=0.01, segattn_lr=0.01, attnres_lr=0.01, moda_lr=0.01, anchorkv_lr=0.01,
                        reference_lr=6e-4, reference_weight_decay=0.1, reference_betas=(0.9, 0.95)):
        if self.config.reference_optimizer:
            decay = set()
            no_decay = set()
            for module_name, module in self.named_modules():
                for param_name, param in module.named_parameters(recurse=False):
                    full_name = f"{module_name}.{param_name}" if module_name else param_name
                    if not param.requires_grad:
                        continue
                    if param_name.endswith("bias"):
                        no_decay.add(full_name)
                    elif param_name.endswith("weight") and isinstance(module, nn.Linear):
                        decay.add(full_name)
                    else:
                        no_decay.add(full_name)
            param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
            decay = {pn for pn in decay if pn in param_dict}
            no_decay = {pn for pn in no_decay if pn in param_dict}
            assert len(decay & no_decay) == 0
            assert len(param_dict.keys() - (decay | no_decay)) == 0
            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": reference_weight_decay},
                {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
            ]
            use_fused = self.transformer.wte.weight.device.type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
            optimizer = torch.optim.AdamW(optim_groups, lr=reference_lr, betas=reference_betas, fused=use_fused)
            for group in optimizer.param_groups:
                group["initial_lr"] = group["lr"]
            return optimizer

        model_dim = self.config.n_embd
        diffattn_params = []
        segattn_params = []
        attnres_params = list(self.attnres_layers.parameters())
        if self.attnres_mixer is not None:
            attnres_params.extend(list(self.attnres_mixer.parameters()))
        attnres_params.extend(list(self.attnres_token_register_embeds.parameters()))
        attnres_params.extend(list(self.attnres_token_query_embeds.parameters()))
        if self.attnres_token_register_bias is not None:
            attnres_params.append(self.attnres_token_register_bias)
        attnres_params.extend(list(self.attnres_token_register_bias_proj.parameters()))
        if self.attnres_final_memory_embed is not None:
            attnres_params.extend(list(self.attnres_final_memory_embed.parameters()))
        if self.attnres_final_memory_proj is not None:
            attnres_params.extend(list(self.attnres_final_memory_proj.parameters()))
        if self.attnres_final_memory_gate_bias is not None:
            attnres_params.append(self.attnres_final_memory_gate_bias)
        if self.attnres_final_memory_gate_proj is not None:
            attnres_params.extend(list(self.attnres_final_memory_gate_proj.parameters()))
        if self.attnres_final_memory_unified_bias is not None:
            attnres_params.append(self.attnres_final_memory_unified_bias)
        if self.attnres_final_memory_unified_proj is not None:
            attnres_params.extend(list(self.attnres_final_memory_unified_proj.parameters()))
        if self.attnres_final_memory_q_mod_proj is not None:
            attnres_params.extend(list(self.attnres_final_memory_q_mod_proj.parameters()))
        if self.attnres_final_memory_k_mod_proj is not None:
            attnres_params.extend(list(self.attnres_final_memory_k_mod_proj.parameters()))
        if self.attnres_final_memory_v_mod_proj is not None:
            attnres_params.extend(list(self.attnres_final_memory_v_mod_proj.parameters()))
        if self.attnres_final_memory_vres_proj is not None:
            attnres_params.extend(list(self.attnres_final_memory_vres_proj.parameters()))
        if self.attnres_final_memory_logit_mod_proj is not None:
            attnres_params.extend(list(self.attnres_final_memory_logit_mod_proj.parameters()))
        if self.attnres_input_bigram_embed is not None:
            attnres_params.extend(list(self.attnres_input_bigram_embed.parameters()))
        if self.attnres_input_bigram_proj is not None:
            attnres_params.extend(list(self.attnres_input_bigram_proj.parameters()))
        if self.attnres_input_trigram_embed is not None:
            attnres_params.extend(list(self.attnres_input_trigram_embed.parameters()))
        if self.attnres_input_trigram_proj is not None:
            attnres_params.extend(list(self.attnres_input_trigram_proj.parameters()))
        if self.attnres_input_smear_gate is not None:
            attnres_params.append(self.attnres_input_smear_gate)
        moda_params = []
        anchorkv_params = []
        for block in self.transformer.h:
            if block.attn.lam_proj is not None:
                diffattn_params.extend(list(block.attn.lam_proj.parameters()))
            if block.attn.moda_enabled:
                if isinstance(block.attn.moda_gate, nn.Parameter):
                    moda_params.append(block.attn.moda_gate)
                if block.attn.c_mk is not None:
                    moda_params.extend(list(block.attn.c_mk.parameters()))
                if block.attn.c_mv is not None:
                    moda_params.extend(list(block.attn.c_mv.parameters()))
            if isinstance(block.attn.anchorkv_gate, nn.Parameter):
                anchorkv_params.append(block.attn.anchorkv_gate)
            if isinstance(block.attn.anchorkv_logit_bias, nn.Parameter):
                anchorkv_params.append(block.attn.anchorkv_logit_bias)
            if block.segattn is not None:
                segattn_params.extend(list(block.segattn.parameters()))
            if block.attnres_attn is not None:
                attnres_params.extend(list(block.attnres_attn.parameters()))
            if block.attnres_mlp is not None:
                attnres_params.extend(list(block.attnres_mlp.parameters()))
        diffattn_param_ids = {id(p) for p in diffattn_params}
        moda_param_ids = {id(p) for p in moda_params}
        anchorkv_param_ids = {id(p) for p in anchorkv_params}
        segattn_param_ids = {id(p) for p in segattn_params}
        attnres_param_ids = {id(p) for p in attnres_params}
        transformer_params = list(self.transformer.h.parameters())
        excluded_ids = diffattn_param_ids | moda_param_ids | anchorkv_param_ids | segattn_param_ids | attnres_param_ids
        matrix_params = [p for p in transformer_params if id(p) not in excluded_ids and p.ndim >= 2]
        transformer_misc_params = [p for p in transformer_params if id(p) not in excluded_ids and p.ndim < 2]
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        if "wpe" in self.transformer:
            embedding_params.extend(list(self.transformer.wpe.parameters()))
        lm_head_params = list(self.lm_head.parameters())
        permix_params = list(self.permix_layers.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (
            len(matrix_params) + len(transformer_misc_params) + len(diffattn_params) + len(moda_params) + len(anchorkv_params) + len(segattn_params) + len(attnres_params) +
            len(embedding_params) + len(lm_head_params) + len(value_embeds_params) +
            len(permix_params) + len(resid_params) + len(x0_params)
        )
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=transformer_misc_params, lr=diffattn_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=diffattn_params, lr=diffattn_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=moda_params, lr=moda_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=anchorkv_params, lr=anchorkv_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=segattn_params, lr=segattn_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=attnres_params, lr=attnres_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        if permix_params:
            param_groups.append(
                dict(kind='adamw', params=permix_params, lr=permix_lr * dmodel_lr_scale,
                     betas=adam_betas, eps=1e-10, weight_decay=0.0)
            )
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        x = self.transformer.wte(idx)
        x, input_memory_stats = self._apply_attnres_input_memory(idx, x)
        self.last_attnres_input_memory_stats = input_memory_stats
        if "wpe" in self.transformer:
            pos = torch.arange(T, dtype=torch.long, device=idx.device).unsqueeze(0)
            x = x + self.transformer.wpe(pos)
        x = self.transformer["drop"](x)
        if self.config.use_rotary:
            assert T <= self.cos.size(1)
            cos_sin = self.cos[:, :T], self.sin[:, :T]
        else:
            cos_sin = None
        if self.config.norm_type == "rms":
            x = norm(x)
        x0 = x
        skip = x0.detach() if self.config.permix_skip_detach else x0
        permix_metrics = {
            "id": [],
            "nonid": [],
            "raw_nonid": [],
            "entropy": [],
            "maxprob": [],
            "swap_frac": [],
            "router_temp": [],
        }
        attnres_metrics = {"latest": [], "x0": [], "maxprob": [], "entropy": []}
        diffattn_metrics = {"lam_mean": [], "lam_max": [], "lam_min": [], "q2_raw_norm": [], "q2_norm": []}
        moda_metrics = {"gate": [], "history": [], "delta_norm": [], "depth_mass": []}
        anchorkv_metrics = {"gate": [], "bias": [], "anchor_mass": [], "delta_norm": []}
        segattn_metrics = {"diag": [], "offdiag": [], "entropy": [], "gate": []}
        attnres_final_stats = None
        attnres_final_memory_stats = None
        if self.attnres_full_enabled:
            x, attnres_metrics, attnres_final_stats, attnres_final_memory_stats = self._forward_full_attnres_hidden(idx, x, cos_sin)
        elif self.attnres_block_enabled:
            x, attnres_metrics, attnres_final_stats, attnres_final_memory_stats = self._forward_block_attnres_hidden(idx, x, cos_sin)
        else:
            history = [x]
            moda_cache_history = []
            anchor_cache = None
            for i, block in enumerate(self.transformer.h):
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
                if self.permix_enabled:
                    x, skip = self.permix_layers[i](x, skip)
                    layer_stats = self.permix_layers[i].last_stats
                    for key in permix_metrics:
                        permix_metrics[key].append(layer_stats[key])
                if self.attnres_legacy_full_enabled:
                    if self.config.attnres_history > 0 and len(history) > self.config.attnres_history:
                        states = [history[0]] + history[-self.config.attnres_history:]
                    else:
                        states = history
                    states = states + [x]
                    x = self.attnres_layers[i](states, x)
                    layer_stats = self.attnres_layers[i].last_stats
                    for key in attnres_metrics:
                        if key in layer_stats:
                            attnres_metrics[key].append(layer_stats[key])
                ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
                block_history = None
                if self.attnres_legacy_block_enabled:
                    if self.config.attnres_history > 0 and len(history) > self.config.attnres_history:
                        block_history = [history[0]] + history[-self.config.attnres_history:]
                    else:
                        block_history = history
                moda_caches = None
                if block.attn.moda_enabled and moda_cache_history:
                    if self.config.moda_history > 0 and len(moda_cache_history) > self.config.moda_history:
                        moda_caches = moda_cache_history[-self.config.moda_history:]
                    else:
                        moda_caches = moda_cache_history
                x = block(x, ve, cos_sin, self.window_sizes[i], block_history=block_history, moda_caches=moda_caches, anchor_cache=anchor_cache)
                history.append(x)
                if (
                    block.attn.last_anchorkv_cache is not None
                    and anchor_cache is None
                    and (i + 1) == self.config.anchorkv_anchor_layer
                ):
                    anchor_cache = block.attn.last_anchorkv_cache
                if block.attn.last_moda_cache is not None:
                    moda_cache_history.append(block.attn.last_moda_cache)
                if block.last_attnres_stats is not None:
                    for key in attnres_metrics:
                        if key in block.last_attnres_stats:
                            attnres_metrics[key].append(block.last_attnres_stats[key])
                if block.attn.last_diffattn_stats is not None:
                    for key in diffattn_metrics:
                        diffattn_metrics[key].append(block.attn.last_diffattn_stats[key])
                if block.attn.last_moda_stats is not None:
                    for key in moda_metrics:
                        moda_metrics[key].append(block.attn.last_moda_stats[key])
                if block.attn.last_anchorkv_stats is not None:
                    for key in anchorkv_metrics:
                        anchorkv_metrics[key].append(block.attn.last_anchorkv_stats[key])
                if block.segattn is not None and block.segattn.last_stats is not None:
                    for key in segattn_metrics:
                        if key in block.segattn.last_stats:
                            segattn_metrics[key].append(block.segattn.last_stats[key])
                if self.permix_enabled:
                    skip = (1 - self.config.permix_skip_ema) * skip + self.config.permix_skip_ema * x
                    if self.config.permix_skip_detach:
                        skip = skip.detach()
        if "ln_f" in self.transformer:
            x = self.transformer["ln_f"](x)
        else:
            x = norm(x)
        if self.permix_enabled:
            self.last_permix_metrics = {
                key: torch.stack(values).mean().detach() for key, values in permix_metrics.items()
            }
            raw_nonid_mean = torch.stack(permix_metrics["raw_nonid"]).mean()
            entropy_mean = torch.stack(permix_metrics["entropy"]).mean()
            aux_loss = x.new_zeros(())
            if self.config.permix_target_raw_nonid_coef > 0:
                raw_target = raw_nonid_mean.new_tensor(self.config.permix_target_raw_nonid)
                aux_loss = aux_loss + self.config.permix_target_raw_nonid_coef * (raw_nonid_mean - raw_target).square()
            if self.config.permix_entropy_coef > 0:
                aux_loss = aux_loss - self.config.permix_entropy_coef * entropy_mean
            aux_loss = aux_loss * self.permix_aux_scale
            self.last_permix_metrics["aux_loss"] = aux_loss.detach()
        else:
            self.last_permix_metrics = None
            aux_loss = x.new_zeros(())
        if attnres_metrics["latest"]:
            self.last_attnres_metrics = {
                f"layer_{key}": torch.stack(values).mean().detach() for key, values in attnres_metrics.items()
            }
            if attnres_final_stats is not None:
                for key, value in attnres_final_stats.items():
                    self.last_attnres_metrics[f"final_{key}"] = value.detach()
            if attnres_final_memory_stats is not None:
                for key, value in attnres_final_memory_stats.items():
                    self.last_attnres_metrics[key] = value.detach()
            if self.last_attnres_input_memory_stats is not None:
                for key, value in self.last_attnres_input_memory_stats.items():
                    self.last_attnres_metrics[key] = value.detach()
            attnres_aux = x.new_zeros(())
            if self.config.attnres_target_latest_coef > 0:
                latest_mean = torch.stack(attnres_metrics["latest"]).mean()
                target_latest = latest_mean.new_tensor(self.config.attnres_target_latest)
                attnres_aux = attnres_aux + self.config.attnres_target_latest_coef * (latest_mean - target_latest).square()
            if self.config.attnres_target_register_coef > 0:
                layer_register = torch.stack(attnres_metrics["register"]).mean()
                final_register = (
                    attnres_final_stats["register"]
                    if attnres_final_stats is not None
                    else x.new_zeros(())
                )

                def band_penalty(value, lower, upper):
                    penalty = value.new_zeros(())
                    if lower > 0:
                        penalty = penalty + F.relu(value.new_tensor(lower) - value).square()
                    if upper > 0:
                        penalty = penalty + F.relu(value - value.new_tensor(upper)).square()
                    return penalty

                attnres_aux = attnres_aux + self.config.attnres_target_register_coef * (
                    band_penalty(
                        layer_register,
                        self.config.attnres_target_register_layer_min,
                        self.config.attnres_target_register_layer_max,
                    )
                    + band_penalty(
                        final_register,
                        self.config.attnres_target_register_final_min,
                        self.config.attnres_target_register_final_max,
                    )
                )
            spread_loss = self._compute_attnres_token_register_spread_loss(idx)
            if spread_loss is not None:
                attnres_aux = attnres_aux + self.attnres_token_register_spread_coef * spread_loss
                self.last_attnres_metrics["token_spread_loss"] = spread_loss.detach()
            self.last_attnres_metrics["aux_loss"] = attnres_aux.detach()
        else:
            self.last_attnres_metrics = None
            attnres_aux = x.new_zeros(())
        if diffattn_metrics["lam_mean"]:
            self.last_diffattn_metrics = {
                key: torch.stack(values).mean().detach() for key, values in diffattn_metrics.items()
            }
        else:
            self.last_diffattn_metrics = None
        if moda_metrics["gate"]:
            self.last_moda_metrics = {
                key: torch.stack(values).mean().detach() for key, values in moda_metrics.items()
            }
        else:
            self.last_moda_metrics = None
        if anchorkv_metrics["gate"]:
            self.last_anchorkv_metrics = {
                key: torch.stack(values).mean().detach() for key, values in anchorkv_metrics.items()
            }
        else:
            self.last_anchorkv_metrics = None
        if segattn_metrics["diag"]:
            self.last_segattn_metrics = {
                key: torch.stack(values).mean().detach() for key, values in segattn_metrics.items() if values
            }
        else:
            self.last_segattn_metrics = None

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            self._observe_lm_head_rotation(logits, targets)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            if self.permix_enabled:
                loss = loss + aux_loss
            if self.attnres_enabled:
                loss = loss + attnres_aux
            return loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

def adamw_step_impl(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


def muon_step_impl(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                   momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


if USE_COMPILE:
    adamw_step_fused = torch.compile(adamw_step_impl, dynamic=False, fullgraph=True)
    muon_step_fused = torch.compile(muon_step_impl, dynamic=False, fullgraph=True)
else:
    adamw_step_fused = adamw_step_impl
    muon_step_fused = muon_step_impl


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = env_int("AUTORESEARCH_ASPECT_RATIO", 64)       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = env_int("AUTORESEARCH_HEAD_DIM", 128)              # target head dimension for attention
WINDOW_PATTERN = env_str("AUTORESEARCH_WINDOW_PATTERN", "SSSL") # sliding window pattern: L=full, S=half context
DIFFATTN_MODE = env_str("AUTORESEARCH_DIFFATTN_MODE", "off").lower() # off | v2
DIFFATTN_Q2_MODE = env_str("AUTORESEARCH_DIFFATTN_Q2_MODE", "wo").lower() # wo
DIFFATTN_KV_SHARE = env_flag("AUTORESEARCH_DIFFATTN_KV_SHARE", True)
DIFFATTN_LAM_INIT = env_float("AUTORESEARCH_DIFFATTN_LAM_INIT", -2.0)
DIFFATTN_LAM_ROUTER_DIM = env_int("AUTORESEARCH_DIFFATTN_LAM_ROUTER_DIM", 0)
DIFFATTN_WO_INIT_STD = env_float("AUTORESEARCH_DIFFATTN_WO_INIT_STD", 0.0)
MODA_MODE = env_str("AUTORESEARCH_MODA_MODE", "off").lower() # off | on
MODA_HISTORY = env_int("AUTORESEARCH_MODA_HISTORY", 0)       # 0 = all previous layers, else last k only
MODA_GATE_INIT = env_float("AUTORESEARCH_MODA_GATE_INIT", -4.0)
MODA_KV_MODE = env_str("AUTORESEARCH_MODA_KV_MODE", "shared").lower() # shared | separate
ANCHORKV_MODE = env_str("AUTORESEARCH_ANCHORKV_MODE", "off").lower() # off | on
ANCHORKV_ANCHOR_LAYER = env_int("AUTORESEARCH_ANCHORKV_ANCHOR_LAYER", 1)
ANCHORKV_GATE_INIT = env_float("AUTORESEARCH_ANCHORKV_GATE_INIT", -5.0)
ANCHORKV_ANCHOR_BIAS_INIT = env_float("AUTORESEARCH_ANCHORKV_ANCHOR_BIAS_INIT", -6.0)
ATTNRES_MODE = env_str("AUTORESEARCH_ATTNRES_MODE", "off").lower() # off | full | block | legacy_full | legacy_block
NANOGPT_REF = env_flag("AUTORESEARCH_NANOGPT_REF", False)
ATTNRES_BLOCK_SIZE = env_int("AUTORESEARCH_ATTNRES_BLOCK_SIZE", 4)
ATTNRES_EPS = env_float("AUTORESEARCH_ATTNRES_EPS", 1e-8)
ATTNRES_WEIGHT_MODE = env_str("AUTORESEARCH_ATTNRES_WEIGHT_MODE", "softmax").lower()  # softmax | sigmoid_norm | diffv2
ATTNRES_DIFF_LAM_INIT = env_float("AUTORESEARCH_ATTNRES_DIFF_LAM_INIT", -2.0)
ATTNRES_NUM_REGISTERS = env_int("AUTORESEARCH_ATTNRES_NUM_REGISTERS", 0)
ATTNRES_REGISTER_MODE = env_str("AUTORESEARCH_ATTNRES_REGISTER_MODE", "learned").lower()  # learned | null
ATTNRES_REGISTER_SCOPE = env_str("AUTORESEARCH_ATTNRES_REGISTER_SCOPE", "all").lower()  # all | attn_only | mlp_only | final_only
ATTNRES_REGISTER_BIAS_INIT = env_float("AUTORESEARCH_ATTNRES_REGISTER_BIAS_INIT", -6.0)
ATTNRES_REGISTER_BIAS_START = env_float("AUTORESEARCH_ATTNRES_REGISTER_BIAS_START", ATTNRES_REGISTER_BIAS_INIT)
ATTNRES_REGISTER_BIAS_TARGET = env_float("AUTORESEARCH_ATTNRES_REGISTER_BIAS_TARGET", ATTNRES_REGISTER_BIAS_INIT)
ATTNRES_REGISTER_BIAS_WARMUP_STEPS = env_int("AUTORESEARCH_ATTNRES_REGISTER_BIAS_WARMUP_STEPS", 0)
ATTNRES_REGISTER_INIT_STD = env_float("AUTORESEARCH_ATTNRES_REGISTER_INIT_STD", 0.0)
ATTNRES_HISTORY = env_int("AUTORESEARCH_ATTNRES_HISTORY", 0)       # legacy modes only
ATTNRES_GATE_INIT = env_float("AUTORESEARCH_ATTNRES_GATE_INIT", -2.0)  # legacy modes only
ATTNRES_LATEST_BIAS_INIT = env_float("AUTORESEARCH_ATTNRES_LATEST_BIAS_INIT", 4.0)  # legacy modes only
ATTNRES_QUERY_INIT_STD = env_float("AUTORESEARCH_ATTNRES_QUERY_INIT_STD", 0.0)  # legacy modes only
ATTNRES_TARGET_LATEST = env_float("AUTORESEARCH_ATTNRES_TARGET_LATEST", 0.0)
ATTNRES_TARGET_LATEST_COEF = env_float("AUTORESEARCH_ATTNRES_TARGET_LATEST_COEF", 0.0)
ATTNRES_TARGET_REGISTER_LAYER_MIN = env_float("AUTORESEARCH_ATTNRES_TARGET_REGISTER_LAYER_MIN", 0.0)
ATTNRES_TARGET_REGISTER_LAYER_MAX = env_float("AUTORESEARCH_ATTNRES_TARGET_REGISTER_LAYER_MAX", 0.0)
ATTNRES_TARGET_REGISTER_FINAL_MIN = env_float("AUTORESEARCH_ATTNRES_TARGET_REGISTER_FINAL_MIN", 0.0)
ATTNRES_TARGET_REGISTER_FINAL_MAX = env_float("AUTORESEARCH_ATTNRES_TARGET_REGISTER_FINAL_MAX", 0.0)
ATTNRES_TARGET_REGISTER_COEF = env_float("AUTORESEARCH_ATTNRES_TARGET_REGISTER_COEF", 0.0)
ATTNRES_TOKEN_REGISTERS = env_int("AUTORESEARCH_ATTNRES_TOKEN_REGISTERS", 0)
ATTNRES_TOKEN_REGISTER_MODE = env_str("AUTORESEARCH_ATTNRES_TOKEN_REGISTER_MODE", "off").lower()
ATTNRES_TOKEN_REGISTER_BIAS_MODE = env_str("AUTORESEARCH_ATTNRES_TOKEN_REGISTER_BIAS_MODE", "static").lower()
ATTNRES_TOKEN_REGISTER_VALUE_MODE = env_str("AUTORESEARCH_ATTNRES_TOKEN_REGISTER_VALUE_MODE", "raw").lower()
ATTNRES_TOKEN_REGISTER_SCALE = env_float("AUTORESEARCH_ATTNRES_TOKEN_REGISTER_SCALE", 1.0)
ATTNRES_TOKEN_REGISTER_SCALE_START = env_float("AUTORESEARCH_ATTNRES_TOKEN_REGISTER_SCALE_START", ATTNRES_TOKEN_REGISTER_SCALE)
ATTNRES_TOKEN_REGISTER_SCALE_WARMUP_STEPS = env_int("AUTORESEARCH_ATTNRES_TOKEN_REGISTER_SCALE_WARMUP_STEPS", 0)
ATTNRES_TOKEN_REGISTER_DELTA_SCALE = env_float("AUTORESEARCH_ATTNRES_TOKEN_REGISTER_DELTA_SCALE", 1.0)
ATTNRES_TOKEN_REGISTER_WEIGHT_CAP = env_float("AUTORESEARCH_ATTNRES_TOKEN_REGISTER_WEIGHT_CAP", 0.0)
ATTNRES_TOKEN_REGISTER_SPREAD_COEF = env_float("AUTORESEARCH_ATTNRES_TOKEN_REGISTER_SPREAD_COEF", 0.0)
ATTNRES_TOKEN_REGISTER_SPREAD_SAMPLES = env_int("AUTORESEARCH_ATTNRES_TOKEN_REGISTER_SPREAD_SAMPLES", 0)
ATTNRES_TOKEN_QUERY_MODE = env_str("AUTORESEARCH_ATTNRES_TOKEN_QUERY_MODE", "off").lower()
ATTNRES_TOKEN_QUERY_VALUE_MODE = env_str("AUTORESEARCH_ATTNRES_TOKEN_QUERY_VALUE_MODE", "raw").lower()
ATTNRES_TOKEN_QUERY_SCALE = env_float("AUTORESEARCH_ATTNRES_TOKEN_QUERY_SCALE", 1.0)
ATTNRES_FINAL_MEMORY_MODE = env_str("AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE", "off").lower()
ATTNRES_FINAL_MEMORY_SOURCE = env_str("AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE", "full").lower()
ATTNRES_FINAL_MEMORY_VALUE_MODE = env_str("AUTORESEARCH_ATTNRES_FINAL_MEMORY_VALUE_MODE", "raw").lower()
ATTNRES_FINAL_MEMORY_SCALE = env_float("AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE", 1.0)
ATTNRES_FINAL_MEMORY_GROUPS = env_int("AUTORESEARCH_ATTNRES_FINAL_MEMORY_GROUPS", 1)
ATTNRES_FINAL_MEMORY_RANK = env_int("AUTORESEARCH_ATTNRES_FINAL_MEMORY_RANK", 0)
ATTNRES_FINAL_MEMORY_RESIDUAL_RANK = env_int("AUTORESEARCH_ATTNRES_FINAL_MEMORY_RESIDUAL_RANK", 0)
ATTNRES_FINAL_MEMORY_RESIDUAL_SCALE = env_float("AUTORESEARCH_ATTNRES_FINAL_MEMORY_RESIDUAL_SCALE", 1.0)
ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS = env_int("AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS", 0)
ATTNRES_FINAL_MEMORY_BIGRAM_BANKS = env_int("AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BANKS", 1)
ATTNRES_FINAL_MEMORY_BIGRAM_SCALE = env_float("AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_SCALE", 1.0)
ATTNRES_FINAL_MEMORY_GATE_CAP = env_float("AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_CAP", 0.0)
ATTNRES_FINAL_MEMORY_GATE_BIAS_INIT = env_float("AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_BIAS_INIT", 0.0)
ATTNRES_FINAL_MEMORY_DELTA_SCALE = env_float("AUTORESEARCH_ATTNRES_FINAL_MEMORY_DELTA_SCALE", 1.0)
ATTNRES_FINAL_MEMORY_QUERY_CAP = env_float("AUTORESEARCH_ATTNRES_FINAL_MEMORY_QUERY_CAP", 0.0)
ATTNRES_FINAL_MEMORY_QUERY_BIAS_INIT = env_float("AUTORESEARCH_ATTNRES_FINAL_MEMORY_QUERY_BIAS_INIT", 0.0)
ATTNRES_FINAL_MEMORY_QUERY_DELTA_SCALE = env_float("AUTORESEARCH_ATTNRES_FINAL_MEMORY_QUERY_DELTA_SCALE", 1.0)
ATTNRES_FINAL_MEMORY_Q_MOD_SCALE = env_float("AUTORESEARCH_ATTNRES_FINAL_MEMORY_Q_MOD_SCALE", 0.0)
ATTNRES_FINAL_MEMORY_K_MOD_SCALE = env_float("AUTORESEARCH_ATTNRES_FINAL_MEMORY_K_MOD_SCALE", 0.0)
ATTNRES_FINAL_MEMORY_V_MOD_SCALE = env_float("AUTORESEARCH_ATTNRES_FINAL_MEMORY_V_MOD_SCALE", 0.0)
ATTNRES_FINAL_MEMORY_LOGIT_MOD_SCALE = env_float("AUTORESEARCH_ATTNRES_FINAL_MEMORY_LOGIT_MOD_SCALE", 0.0)
ATTNRES_FINAL_MEMORY_VRES_SCALE = env_float("AUTORESEARCH_ATTNRES_FINAL_MEMORY_VRES_SCALE", 0.0)
ATTNRES_INPUT_MEMORY_MODE = env_str("AUTORESEARCH_ATTNRES_INPUT_MEMORY_MODE", "off").lower()
ATTNRES_INPUT_MEMORY_VALUE_MODE = env_str("AUTORESEARCH_ATTNRES_INPUT_MEMORY_VALUE_MODE", "rmsnorm").lower()
ATTNRES_INPUT_MEMORY_SCALE = env_float("AUTORESEARCH_ATTNRES_INPUT_MEMORY_SCALE", 1.0)
ATTNRES_INPUT_MEMORY_HASH_DIM = env_int("AUTORESEARCH_ATTNRES_INPUT_MEMORY_HASH_DIM", 64)
ATTNRES_INPUT_MEMORY_BIGRAM_BUCKETS = env_int("AUTORESEARCH_ATTNRES_INPUT_MEMORY_BIGRAM_BUCKETS", 0)
ATTNRES_INPUT_MEMORY_BIGRAM_BANKS = env_int("AUTORESEARCH_ATTNRES_INPUT_MEMORY_BIGRAM_BANKS", 1)
ATTNRES_INPUT_MEMORY_BIGRAM_SCALE = env_float("AUTORESEARCH_ATTNRES_INPUT_MEMORY_BIGRAM_SCALE", 1.0)
ATTNRES_INPUT_MEMORY_TRIGRAM_BUCKETS = env_int("AUTORESEARCH_ATTNRES_INPUT_MEMORY_TRIGRAM_BUCKETS", 0)
ATTNRES_INPUT_MEMORY_TRIGRAM_BANKS = env_int("AUTORESEARCH_ATTNRES_INPUT_MEMORY_TRIGRAM_BANKS", 1)
ATTNRES_INPUT_MEMORY_TRIGRAM_SCALE = env_float("AUTORESEARCH_ATTNRES_INPUT_MEMORY_TRIGRAM_SCALE", 1.0)
ATTNRES_INPUT_MEMORY_SMEAR = env_flag("AUTORESEARCH_ATTNRES_INPUT_MEMORY_SMEAR", False)
ATTNRES_INPUT_MEMORY_SMEAR_BIAS_INIT = env_float("AUTORESEARCH_ATTNRES_INPUT_MEMORY_SMEAR_BIAS_INIT", -4.0)
UNTIE_LM_HEAD = env_flag("AUTORESEARCH_UNTIE_LM_HEAD", False)
LM_HEAD_ROTATE_EVERY = env_int("AUTORESEARCH_LM_HEAD_ROTATE_EVERY", 0)
LM_HEAD_ROTATE_MAX_POSITIONS = env_int("AUTORESEARCH_LM_HEAD_ROTATE_MAX_POSITIONS", 64)
LM_HEAD_ROTATE_BUFFER_ROWS = env_int("AUTORESEARCH_LM_HEAD_ROTATE_BUFFER_ROWS", 2048)
LM_HEAD_ROTATE_RANK = env_int("AUTORESEARCH_LM_HEAD_ROTATE_RANK", 16)
LM_HEAD_ROTATE_ALPHA = env_float("AUTORESEARCH_LM_HEAD_ROTATE_ALPHA", 0.05)
LM_HEAD_ROTATE_NORMALIZE_ROWS = env_flag("AUTORESEARCH_LM_HEAD_ROTATE_NORMALIZE_ROWS", True)
LM_HEAD_ROTATE_BUFFER_DEVICE = env_str("AUTORESEARCH_LM_HEAD_ROTATE_BUFFER_DEVICE", "cpu")
PERMIX_MODE = env_str("AUTORESEARCH_PERMIX_MODE", "off").lower() # off | skip
PERMIX_SKIP_DETACH = env_flag("AUTORESEARCH_PERMIX_SKIP_DETACH", True)
PERMIX_SKIP_EMA = env_float("AUTORESEARCH_PERMIX_SKIP_EMA", 0.5)
PERMIX_IDENTITY_BIAS = env_float("AUTORESEARCH_PERMIX_IDENTITY_BIAS", 2.0)
PERMIX_ROUTER_DIM = env_int("AUTORESEARCH_PERMIX_ROUTER_DIM", 0)
PERMIX_MIX_STRENGTH = env_float("AUTORESEARCH_PERMIX_MIX_STRENGTH", 1.0)
PERMIX_ROUTER_TEMPERATURE = env_float("AUTORESEARCH_PERMIX_ROUTER_TEMPERATURE", 1.0)
PERMIX_TARGET_RAW_NONID = env_float("AUTORESEARCH_PERMIX_TARGET_RAW_NONID", 0.0)
PERMIX_TARGET_RAW_NONID_COEF = env_float("AUTORESEARCH_PERMIX_TARGET_RAW_NONID_COEF", 0.0)
PERMIX_ENTROPY_COEF = env_float("AUTORESEARCH_PERMIX_ENTROPY_COEF", 0.0)
PERMIX_AUX_SCALE = env_float("AUTORESEARCH_PERMIX_AUX_SCALE", 1.0)
SEGATTN_MODE = env_str("AUTORESEARCH_SEGATTN_MODE", "off").lower() # off | basic | gated | branch_gated | dual_gated
SEGATTN_NUM_SEGMENTS = env_int("AUTORESEARCH_SEGATTN_NUM_SEGMENTS", 4)
SEGATTN_PROJ_DIM = env_int("AUTORESEARCH_SEGATTN_PROJ_DIM", 0)
SEGATTN_GATE_INIT = env_float("AUTORESEARCH_SEGATTN_GATE_INIT", -2.0)
SEGATTN_GATE_CAP = env_float("AUTORESEARCH_SEGATTN_GATE_CAP", 1.0)
SEGATTN_START_LAYER = env_int("AUTORESEARCH_SEGATTN_START_LAYER", 0)
SEGATTN_LR = env_float("AUTORESEARCH_SEGATTN_LR", 0.01)      # learning rate for segment-attention modules (AdamW)
REFERENCE_LR = env_float("AUTORESEARCH_REFERENCE_LR", 3e-4)
REFERENCE_WEIGHT_DECAY = env_float("AUTORESEARCH_REFERENCE_WEIGHT_DECAY", 0.1)
REFERENCE_BETA1 = env_float("AUTORESEARCH_REFERENCE_BETA1", 0.9)
REFERENCE_BETA2 = env_float("AUTORESEARCH_REFERENCE_BETA2", 0.95)
REFERENCE_DROPOUT = env_float("AUTORESEARCH_REFERENCE_DROPOUT", 0.2)
REFERENCE_BIAS = env_flag("AUTORESEARCH_REFERENCE_BIAS", False)
SEED = env_int("AUTORESEARCH_SEED", 42)

# Optimization
TOTAL_BATCH_SIZE = env_int("AUTORESEARCH_TOTAL_BATCH_SIZE", 2**19) # ~524K tokens per optimizer step
EMBEDDING_LR = env_float("AUTORESEARCH_EMBEDDING_LR", 0.6)      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = env_float("AUTORESEARCH_UNEMBEDDING_LR", 0.004)  # learning rate for lm_head (Adam)
MATRIX_LR = env_float("AUTORESEARCH_MATRIX_LR", 0.04)        # learning rate for matrix parameters (Muon)
PERMIX_LR = env_float("AUTORESEARCH_PERMIX_LR", 0.1)         # learning rate for Permix routers (Adam)
DIFFATTN_LR = env_float("AUTORESEARCH_DIFFATTN_LR", 0.01)    # learning rate for small DiffAttn controllers (Adam)
MODA_LR = env_float("AUTORESEARCH_MODA_LR", 0.01)            # learning rate for MoDA depth-KV params (Adam)
ANCHORKV_LR = env_float("AUTORESEARCH_ANCHORKV_LR", 0.01)    # learning rate for Anchor-KV gate params (Adam)
ATTNRES_LR = env_float("AUTORESEARCH_ATTNRES_LR", 0.01)      # learning rate for AttnRes controllers (Adam)
DIFFATTN_LAM_MODE = env_str("AUTORESEARCH_DIFFATTN_LAM_MODE", "token")
SCALAR_LR = env_float("AUTORESEARCH_SCALAR_LR", 0.5)         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = env_float("AUTORESEARCH_WEIGHT_DECAY", 0.2)      # cautious weight decay for Muon
ADAM_BETAS = (env_float("AUTORESEARCH_ADAM_BETA1", 0.8), env_float("AUTORESEARCH_ADAM_BETA2", 0.95))
WARMUP_RATIO = env_float("AUTORESEARCH_WARMUP_RATIO", 0.0)      # fraction of time budget for LR warmup
WARMDOWN_RATIO = env_float("AUTORESEARCH_WARMDOWN_RATIO", 0.5)    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = env_float("AUTORESEARCH_FINAL_LR_FRAC", 0.0)     # final LR as fraction of initial

# Model size
DEPTH = env_int("AUTORESEARCH_DEPTH", 8)               # number of transformer layers
DEVICE_BATCH_SIZE = env_int("AUTORESEARCH_DEVICE_BATCH_SIZE", 128)  # per-device batch size (reduce if OOM)

# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

H100_BF16_PEAK_FLOPS = 989.5e12

def build_model_config(depth, vocab_size):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    use_reference_backbone = NANOGPT_REF
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        dropout=REFERENCE_DROPOUT if use_reference_backbone else 0.0,
        bias=REFERENCE_BIAS if use_reference_backbone else False,
        use_value_embeds=not use_reference_backbone,
        use_rotary=not use_reference_backbone,
        use_qk_norm=not use_reference_backbone,
        use_absolute_positions=use_reference_backbone,
        norm_type="layernorm" if use_reference_backbone else "rms",
        mlp_activation="gelu" if use_reference_backbone else "relu2",
        reference_init=use_reference_backbone,
        reference_optimizer=use_reference_backbone,
        tie_lm_head=not UNTIE_LM_HEAD,
        lm_head_rotation_every=LM_HEAD_ROTATE_EVERY,
        lm_head_rotation_max_positions=LM_HEAD_ROTATE_MAX_POSITIONS,
        lm_head_rotation_buffer_rows=LM_HEAD_ROTATE_BUFFER_ROWS,
        lm_head_rotation_rank=LM_HEAD_ROTATE_RANK,
        lm_head_rotation_alpha=LM_HEAD_ROTATE_ALPHA,
        lm_head_rotation_normalize_rows=LM_HEAD_ROTATE_NORMALIZE_ROWS,
        lm_head_rotation_buffer_device=LM_HEAD_ROTATE_BUFFER_DEVICE,
        window_pattern=WINDOW_PATTERN,
        diffattn_mode=DIFFATTN_MODE,
        diffattn_q2_mode=DIFFATTN_Q2_MODE,
        diffattn_kv_share=DIFFATTN_KV_SHARE,
        diffattn_lam_mode=DIFFATTN_LAM_MODE,
        diffattn_lam_init=DIFFATTN_LAM_INIT,
        diffattn_lam_router_dim=DIFFATTN_LAM_ROUTER_DIM,
        diffattn_wo_init_std=DIFFATTN_WO_INIT_STD,
        moda_mode=MODA_MODE,
        moda_history=MODA_HISTORY,
        moda_gate_init=MODA_GATE_INIT,
        moda_kv_mode=MODA_KV_MODE,
        anchorkv_mode=ANCHORKV_MODE,
        anchorkv_anchor_layer=ANCHORKV_ANCHOR_LAYER,
        anchorkv_gate_init=ANCHORKV_GATE_INIT,
        anchorkv_anchor_bias_init=ANCHORKV_ANCHOR_BIAS_INIT,
        attnres_mode=ATTNRES_MODE,
        attnres_block_size=ATTNRES_BLOCK_SIZE,
        attnres_eps=ATTNRES_EPS,
        attnres_weight_mode=ATTNRES_WEIGHT_MODE,
        attnres_diff_lam_init=ATTNRES_DIFF_LAM_INIT,
        attnres_num_registers=ATTNRES_NUM_REGISTERS,
        attnres_register_mode=ATTNRES_REGISTER_MODE,
        attnres_register_scope=ATTNRES_REGISTER_SCOPE,
        attnres_register_bias_init=ATTNRES_REGISTER_BIAS_INIT,
        attnres_register_bias_start=ATTNRES_REGISTER_BIAS_START,
        attnres_register_bias_target=ATTNRES_REGISTER_BIAS_TARGET,
        attnres_register_bias_warmup_steps=ATTNRES_REGISTER_BIAS_WARMUP_STEPS,
        attnres_register_init_std=ATTNRES_REGISTER_INIT_STD,
        attnres_history=ATTNRES_HISTORY,
        attnres_gate_init=ATTNRES_GATE_INIT,
        attnres_latest_bias_init=ATTNRES_LATEST_BIAS_INIT,
        attnres_query_init_std=ATTNRES_QUERY_INIT_STD,
        attnres_target_latest=ATTNRES_TARGET_LATEST,
        attnres_target_latest_coef=ATTNRES_TARGET_LATEST_COEF,
        attnres_target_register_layer_min=ATTNRES_TARGET_REGISTER_LAYER_MIN,
        attnres_target_register_layer_max=ATTNRES_TARGET_REGISTER_LAYER_MAX,
        attnres_target_register_final_min=ATTNRES_TARGET_REGISTER_FINAL_MIN,
        attnres_target_register_final_max=ATTNRES_TARGET_REGISTER_FINAL_MAX,
        attnres_target_register_coef=ATTNRES_TARGET_REGISTER_COEF,
        attnres_token_registers=ATTNRES_TOKEN_REGISTERS,
        attnres_token_register_mode=ATTNRES_TOKEN_REGISTER_MODE,
        attnres_token_register_bias_mode=ATTNRES_TOKEN_REGISTER_BIAS_MODE,
        attnres_token_register_value_mode=ATTNRES_TOKEN_REGISTER_VALUE_MODE,
        attnres_token_register_scale=ATTNRES_TOKEN_REGISTER_SCALE,
        attnres_token_register_scale_start=ATTNRES_TOKEN_REGISTER_SCALE_START,
        attnres_token_register_scale_warmup_steps=ATTNRES_TOKEN_REGISTER_SCALE_WARMUP_STEPS,
        attnres_token_register_delta_scale=ATTNRES_TOKEN_REGISTER_DELTA_SCALE,
        attnres_token_register_weight_cap=ATTNRES_TOKEN_REGISTER_WEIGHT_CAP,
        attnres_token_register_spread_coef=ATTNRES_TOKEN_REGISTER_SPREAD_COEF,
        attnres_token_register_spread_samples=ATTNRES_TOKEN_REGISTER_SPREAD_SAMPLES,
        attnres_token_query_mode=ATTNRES_TOKEN_QUERY_MODE,
        attnres_token_query_value_mode=ATTNRES_TOKEN_QUERY_VALUE_MODE,
        attnres_token_query_scale=ATTNRES_TOKEN_QUERY_SCALE,
        attnres_final_memory_mode=ATTNRES_FINAL_MEMORY_MODE,
        attnres_final_memory_source=ATTNRES_FINAL_MEMORY_SOURCE,
        attnres_final_memory_value_mode=ATTNRES_FINAL_MEMORY_VALUE_MODE,
        attnres_final_memory_scale=ATTNRES_FINAL_MEMORY_SCALE,
        attnres_final_memory_groups=ATTNRES_FINAL_MEMORY_GROUPS,
        attnres_final_memory_rank=ATTNRES_FINAL_MEMORY_RANK,
        attnres_final_memory_residual_rank=ATTNRES_FINAL_MEMORY_RESIDUAL_RANK,
        attnres_final_memory_residual_scale=ATTNRES_FINAL_MEMORY_RESIDUAL_SCALE,
        attnres_final_memory_bigram_buckets=ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS,
        attnres_final_memory_bigram_banks=ATTNRES_FINAL_MEMORY_BIGRAM_BANKS,
        attnres_final_memory_bigram_scale=ATTNRES_FINAL_MEMORY_BIGRAM_SCALE,
        attnres_final_memory_gate_cap=ATTNRES_FINAL_MEMORY_GATE_CAP,
        attnres_final_memory_gate_bias_init=ATTNRES_FINAL_MEMORY_GATE_BIAS_INIT,
        attnres_final_memory_delta_scale=ATTNRES_FINAL_MEMORY_DELTA_SCALE,
        attnres_final_memory_query_cap=ATTNRES_FINAL_MEMORY_QUERY_CAP,
        attnres_final_memory_query_bias_init=ATTNRES_FINAL_MEMORY_QUERY_BIAS_INIT,
        attnres_final_memory_query_delta_scale=ATTNRES_FINAL_MEMORY_QUERY_DELTA_SCALE,
        attnres_final_memory_q_mod_scale=ATTNRES_FINAL_MEMORY_Q_MOD_SCALE,
        attnres_final_memory_k_mod_scale=ATTNRES_FINAL_MEMORY_K_MOD_SCALE,
        attnres_final_memory_v_mod_scale=ATTNRES_FINAL_MEMORY_V_MOD_SCALE,
        attnres_final_memory_logit_mod_scale=ATTNRES_FINAL_MEMORY_LOGIT_MOD_SCALE,
        attnres_final_memory_vres_scale=ATTNRES_FINAL_MEMORY_VRES_SCALE,
        attnres_input_memory_mode=ATTNRES_INPUT_MEMORY_MODE,
        attnres_input_memory_value_mode=ATTNRES_INPUT_MEMORY_VALUE_MODE,
        attnres_input_memory_scale=ATTNRES_INPUT_MEMORY_SCALE,
        attnres_input_memory_hash_dim=ATTNRES_INPUT_MEMORY_HASH_DIM,
        attnres_input_memory_bigram_buckets=ATTNRES_INPUT_MEMORY_BIGRAM_BUCKETS,
        attnres_input_memory_bigram_banks=ATTNRES_INPUT_MEMORY_BIGRAM_BANKS,
        attnres_input_memory_bigram_scale=ATTNRES_INPUT_MEMORY_BIGRAM_SCALE,
        attnres_input_memory_trigram_buckets=ATTNRES_INPUT_MEMORY_TRIGRAM_BUCKETS,
        attnres_input_memory_trigram_banks=ATTNRES_INPUT_MEMORY_TRIGRAM_BANKS,
        attnres_input_memory_trigram_scale=ATTNRES_INPUT_MEMORY_TRIGRAM_SCALE,
        attnres_input_memory_smear=ATTNRES_INPUT_MEMORY_SMEAR,
        attnres_input_memory_smear_bias_init=ATTNRES_INPUT_MEMORY_SMEAR_BIAS_INIT,
        permix_mode=PERMIX_MODE,
        permix_skip_detach=PERMIX_SKIP_DETACH,
        permix_skip_ema=PERMIX_SKIP_EMA,
        permix_identity_bias=PERMIX_IDENTITY_BIAS,
        permix_router_dim=PERMIX_ROUTER_DIM,
        permix_mix_strength=PERMIX_MIX_STRENGTH,
        permix_router_temperature=PERMIX_ROUTER_TEMPERATURE,
        permix_target_raw_nonid=PERMIX_TARGET_RAW_NONID,
        permix_target_raw_nonid_coef=PERMIX_TARGET_RAW_NONID_COEF,
        permix_entropy_coef=PERMIX_ENTROPY_COEF,
        permix_aux_scale=PERMIX_AUX_SCALE,
        segattn_mode=SEGATTN_MODE,
        segattn_num_segments=SEGATTN_NUM_SEGMENTS,
        segattn_proj_dim=SEGATTN_PROJ_DIM,
        segattn_gate_init=SEGATTN_GATE_INIT,
        segattn_gate_cap=SEGATTN_GATE_CAP,
        segattn_start_layer=SEGATTN_START_LAYER,
    )

def main():
    t_start = time.time()
    if Tokenizer is None:
        raise RuntimeError(
            "Tokenizer and dataloader dependencies are unavailable. "
            "Run `uv sync` and `bash scripts/prepare_autodl.sh` before training."
        ) from _PREPARE_IMPORT_ERROR
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")

    config = build_model_config(DEPTH, vocab_size)
    print(f"Model config: {asdict(config)}")

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()

    param_counts = model.num_scaling_params()
    print("Parameter counts:")
    for key, value in param_counts.items():
        print(f"  {key:24s}: {value:,}")
    num_params = param_counts['total']
    num_flops_per_token = model.estimate_flops()
    print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
    assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

    optimizer = model.setup_optimizer(
        unembedding_lr=UNEMBEDDING_LR,
        embedding_lr=EMBEDDING_LR,
        scalar_lr=SCALAR_LR,
        adam_betas=ADAM_BETAS,
        matrix_lr=MATRIX_LR,
        weight_decay=WEIGHT_DECAY,
        permix_lr=PERMIX_LR,
        diffattn_lr=DIFFATTN_LR,
        moda_lr=MODA_LR,
        anchorkv_lr=ANCHORKV_LR,
        segattn_lr=SEGATTN_LR,
        attnres_lr=ATTNRES_LR,
        reference_lr=REFERENCE_LR,
        reference_weight_decay=REFERENCE_WEIGHT_DECAY,
        reference_betas=(REFERENCE_BETA1, REFERENCE_BETA2),
    )

    if USE_COMPILE:
        model = torch.compile(model, dynamic=False)

    train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
    x, y, epoch = next(train_loader)  # prefetch first batch

    if MAX_STEPS > 0:
        budget_mode = "steps"
        budget_total = MAX_STEPS
        print(f"Training budget: {MAX_STEPS} steps")
    elif TOKEN_BUDGET > 0:
        budget_mode = "tokens"
        budget_total = TOKEN_BUDGET
        print(f"Training budget: {TOKEN_BUDGET / 1e6:.1f}M tokens")
    else:
        budget_mode = "time"
        budget_total = TRAIN_TIME_BUDGET
        print(f"Training budget: {TRAIN_TIME_BUDGET}s")
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    # Schedules (progress depends on the active training budget)

    def get_lr_multiplier(progress):
        if progress < WARMUP_RATIO:
            return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
        elif progress < 1.0 - WARMDOWN_RATIO:
            return 1.0
        else:
            cooldown = (1.0 - progress) / WARMDOWN_RATIO
            return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

    def get_muon_momentum(step):
        frac = min(step / 300, 1)
        return (1 - frac) * 0.85 + frac * 0.95

    def get_weight_decay(progress):
        return WEIGHT_DECAY * (1 - progress)

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------

    t_start_training = time.time()
    smooth_train_loss = 0
    total_training_time = 0
    step = 0

    def get_progress(total_training_time, step):
        if budget_mode == "steps":
            return min(step / max(MAX_STEPS, 1), 1.0)
        if budget_mode == "tokens":
            total_tokens = step * TOTAL_BATCH_SIZE
            return min(total_tokens / max(TOKEN_BUDGET, 1), 1.0)
        return min(total_training_time / TRAIN_TIME_BUDGET, 1.0)

    def get_remaining(total_training_time, step):
        if budget_mode == "steps":
            return f"{max(MAX_STEPS - step, 0)} steps"
        if budget_mode == "tokens":
            remaining_tokens = max(TOKEN_BUDGET - step * TOTAL_BATCH_SIZE, 0)
            return f"{remaining_tokens / 1e6:.1f}M tok"
        return f"{max(TRAIN_TIME_BUDGET - total_training_time, 0):.0f}s"

    while True:
        torch.cuda.synchronize()
        t0 = time.time()
        progress = get_progress(total_training_time, step)
        model.permix_aux_scale = PERMIX_AUX_SCALE if PERMIX_MODE != "off" else 1.0
        model.set_attnres_runtime_step(step)
        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                loss = model(x, y)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            x, y, epoch = next(train_loader)
        model.last_attnres_grad_metrics = model.collect_attnres_grad_metrics()

        # Progress and schedules
        lrm = get_lr_multiplier(progress)
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(progress)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group.get("kind") == "muon":
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay
        optimizer.step()
        lm_head_rotation_stats = model.maybe_rotate_lm_head(step + 1, optimizer=optimizer)
        model.zero_grad(set_to_none=True)
        if lm_head_rotation_stats is not None:
            print(
                f"[lm_head_rotate] step={step + 1} "
                f"lost={lm_head_rotation_stats['lost_before']:.6f}->{lm_head_rotation_stats['lost_after']:.6f} "
                f"rows={lm_head_rotation_stats['rows_used']} "
                f"rank={lm_head_rotation_stats['rank']} "
                f"alpha={lm_head_rotation_stats['alpha']:.4f}",
                flush=True,
            )

        train_loss_f = train_loss.item()

        # Fast fail: abort if loss is exploding or NaN
        if math.isnan(train_loss_f) or train_loss_f > 100:
            print("FAIL")
            exit(1)

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        if step > 10:
            total_training_time += dt

        # Logging
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
        pct_done = 100 * progress
        tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
        mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
        remaining = get_remaining(total_training_time, step)
        permix_log = ""
        attnres_log = ""
        diffattn_log = ""
        moda_log = ""
        anchorkv_log = ""
        segattn_log = ""
        if model.last_permix_metrics is not None:
            pm = model.last_permix_metrics
            permix_log = (
                f" | pm_id: {pm['id'].item():.2f}"
                f" | pm_nonid: {pm['nonid'].item():.2f}"
                f" | pm_raw: {pm['raw_nonid'].item():.2f}"
                f" | pm_swap: {pm['swap_frac'].item():.2f}"
                f" | pm_ent: {pm['entropy'].item():.2f}"
                f" | pm_aux: {pm['aux_loss'].item():.3f}"
            )
        if model.last_diffattn_metrics is not None:
            dm = model.last_diffattn_metrics
            diffattn_log = (
                f" | dm_lam: {dm['lam_mean'].item():.2f}"
                f" | dm_q2r: {dm['q2_raw_norm'].item():.2f}"
            )
        if model.last_moda_metrics is not None:
            mm = model.last_moda_metrics
            moda_log = (
                f" | md_gate: {mm['gate'].item():.2f}"
                f" | md_hist: {mm['history'].item():.1f}"
                f" | md_dlt: {mm['delta_norm'].item():.2f}"
                f" | md_mass: {mm['depth_mass'].item():.2f}"
            )
        if model.last_anchorkv_metrics is not None:
            am = model.last_anchorkv_metrics
            anchorkv_log = (
                f" | ak_gate: {am['gate'].item():.2f}"
                f" | ak_bias: {am['bias'].item():.2f}"
                f" | ak_mass: {am['anchor_mass'].item():.2f}"
                f" | ak_dlt: {am['delta_norm'].item():.2f}"
            )
        if model.last_attnres_metrics is not None:
            am = model.last_attnres_metrics
            attnres_log = (
                f" | ar_last: {am['layer_latest'].item():.2f}"
                f" | ar_x0: {am['layer_x0'].item():.2f}"
                f" | ar_reg: {am['layer_register'].item():.2f}"
                f" | ar_ent: {am['layer_entropy'].item():.2f}"
            )
            if "final_latest" in am:
                attnres_log += f" | ar_flast: {am['final_latest'].item():.2f}"
            if "final_register" in am:
                attnres_log += f" | ar_freg: {am['final_register'].item():.2f}"
            if "final_memory_gate" in am:
                attnres_log += f" | ar_fmem: {am['final_memory_gate'].item():.2f}"
            if "final_memory_query_gate" in am:
                attnres_log += f" | ar_fq: {am['final_memory_query_gate'].item():.2f}"
            if "memory_total" in am:
                attnres_log += f" | ar_mem: {am['memory_total'].item():.2f}"
            if "input_memory_delta_norm" in am:
                attnres_log += f" | ar_imem: {am['input_memory_delta_norm'].item():.2f}"
            if "input_memory_smear_gate" in am:
                attnres_log += f" | ar_smear: {am['input_memory_smear_gate'].item():.2f}"
            if "aux_loss" in am:
                attnres_log += f" | ar_aux: {am['aux_loss'].item():.3f}"
        if model.last_segattn_metrics is not None:
            sm = model.last_segattn_metrics
            segattn_log = (
                f" | seg_diag: {sm['diag'].item():.2f}"
                f" | seg_off: {sm['offdiag'].item():.2f}"
                f" | seg_ent: {sm['entropy'].item():.2f}"
            )
            if "gate" in sm:
                segattn_log += f" | seg_gate: {sm['gate'].item():.2f}"

        print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining}{permix_log}{attnres_log}{diffattn_log}{moda_log}{anchorkv_log}{segattn_log}    ", end="", flush=True)

        # GC management (Python's GC causes ~500ms stalls)
        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1

        if budget_mode == "steps":
            if step >= MAX_STEPS:
                break
        elif budget_mode == "tokens":
            if step * TOTAL_BATCH_SIZE >= TOKEN_BUDGET:
                break
        else:
            # Time's up — but only stop after warmup steps so we don't count compilation
            if step > 10 and total_training_time >= TRAIN_TIME_BUDGET:
                break

    print()  # newline after \r training log

    total_tokens = step * TOTAL_BATCH_SIZE

    # Final eval
    model.eval()
    model.set_attnres_runtime_step(step)
    with autocast_ctx:
        val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

    # Final summary
    t_end = time.time()
    startup_time = t_start_training - t_start
    steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"mfu_percent:      {steady_state_mfu:.2f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"budget_mode:      {budget_mode}")
    if budget_mode == "time":
        print(f"time_budget_s:    {TRAIN_TIME_BUDGET:.1f}")
    elif budget_mode == "steps":
        print(f"max_steps:        {MAX_STEPS}")
    else:
        print(f"token_budget_M:   {TOKEN_BUDGET / 1e6:.1f}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"depth:            {DEPTH}")
    print(f"seed:             {SEED}")
    print(f"nanogpt_ref:      {int(NANOGPT_REF)}")
    print(f"tie_lm_head:      {int(model.tie_lm_head)}")
    print(f"lm_head_rotation_every: {LM_HEAD_ROTATE_EVERY}")
    print(f"lm_head_rotation_rank:  {LM_HEAD_ROTATE_RANK}")
    print(f"lm_head_rotation_alpha: {LM_HEAD_ROTATE_ALPHA:.6f}")
    print(f"lm_head_rotation_updates: {model.lm_head_rotation_updates}")
    if model.last_lm_head_rotation_stats is not None:
        print(f"lm_head_rotation_lost_before: {model.last_lm_head_rotation_stats['lost_before'].item():.6f}")
        print(f"lm_head_rotation_lost_after:  {model.last_lm_head_rotation_stats['lost_after'].item():.6f}")
        print(f"lm_head_rotation_rows_used:   {model.last_lm_head_rotation_stats['rows_used'].item():.0f}")
    print(f"moda_mode:        {MODA_MODE}")
    if model.last_moda_metrics is not None:
        print(f"moda_history:     {MODA_HISTORY}")
        print(f"moda_gate_init:   {MODA_GATE_INIT:.6f}")
        print(f"moda_kv_mode:     {MODA_KV_MODE}")
        print(f"moda_lr:          {MODA_LR:.6f}")
        print(f"moda_gate:        {model.last_moda_metrics['gate'].item():.6f}")
        print(f"moda_hist_used:   {model.last_moda_metrics['history'].item():.6f}")
        print(f"moda_delta_norm:  {model.last_moda_metrics['delta_norm'].item():.6f}")
        print(f"moda_depth_mass:  {model.last_moda_metrics['depth_mass'].item():.6f}")
    print(f"anchorkv_mode:    {ANCHORKV_MODE}")
    if model.last_anchorkv_metrics is not None:
        print(f"anchorkv_anchor_layer: {ANCHORKV_ANCHOR_LAYER}")
        print(f"anchorkv_gate_init: {ANCHORKV_GATE_INIT:.6f}")
        print(f"anchorkv_anchor_bias_init: {ANCHORKV_ANCHOR_BIAS_INIT:.6f}")
        print(f"anchorkv_lr:      {ANCHORKV_LR:.6f}")
        print(f"anchorkv_gate:    {model.last_anchorkv_metrics['gate'].item():.6f}")
        print(f"anchorkv_bias:    {model.last_anchorkv_metrics['bias'].item():.6f}")
        print(f"anchorkv_mass:    {model.last_anchorkv_metrics['anchor_mass'].item():.6f}")
        print(f"anchorkv_delta_norm: {model.last_anchorkv_metrics['delta_norm'].item():.6f}")
    print(f"attnres_mode:     {ATTNRES_MODE}")
    if model.last_attnres_metrics is not None:
        print(f"attnres_block_size: {ATTNRES_BLOCK_SIZE}")
        print(f"attnres_eps: {ATTNRES_EPS:.6f}")
        print(f"attnres_weight_mode: {ATTNRES_WEIGHT_MODE}")
        print(f"attnres_diff_lam_init: {ATTNRES_DIFF_LAM_INIT:.6f}")
        print(f"attnres_num_registers: {ATTNRES_NUM_REGISTERS}")
        print(f"attnres_register_mode: {ATTNRES_REGISTER_MODE}")
        print(f"attnres_register_scope: {ATTNRES_REGISTER_SCOPE}")
        print(f"attnres_register_bias_init: {ATTNRES_REGISTER_BIAS_INIT:.6f}")
        print(f"attnres_register_bias_start: {ATTNRES_REGISTER_BIAS_START:.6f}")
        print(f"attnres_register_bias_target: {ATTNRES_REGISTER_BIAS_TARGET:.6f}")
        print(f"attnres_register_bias_warmup_steps: {ATTNRES_REGISTER_BIAS_WARMUP_STEPS}")
        print(f"attnres_register_init_std: {ATTNRES_REGISTER_INIT_STD:.6f}")
        print(f"attnres_target_latest: {ATTNRES_TARGET_LATEST:.6f}")
        print(f"attnres_target_latest_coef: {ATTNRES_TARGET_LATEST_COEF:.6f}")
        print(f"attnres_target_register_layer_min: {ATTNRES_TARGET_REGISTER_LAYER_MIN:.6f}")
        print(f"attnres_target_register_layer_max: {ATTNRES_TARGET_REGISTER_LAYER_MAX:.6f}")
        print(f"attnres_target_register_final_min: {ATTNRES_TARGET_REGISTER_FINAL_MIN:.6f}")
        print(f"attnres_target_register_final_max: {ATTNRES_TARGET_REGISTER_FINAL_MAX:.6f}")
        print(f"attnres_target_register_coef: {ATTNRES_TARGET_REGISTER_COEF:.6f}")
        print(f"attnres_token_registers: {ATTNRES_TOKEN_REGISTERS}")
        print(f"attnres_token_register_mode: {ATTNRES_TOKEN_REGISTER_MODE}")
        print(f"attnres_token_register_bias_mode: {ATTNRES_TOKEN_REGISTER_BIAS_MODE}")
        print(f"attnres_token_register_value_mode: {ATTNRES_TOKEN_REGISTER_VALUE_MODE}")
        print(f"attnres_token_register_scale: {ATTNRES_TOKEN_REGISTER_SCALE:.6f}")
        print(f"attnres_token_register_scale_start: {ATTNRES_TOKEN_REGISTER_SCALE_START:.6f}")
        print(f"attnres_token_register_scale_warmup_steps: {ATTNRES_TOKEN_REGISTER_SCALE_WARMUP_STEPS}")
        print(f"attnres_token_register_delta_scale: {ATTNRES_TOKEN_REGISTER_DELTA_SCALE:.6f}")
        print(f"attnres_token_register_weight_cap: {ATTNRES_TOKEN_REGISTER_WEIGHT_CAP:.6f}")
        print(f"attnres_token_register_spread_coef: {ATTNRES_TOKEN_REGISTER_SPREAD_COEF:.6f}")
        print(f"attnres_token_register_spread_samples: {ATTNRES_TOKEN_REGISTER_SPREAD_SAMPLES}")
        print(f"attnres_token_query_mode: {ATTNRES_TOKEN_QUERY_MODE}")
        print(f"attnres_token_query_value_mode: {ATTNRES_TOKEN_QUERY_VALUE_MODE}")
        print(f"attnres_token_query_scale: {ATTNRES_TOKEN_QUERY_SCALE:.6f}")
        print(f"attnres_final_memory_mode: {ATTNRES_FINAL_MEMORY_MODE}")
        print(f"attnres_final_memory_source: {ATTNRES_FINAL_MEMORY_SOURCE}")
        print(f"attnres_final_memory_value_mode: {ATTNRES_FINAL_MEMORY_VALUE_MODE}")
        print(f"attnres_final_memory_scale: {ATTNRES_FINAL_MEMORY_SCALE:.6f}")
        print(f"attnres_final_memory_groups: {ATTNRES_FINAL_MEMORY_GROUPS}")
        print(f"attnres_final_memory_rank: {ATTNRES_FINAL_MEMORY_RANK}")
        print(f"attnres_final_memory_residual_rank: {ATTNRES_FINAL_MEMORY_RESIDUAL_RANK}")
        print(f"attnres_final_memory_residual_scale: {ATTNRES_FINAL_MEMORY_RESIDUAL_SCALE:.6f}")
        print(f"attnres_final_memory_bigram_buckets: {ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS}")
        print(f"attnres_final_memory_bigram_banks: {ATTNRES_FINAL_MEMORY_BIGRAM_BANKS}")
        print(f"attnres_final_memory_bigram_scale: {ATTNRES_FINAL_MEMORY_BIGRAM_SCALE:.6f}")
        print(f"attnres_final_memory_gate_cap: {ATTNRES_FINAL_MEMORY_GATE_CAP:.6f}")
        print(f"attnres_final_memory_gate_bias_init: {ATTNRES_FINAL_MEMORY_GATE_BIAS_INIT:.6f}")
        print(f"attnres_final_memory_delta_scale: {ATTNRES_FINAL_MEMORY_DELTA_SCALE:.6f}")
        print(f"attnres_final_memory_query_cap: {ATTNRES_FINAL_MEMORY_QUERY_CAP:.6f}")
        print(f"attnres_final_memory_query_bias_init: {ATTNRES_FINAL_MEMORY_QUERY_BIAS_INIT:.6f}")
        print(f"attnres_final_memory_query_delta_scale: {ATTNRES_FINAL_MEMORY_QUERY_DELTA_SCALE:.6f}")
        print(f"attnres_final_memory_q_mod_scale: {ATTNRES_FINAL_MEMORY_Q_MOD_SCALE:.6f}")
        print(f"attnres_final_memory_k_mod_scale: {ATTNRES_FINAL_MEMORY_K_MOD_SCALE:.6f}")
        print(f"attnres_final_memory_v_mod_scale: {ATTNRES_FINAL_MEMORY_V_MOD_SCALE:.6f}")
        print(f"attnres_final_memory_logit_mod_scale: {ATTNRES_FINAL_MEMORY_LOGIT_MOD_SCALE:.6f}")
        print(f"attnres_final_memory_vres_scale: {ATTNRES_FINAL_MEMORY_VRES_SCALE:.6f}")
        print(f"attnres_input_memory_mode: {ATTNRES_INPUT_MEMORY_MODE}")
        print(f"attnres_input_memory_value_mode: {ATTNRES_INPUT_MEMORY_VALUE_MODE}")
        print(f"attnres_input_memory_scale: {ATTNRES_INPUT_MEMORY_SCALE:.6f}")
        print(f"attnres_input_memory_hash_dim: {ATTNRES_INPUT_MEMORY_HASH_DIM}")
        print(f"attnres_input_memory_bigram_buckets: {ATTNRES_INPUT_MEMORY_BIGRAM_BUCKETS}")
        print(f"attnres_input_memory_bigram_banks: {ATTNRES_INPUT_MEMORY_BIGRAM_BANKS}")
        print(f"attnres_input_memory_bigram_scale: {ATTNRES_INPUT_MEMORY_BIGRAM_SCALE:.6f}")
        print(f"attnres_input_memory_trigram_buckets: {ATTNRES_INPUT_MEMORY_TRIGRAM_BUCKETS}")
        print(f"attnres_input_memory_trigram_banks: {ATTNRES_INPUT_MEMORY_TRIGRAM_BANKS}")
        print(f"attnres_input_memory_trigram_scale: {ATTNRES_INPUT_MEMORY_TRIGRAM_SCALE:.6f}")
        print(f"attnres_input_memory_smear: {int(ATTNRES_INPUT_MEMORY_SMEAR)}")
        print(f"attnres_input_memory_smear_bias_init: {ATTNRES_INPUT_MEMORY_SMEAR_BIAS_INIT:.6f}")
        print(f"reference_lr:     {REFERENCE_LR:.6f}")
        print(f"attnres_layer_latest:   {model.last_attnres_metrics['layer_latest'].item():.6f}")
        print(f"attnres_layer_x0:       {model.last_attnres_metrics['layer_x0'].item():.6f}")
        print(f"attnres_layer_register: {model.last_attnres_metrics['layer_register'].item():.6f}")
        print(f"attnres_layer_maxprob:  {model.last_attnres_metrics['layer_maxprob'].item():.6f}")
        print(f"attnres_layer_entropy:  {model.last_attnres_metrics['layer_entropy'].item():.6f}")
        if 'final_latest' in model.last_attnres_metrics:
            print(f"attnres_final_latest:   {model.last_attnres_metrics['final_latest'].item():.6f}")
            print(f"attnres_final_x0:       {model.last_attnres_metrics['final_x0'].item():.6f}")
            print(f"attnres_final_register: {model.last_attnres_metrics['final_register'].item():.6f}")
            print(f"attnres_final_maxprob:  {model.last_attnres_metrics['final_maxprob'].item():.6f}")
            print(f"attnres_final_entropy:  {model.last_attnres_metrics['final_entropy'].item():.6f}")
        if 'token_spread_loss' in model.last_attnres_metrics:
            print(f"attnres_token_spread_loss: {model.last_attnres_metrics['token_spread_loss'].item():.6f}")
        if 'final_memory_gate' in model.last_attnres_metrics:
            print(f"attnres_final_memory_gate: {model.last_attnres_metrics['final_memory_gate'].item():.6f}")
            print(f"attnres_final_memory_gate_std: {model.last_attnres_metrics['final_memory_gate_std'].item():.6f}")
            print(f"attnres_final_memory_delta_norm: {model.last_attnres_metrics['final_memory_delta_norm'].item():.6f}")
            print(f"attnres_final_memory_rms: {model.last_attnres_metrics['final_memory_rms'].item():.6f}")
        if 'final_memory_query_gate' in model.last_attnres_metrics:
            print(f"attnres_final_memory_query_gate: {model.last_attnres_metrics['final_memory_query_gate'].item():.6f}")
            print(f"attnres_final_memory_query_gate_std: {model.last_attnres_metrics['final_memory_query_gate_std'].item():.6f}")
            print(f"attnres_final_memory_query_norm: {model.last_attnres_metrics['final_memory_query_norm'].item():.6f}")
            print(f"attnres_final_memory_query_rms: {model.last_attnres_metrics['final_memory_query_rms'].item():.6f}")
            if 'final_memory_query_delta_norm' in model.last_attnres_metrics:
                print(f"attnres_final_memory_query_delta_norm: {model.last_attnres_metrics['final_memory_query_delta_norm'].item():.6f}")
            print(f"attnres_final_memory_rms: {model.last_attnres_metrics['final_memory_rms'].item():.6f}")
        if 'final_memory_mod_q_delta_norm' in model.last_attnres_metrics:
            print(f"attnres_final_memory_mod_q_delta_norm: {model.last_attnres_metrics['final_memory_mod_q_delta_norm'].item():.6f}")
        if 'final_memory_mod_k_delta_norm' in model.last_attnres_metrics:
            print(f"attnres_final_memory_mod_k_delta_norm: {model.last_attnres_metrics['final_memory_mod_k_delta_norm'].item():.6f}")
            print(f"attnres_final_memory_mod_k_scale_mean: {model.last_attnres_metrics['final_memory_mod_k_scale_mean'].item():.6f}")
        if 'final_memory_mod_v_delta_norm' in model.last_attnres_metrics:
            print(f"attnres_final_memory_mod_v_delta_norm: {model.last_attnres_metrics['final_memory_mod_v_delta_norm'].item():.6f}")
            print(f"attnres_final_memory_mod_v_scale_mean: {model.last_attnres_metrics['final_memory_mod_v_scale_mean'].item():.6f}")
        if 'final_memory_mod_logit_delta_norm' in model.last_attnres_metrics:
            print(f"attnres_final_memory_mod_logit_delta_norm: {model.last_attnres_metrics['final_memory_mod_logit_delta_norm'].item():.6f}")
        if 'final_memory_vres_delta_norm' in model.last_attnres_metrics:
            print(f"attnres_final_memory_vres_delta_norm: {model.last_attnres_metrics['final_memory_vres_delta_norm'].item():.6f}")
        if 'final_memory_rms' in model.last_attnres_metrics and 'final_memory_gate' not in model.last_attnres_metrics and 'final_memory_query_gate' not in model.last_attnres_metrics:
            print(f"attnres_final_memory_rms: {model.last_attnres_metrics['final_memory_rms'].item():.6f}")
        if 'input_memory_bigram_rms' in model.last_attnres_metrics:
            print(f"attnres_input_memory_bigram_rms: {model.last_attnres_metrics['input_memory_bigram_rms'].item():.6f}")
        if 'input_memory_trigram_rms' in model.last_attnres_metrics:
            print(f"attnres_input_memory_trigram_rms: {model.last_attnres_metrics['input_memory_trigram_rms'].item():.6f}")
        if 'input_memory_smear_gate' in model.last_attnres_metrics:
            print(f"attnres_input_memory_smear_gate: {model.last_attnres_metrics['input_memory_smear_gate'].item():.6f}")
        if 'input_memory_delta_norm' in model.last_attnres_metrics:
            print(f"attnres_input_memory_delta_norm: {model.last_attnres_metrics['input_memory_delta_norm'].item():.6f}")
        if 'input_memory_rms' in model.last_attnres_metrics:
            print(f"attnres_input_memory_rms: {model.last_attnres_metrics['input_memory_rms'].item():.6f}")
        if 'memory_total' in model.last_attnres_metrics:
            print(f"attnres_final_memory_total: {model.last_attnres_metrics['memory_total'].item():.6f}")
            for key in sorted(model.last_attnres_metrics):
                if key.startswith('memory_') and key != 'memory_total':
                    print(f"attnres_final_{key}: {model.last_attnres_metrics[key].item():.6f}")
        print(f"attnres_aux_loss: {model.last_attnres_metrics['aux_loss'].item():.6f}")
    if model.last_attnres_grad_metrics is not None:
        for key, value in model.last_attnres_grad_metrics.items():
            print(f"attnres_{key}: {value.item():.6f}")
    print(f"diffattn_mode:    {DIFFATTN_MODE}")
    if model.last_diffattn_metrics is not None:
        print(f"diffattn_q2_mode: {DIFFATTN_Q2_MODE}")
        print(f"diffattn_kv_share: {int(DIFFATTN_KV_SHARE)}")
        print(f"diffattn_lam_mode: {DIFFATTN_LAM_MODE}")
        print(f"diffattn_lam_init: {DIFFATTN_LAM_INIT:.6f}")
        print(f"diffattn_wo_init_std: {DIFFATTN_WO_INIT_STD:.6f}")
        print(f"diffattn_lr:      {DIFFATTN_LR:.6f}")
        print(f"diffattn_lam_mean: {model.last_diffattn_metrics['lam_mean'].item():.6f}")
        print(f"diffattn_lam_max:  {model.last_diffattn_metrics['lam_max'].item():.6f}")
        print(f"diffattn_lam_min:  {model.last_diffattn_metrics['lam_min'].item():.6f}")
        print(f"diffattn_q2_raw_norm: {model.last_diffattn_metrics['q2_raw_norm'].item():.6f}")
        print(f"diffattn_q2_norm:  {model.last_diffattn_metrics['q2_norm'].item():.6f}")
    print(f"permix_mode:      {PERMIX_MODE}")
    if model.last_permix_metrics is not None:
        print(f"permix_mix_str:   {PERMIX_MIX_STRENGTH:.6f}")
        print(f"permix_router_t:  {PERMIX_ROUTER_TEMPERATURE:.6f}")
        print(f"permix_raw_tgt:   {PERMIX_TARGET_RAW_NONID:.6f}")
        print(f"permix_raw_coef:  {PERMIX_TARGET_RAW_NONID_COEF:.6f}")
        print(f"permix_ent_coef:  {PERMIX_ENTROPY_COEF:.6f}")
        print(f"permix_aux_scale: {PERMIX_AUX_SCALE:.6f}")
        print(f"permix_id:        {model.last_permix_metrics['id'].item():.6f}")
        print(f"permix_nonid:     {model.last_permix_metrics['nonid'].item():.6f}")
        print(f"permix_raw_nonid: {model.last_permix_metrics['raw_nonid'].item():.6f}")
        print(f"permix_entropy:   {model.last_permix_metrics['entropy'].item():.6f}")
        print(f"permix_maxprob:   {model.last_permix_metrics['maxprob'].item():.6f}")
        print(f"permix_swap_frac: {model.last_permix_metrics['swap_frac'].item():.6f}")
        print(f"permix_aux_loss:  {model.last_permix_metrics['aux_loss'].item():.6f}")
    print(f"segattn_mode:     {SEGATTN_MODE}")
    if model.last_segattn_metrics is not None:
        print(f"segattn_segments: {SEGATTN_NUM_SEGMENTS}")
        print(f"segattn_proj_dim: {SEGATTN_PROJ_DIM}")
        print(f"segattn_gate_init: {SEGATTN_GATE_INIT:.6f}")
        print(f"segattn_gate_cap: {SEGATTN_GATE_CAP:.6f}")
        print(f"segattn_start_layer: {SEGATTN_START_LAYER}")
        print(f"segattn_lr:       {SEGATTN_LR:.6f}")
        print(f"segattn_diag:     {model.last_segattn_metrics['diag'].item():.6f}")
        print(f"segattn_offdiag:  {model.last_segattn_metrics['offdiag'].item():.6f}")
        print(f"segattn_entropy:  {model.last_segattn_metrics['entropy'].item():.6f}")
        if "gate" in model.last_segattn_metrics:
            print(f"segattn_gate:     {model.last_segattn_metrics['gate'].item():.6f}")


if __name__ == "__main__":
    main()
