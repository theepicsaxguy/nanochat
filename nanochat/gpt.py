"""GPT model with optional Recurrent Depth (Huginn-style latent reasoning).

Standard mode:  prelude → coda (original dense transformer, n_recurrent=0)
Recurrent mode: prelude → [core × r] → coda  (latent reasoning, r variable at inference)

Key insight (arXiv 2502.05171, Huginn): iterating a shared core block r times at
test-time gives effective depth far beyond model parameters. A 3.5B model iterated
50× matches a 50B dense model. Brain-like: same circuits recycled for harder problems.

At training time:  r = config.train_recurrence (fixed, e.g. 4)
At inference time: r can be 1,4,8,16,32 — no extra VRAM cost
Truncated backprop: only last k=config.k_backprop iterations receive gradients
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "L"
    dfa_layers: tuple[int, ...] = ()
    dfa_weight: float = 0.0
    dfa_start_frac: float = 0.0
    dfa_end_frac: float = 0.8
    # -------------------------------------------------------------------------
    # Recurrent Depth (Huginn-style latent reasoning, arXiv:2502.05171)
    # n_recurrent > 0 enables: prelude(n_prelude layers) → core(n_recurrent, shared) → coda(n_coda)
    # n_prelude + n_recurrent + n_coda should equal n_layer for VRAM/compute equivalence
    n_prelude: int = 0       # layers before the shared recurrent core (0 = disabled)
    n_recurrent: int = 0     # layers in the shared recurrent core (0 = disabled)
    n_coda: int = 0          # layers after the shared recurrent core
    train_recurrence: int = 4  # fixed iterations during training (torch.compile-friendly)
    k_backprop: int = 4        # last N iterations receive gradients (truncated backprop)
    adaptive_recurrence: bool = False
    adaptive_recurrence_eval_only: bool = True
    ponder_stage_start: float = 0.0
    ponder_warmup_end: float = 0.0
    ponder_lambda: float = 0.0
    ponder_target_frac: float = 0.1
    acttail_weight: float = 0.0
    acttail_target: float = 0.8
    acttail_start_frac: float = 0.0
    acttail_scope: str = "recurrent_ffn_only"
    prores_enable: bool = False
    prores_warmup_frac: float = 0.05
    prores_mode: str = "linear"
    eval_adaptive_exit_threshold: float = 0.0
    eval_max_recurrence: int = 0
    # -------------------------------------------------------------------------
    # Sparse MoE (kept for backward compat with old checkpoints, not active by default)
    moe_num_experts: int = 1
    moe_top_k: int = 1
    moe_layers: tuple[int, ...] = ()
    moe_aux_weight: float = 0.01
    use_grad_checkpoint: bool = False


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx, ve_enabled=True):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        # ve_enabled=False for recurrent core blocks (value embeddings complicate shared-weight recurrence)
        self.ve_gate = (Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
                        if ve_enabled and has_ve(layer_idx, config.n_layer) else None)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None and self.ve_gate is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        intermediate = int(config.n_embd * 3)
        self.c_gate = Linear(config.n_embd, intermediate, bias=False)
        self.c_fc = Linear(config.n_embd, intermediate, bias=False)
        self.c_proj = Linear(intermediate, config.n_embd, bias=False)

    def forward(self, x, return_hidden=False):
        h = F.silu(self.c_gate(x)) * self.c_fc(x)
        y = self.c_proj(h)
        if return_hidden:
            return y, h
        return y


class Block(nn.Module):
    def __init__(self, config, layer_idx, ve_enabled=True):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx, ve_enabled=ve_enabled)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, return_mlp_hidden=False):
        x = x + norm(self.attn(norm(x), ve, cos_sin, window_size, kv_cache))
        if return_mlp_hidden:
            mlp_out, mlp_hidden = self.mlp(norm(x), return_hidden=True)
            x = x + norm(mlp_out)
            return x, mlp_hidden
        x = x + norm(self.mlp(norm(x)))
        return x


class RecurrentAdapter(nn.Module):
    """Projects concat(current_state, initial_state) → n_embd.
    Re-injects the initial context at every recurrent iteration so the model
    remembers where it started — analogous to sensory re-entry in the brain."""
    def __init__(self, n_embd):
        super().__init__()
        self.proj = Linear(2 * n_embd, n_embd, bias=False)

    def forward(self, state, initial):
        return self.proj(torch.cat([state, initial], dim=-1))


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        self.use_recurrence = (config.n_recurrent > 0 and config.n_prelude >= 0 and config.n_coda >= 0
                               and config.n_prelude + config.n_recurrent + config.n_coda == config.n_layer)

        # Compute per-layer window sizes for sliding window attention
        self.window_sizes = self._compute_window_sizes(config)

        # Pad vocab for efficiency (DDP, tensor cores).
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")

        if self.use_recurrence:
            # Recurrent depth architecture: prelude + shared core + coda
            # Value embeddings only in prelude/coda — shared core uses ve_enabled=False
            # to avoid shape complications with parameter sharing across iterations.
            self.transformer = nn.ModuleDict({
                "wte": nn.Embedding(padded_vocab_size, config.n_embd),
                "prelude": nn.ModuleList([
                    Block(config, i, ve_enabled=True)
                    for i in range(config.n_prelude)
                ]),
                "core": nn.ModuleList([
                    Block(config, config.n_prelude + i, ve_enabled=False)
                    for i in range(config.n_recurrent)
                ]),
                "coda": nn.ModuleList([
                    Block(config, config.n_prelude + config.n_recurrent + i, ve_enabled=True)
                    for i in range(config.n_coda)
                ]),
            })
            # Adapter: re-injects initial context at each recurrent iteration
            self.recurrent_adapter = RecurrentAdapter(config.n_embd)
            gate_hidden = max(config.n_embd // 4, 32)
            self.recurrent_gate_heads = nn.ModuleList([
                nn.Sequential(
                    Linear(config.n_embd, gate_hidden, bias=False),
                    nn.SiLU(),
                    Linear(gate_hidden, 1, bias=False),
                )
                for _ in range(max(config.train_recurrence - 1, 0))
            ])
            # Per-layer scalars for prelude + core + coda (n_layer total)
            self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
            self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
            # Layer mix over prelude(1) + coda output(1) = simpler than full layer_mix
            # We just use the final coda output (no layer mixing for recurrent model)
            self.layer_mix = nn.Parameter(torch.zeros(2))  # [prelude_weight, final_weight]
        else:
            # Original dense transformer (backward compatible)
            self.transformer = nn.ModuleDict({
                "wte": nn.Embedding(padded_vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
            })
            self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
            self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
            self.layer_mix = nn.Parameter(torch.zeros(config.n_layer + 1))

        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)

        # Value embeddings (ResFormer-style): only for non-recurrent-core layers
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        if self.use_recurrence:
            # Value embeds for prelude + coda only
            ve_layers = []
            for i in range(config.n_prelude):
                if has_ve(i, config.n_layer):
                    ve_layers.append(i)
            for i in range(config.n_coda):
                real_idx = config.n_prelude + config.n_recurrent + i
                if has_ve(real_idx, config.n_layer):
                    ve_layers.append(real_idx)
            self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in ve_layers})
        else:
            self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})

        # Rotary embeddings (precomputed, over-allocated by 10×)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """Initialize all model parameters."""
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5

        def init_block(block):
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_gate.weight, -s, s)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        if self.use_recurrence:
            for block in self.transformer.prelude:
                init_block(block)
            for block in self.transformer.core:
                init_block(block)
            for block in self.transformer.coda:
                init_block(block)
            # Adapter: small init so first iteration starts close to identity
            torch.nn.init.normal_(self.recurrent_adapter.proj.weight, mean=0.0, std=0.02)
            for gate in self.recurrent_gate_heads:
                for module in gate:
                    if isinstance(module, Linear):
                        torch.nn.init.zeros_(module.weight)
            self.layer_mix.fill_(-10.0)
            self.layer_mix.data[-1] = 0.0  # weight final output
        else:
            for block in self.transformer.h:
                init_block(block)
            self.layer_mix.fill_(-10.0)
            self.layer_mix.data[-1] = 0.0

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)

        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=50000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """Return estimated FLOPs per token (forward + backward) at training recurrence."""
        r = self.config.train_recurrence if self.use_recurrence else 1
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        adapter_numel = self.recurrent_adapter.proj.weight.numel() if self.use_recurrence else 0
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                           self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                           self.layer_mix.numel() + adapter_numel)
        # With recurrence, core params are activated r times per token
        if self.use_recurrence:
            core_params = sum(p.numel() for p in self.transformer.core.parameters())
            non_core_params = (nparams - nparams_exclude - core_params)
            effective_matrix_params = non_core_params + r * core_params
        else:
            effective_matrix_params = nparams - nparams_exclude

        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Effective layers for attention FLOPs
        if self.use_recurrence:
            n_prelude_layers = self.config.n_prelude
            n_coda_layers = self.config.n_coda
            n_core_layers = self.config.n_recurrent
            effective_n_layers = n_prelude_layers + r * n_core_layers + n_coda_layers
        else:
            effective_n_layers = self.config.n_layer

        attn_flops = 0
        for i in range(min(effective_n_layers, len(self.window_sizes))):
            ws = self.window_sizes[min(i, len(self.window_sizes)-1)]
            window = ws[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq

        num_flops_per_token = 6 * effective_matrix_params + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """Return parameter counts for scaling law analysis."""
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.layer_mix.numel()

        if self.use_recurrence:
            transformer_matrices = (sum(p.numel() for p in self.transformer.prelude.parameters()) +
                                    sum(p.numel() for p in self.transformer.core.parameters()) +
                                    sum(p.numel() for p in self.transformer.coda.parameters()) +
                                    sum(p.numel() for p in self.recurrent_adapter.parameters()) +
                                    sum(p.numel() for p in self.recurrent_gate_heads.parameters()))
        else:
            transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())

        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        if self.use_recurrence:
            matrix_params = (list(self.transformer.prelude.parameters()) +
                             list(self.transformer.core.parameters()) +
                             list(self.transformer.coda.parameters()) +
                             list(self.recurrent_adapter.parameters()))
            # Only include gate heads when adaptive recurrence is enabled — without it
            # they never receive gradients and Muon would crash on None gradient.
            if self.config.adaptive_recurrence:
                matrix_params = matrix_params + list(self.recurrent_gate_heads.parameters())
        else:
            matrix_params = list(self.transformer.h.parameters())

        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        mix_params = [self.layer_mix]
        # Gate heads: use Muon when adaptive_recurrence is enabled (they get gradients),
        # otherwise park them in AdamW scalar group — AdamW skips None-gradient params.
        gate_head_params = list(self.recurrent_gate_heads.parameters()) if self.use_recurrence and not self.config.adaptive_recurrence else []

        all_param_count = (len(matrix_params) + len(embedding_params) + len(lm_head_params) +
                           len(value_embeds_params) + len(resid_params) + len(x0_params) + len(mix_params) +
                           len(gate_head_params))
        assert len(list(self.parameters())) == all_param_count

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.9, 0.98), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.9, 0.98), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.9, 0.98), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.9, 0.98), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=mix_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            # Gate heads parked in AdamW when not actively trained (AdamW skips None-grad params)
            *([dict(kind='adamw', params=gate_head_params, lr=scalar_lr, betas=(0.9, 0.98), eps=1e-10, weight_decay=0.0)] if gate_head_params else []),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def _run_block(self, block, x, idx, layer_idx, cos_sin, kv_cache, return_mlp_hidden=False):
        """Run a single block, injecting value embeddings if available."""
        ve = self.value_embeds[str(layer_idx)](idx).to(x.dtype) if str(layer_idx) in self.value_embeds else None
        return block(
            x, ve, cos_sin, self.window_sizes[min(layer_idx, len(self.window_sizes)-1)], kv_cache,
            return_mlp_hidden=return_mlp_hidden,
        )

    def _resolve_ponder_lambda(self, progress, ponder_lambda_override):
        if ponder_lambda_override is not None:
            return ponder_lambda_override
        if self.config.ponder_lambda <= 0.0 or progress is None:
            return self.config.ponder_lambda
        if progress < self.config.ponder_stage_start:
            return 0.0
        if progress >= self.config.ponder_warmup_end:
            return self.config.ponder_lambda
        span = max(self.config.ponder_warmup_end - self.config.ponder_stage_start, 1e-8)
        frac = (progress - self.config.ponder_stage_start) / span
        return self.config.ponder_lambda * frac

    def _resolve_acttail_weight(self, progress, acttail_weight_override):
        if acttail_weight_override is not None:
            return acttail_weight_override
        if progress is None or self.config.acttail_weight <= 0.0:
            return self.config.acttail_weight
        if progress < self.config.acttail_start_frac:
            return 0.0
        return self.config.acttail_weight

    def _get_prores_scales(self, progress, prores_progress_override):
        if not self.config.prores_enable or self.config.prores_warmup_frac <= 0.0:
            return None
        if prores_progress_override is not None:
            progress = prores_progress_override
        if progress is None:
            return None
        if progress >= self.config.prores_warmup_frac:
            return None
        target_layers = list(range(self.config.n_prelude, self.config.n_prelude + self.config.n_recurrent))
        if not target_layers:
            return None
        warm = max(progress / self.config.prores_warmup_frac, 0.0)
        scales = torch.ones(self.config.n_layer, device=self.resid_lambdas.device, dtype=self.resid_lambdas.dtype)
        n_target = len(target_layers)
        positions = torch.arange(n_target, device=scales.device, dtype=scales.dtype)
        if self.config.prores_mode == "linear":
            target_scales = (warm * n_target - positions).clamp_(0.0, 1.0)
        else:
            target_scales = torch.full((n_target,), warm, device=scales.device, dtype=scales.dtype).clamp_(0.0, 1.0)
        scales[target_layers] = target_scales
        return scales

    def _apply_residual_mix(self, x, x0, layer_idx, prores_scales):
        resid_scale = self.resid_lambdas[layer_idx]
        if prores_scales is not None:
            resid_scale = resid_scale * prores_scales[layer_idx]
        return resid_scale * x + self.x0_lambdas[layer_idx] * x0

    def _bottomk_mask_update(self, gate_scores, active_mask):
        prune_frac = float(self.config.ponder_target_frac)
        if prune_frac <= 0.0:
            return active_mask, active_mask.float().mean(dim=1)
        B, T = gate_scores.shape
        prune_k = min(T, max(1, int(round(prune_frac * T))))
        masked_scores = torch.where(active_mask, gate_scores, torch.full_like(gate_scores, float("inf")))
        prune_idx = torch.topk(masked_scores, k=prune_k, dim=-1, largest=False).indices
        active_counts = active_mask.sum(dim=-1)
        allowed_prune = torch.clamp(active_counts - 1, min=0, max=prune_k)
        prune_selector = torch.arange(prune_k, device=gate_scores.device).unsqueeze(0) < allowed_prune.unsqueeze(1)
        prune_mask = torch.zeros_like(active_mask)
        prune_mask.scatter_(1, prune_idx, prune_selector)
        next_active = active_mask & (~prune_mask)
        return next_active, next_active.float().mean(dim=1)

    def _acttail_loss(self, activations):
        if not activations:
            zero = self.lm_head.weight.new_zeros(())
            return zero, zero, zero
        target_sparsity = float(self.config.acttail_target)
        keep_frac = min(max(1.0 - target_sparsity, 1e-3), 1.0)
        chunks = [a.reshape(-1, a.size(-1)) for a in activations]
        flat = torch.cat(chunks, dim=0)
        abs_flat = flat.abs()
        k_keep = min(abs_flat.size(-1), max(1, int(round(keep_frac * abs_flat.size(-1)))))
        if k_keep >= abs_flat.size(-1):
            tail_penalty = abs_flat.new_zeros(())
            active_frac = abs_flat.new_ones(())
            threshold = abs_flat.new_zeros(())
        else:
            topk_vals = torch.topk(abs_flat, k=k_keep, dim=-1, largest=True).values
            threshold = topk_vals[..., -1:]
            keep_mask = abs_flat >= threshold
            tail_penalty = torch.where(keep_mask, torch.zeros_like(abs_flat), abs_flat).mean()
            active_frac = keep_mask.float().mean()
            threshold = threshold.mean()
        return tail_penalty, active_frac, threshold

    def _compute_adaptive_exit_logits(self, x, idx, x0, cos_sin, kv_cache, prores_scales):
        dfa_hidden = []
        n_coda_start = self.config.n_prelude + self.config.n_recurrent
        y = x
        for i, block in enumerate(self.transformer.coda):
            layer_abs = n_coda_start + i
            y = self._apply_residual_mix(y, x0, layer_abs, prores_scales)
            y = self._run_block(block, y, idx, layer_abs, cos_sin, kv_cache)
        y = norm(y)
        logits = self.lm_head(y)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        softcap = 15
        logits = softcap * torch.tanh(logits / softcap)
        return y, logits, dfa_hidden

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', progress=None,
                recurrence: Optional[int] = None, dfa_weight_override: Optional[float] = None,
                adaptive_exit_threshold: Optional[float] = None, max_recurrence: Optional[int] = None,
                ponder_lambda_override: Optional[float] = None, acttail_weight_override: Optional[float] = None,
                prores_progress_override: Optional[float] = None, return_info: bool = False):
        """
        Forward pass.

        recurrence: override recurrence depth at inference (None = use config.train_recurrence).
                    Set to 8, 16, 32 at test time for deeper latent reasoning.
        """
        B, T = idx.size()

        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device
        assert self.cos.dtype == COMPUTE_DTYPE

        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # Embed the tokens
        x = self.transformer.wte(idx)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        info = {}
        prores_scales = self._get_prores_scales(progress, prores_progress_override)

        if self.use_recurrence:
            # -------------------------------------------------------
            # RECURRENT DEPTH FORWARD PASS
            # Architecture: prelude → [adapter + core] × r → coda
            # -------------------------------------------------------
            configured_eval_max = self.config.eval_max_recurrence if self.config.eval_max_recurrence > 0 else None
            if max_recurrence is None:
                max_recurrence = configured_eval_max
            r = recurrence if recurrence is not None else self.config.train_recurrence
            if max_recurrence is not None:
                r = min(r, max_recurrence)
            k_bp = min(r, self.config.k_backprop)  # backprop through last k_bp iterations
            adaptive_training = self.config.adaptive_recurrence and (self.training or not self.config.adaptive_recurrence_eval_only)
            ponder_lambda = self._resolve_ponder_lambda(progress, ponder_lambda_override) if adaptive_training else 0.0
            acttail_weight = self._resolve_acttail_weight(progress, acttail_weight_override)
            if adaptive_exit_threshold is None and not self.training and self.config.eval_adaptive_exit_threshold > 0:
                adaptive_exit_threshold = self.config.eval_adaptive_exit_threshold
            enable_adaptive_exit = (adaptive_exit_threshold is not None and adaptive_exit_threshold > 0 and not self.training)
            adaptive_exit_threshold = float(adaptive_exit_threshold) if adaptive_exit_threshold is not None else None

            # Prelude: standard transformer layers (with x0 injection)
            mix_weights = F.softmax(self.layer_mix, dim=0)
            x_avg = mix_weights[0] * x
            dfa_hidden = []
            for i, block in enumerate(self.transformer.prelude):
                x = self._apply_residual_mix(x, x0, i, prores_scales)
                x = self._run_block(block, x, idx, i, cos_sin, kv_cache)
                if targets is not None and i in self.config.dfa_layers:
                    dfa_hidden.append((i, x))

            initial_state = x  # save prelude output for re-injection at each iteration

            # Recurrent core: shared block run r times
            n_pre = self.config.n_prelude
            active_mask = torch.ones(B, T, dtype=torch.bool, device=idx.device)
            recurrent_active_fracs = []
            recurrent_gate_means = []
            acttail_activations = []
            ponder_terms = []
            prev_exit_log_probs = None
            exit_kl = None
            recurrence_steps_used = 0
            for iteration in range(r):
                prev_x = x
                # Re-inject initial context via adapter (brain's "sensory re-entry")
                x = self.recurrent_adapter(x, initial_state)
                # Apply per-layer scalars for core (uses lambda indices n_pre..n_pre+n_recurrent)
                for j, block in enumerate(self.transformer.core):
                    layer_abs = n_pre + j
                    x = self._apply_residual_mix(x, x0, layer_abs, prores_scales)
                    want_hidden = acttail_weight > 0.0 and self.config.acttail_scope == "recurrent_ffn_only"
                    block_out = self._run_block(block, x, idx, layer_abs, cos_sin, kv_cache, return_mlp_hidden=want_hidden)
                    if want_hidden:
                        x, mlp_hidden = block_out
                        acttail_activations.append(mlp_hidden)
                    else:
                        x = block_out
                    if targets is not None and layer_abs in self.config.dfa_layers:
                        dfa_hidden.append((layer_abs, x))
                    if adaptive_training:
                        x = torch.where(active_mask.unsqueeze(-1), x, prev_x)
                recurrence_steps_used = iteration + 1
                recurrent_active_fracs.append(active_mask.float().mean())

                if adaptive_training and iteration < r - 1 and len(self.recurrent_gate_heads) > 0:
                    gate_head = self.recurrent_gate_heads[min(iteration, len(self.recurrent_gate_heads) - 1)]
                    gate_scores = gate_head(norm(x)).squeeze(-1)
                    recurrent_gate_means.append(torch.sigmoid(gate_scores).mean())
                    ponder_terms.append(torch.sigmoid(gate_scores[active_mask]).mean() if active_mask.any() else gate_scores.new_zeros(()))
                    active_mask, next_active_frac = self._bottomk_mask_update(gate_scores, active_mask)
                    recurrent_active_fracs.append(next_active_frac.mean())

                if enable_adaptive_exit and iteration < r - 1:
                    _, provisional_logits, _ = self._compute_adaptive_exit_logits(x, idx, x0, cos_sin, kv_cache, prores_scales)
                    curr_log_probs = F.log_softmax(provisional_logits, dim=-1)
                    if prev_exit_log_probs is not None:
                        prev_probs = prev_exit_log_probs.exp()
                        exit_kl = (prev_probs * (prev_exit_log_probs - curr_log_probs)).sum(dim=-1).mean()
                        if exit_kl.item() <= adaptive_exit_threshold:
                            prev_exit_log_probs = curr_log_probs
                            break
                    prev_exit_log_probs = curr_log_probs

                # Truncated backprop: detach the running state for early iterations.
                # initial_state (prelude output) is intentionally NOT detached —
                # it must stay connected so prelude parameters receive gradients.
                if iteration < r - k_bp:
                    x = x.detach()

            # Coda: standard transformer layers
            n_coda_start = n_pre + self.config.n_recurrent
            if enable_adaptive_exit and prev_exit_log_probs is not None and recurrence_steps_used < r:
                for i, block in enumerate(self.transformer.coda):
                    layer_abs = n_coda_start + i
                    x = self._apply_residual_mix(x, x0, layer_abs, prores_scales)
                    x = self._run_block(block, x, idx, layer_abs, cos_sin, kv_cache)
                    if targets is not None and layer_abs in self.config.dfa_layers:
                        dfa_hidden.append((layer_abs, x))
            else:
                for i, block in enumerate(self.transformer.coda):
                    layer_abs = n_coda_start + i
                    x = self._apply_residual_mix(x, x0, layer_abs, prores_scales)
                    x = self._run_block(block, x, idx, layer_abs, cos_sin, kv_cache)
                    if targets is not None and layer_abs in self.config.dfa_layers:
                        dfa_hidden.append((layer_abs, x))

            x_avg = x_avg + mix_weights[1] * x  # final output
            x = x_avg
            info.update({
                "avg_recurrence_depth": x.new_tensor(float(recurrence_steps_used)),
                "configured_recurrence_depth": x.new_tensor(float(r)),
                "recurrent_halted_fraction": x.new_tensor(1.0 - (recurrence_steps_used / max(r, 1))),
                "recurrent_active_fraction": torch.stack(recurrent_active_fracs).mean() if recurrent_active_fracs else x.new_tensor(1.0),
                "ponder_gate_mean": torch.stack(recurrent_gate_means).mean() if recurrent_gate_means else x.new_zeros(()),
                "adaptive_exit_kl": exit_kl if exit_kl is not None else x.new_zeros(()),
            })
            if prores_scales is not None:
                info["prores_scale_mean"] = prores_scales[n_pre:n_pre + self.config.n_recurrent].mean()
            else:
                info["prores_scale_mean"] = x.new_tensor(1.0)

        else:
            # -------------------------------------------------------
            # ORIGINAL DENSE FORWARD PASS (backward compatible)
            # -------------------------------------------------------
            mix_weights = F.softmax(self.layer_mix, dim=0)
            x_avg = mix_weights[0] * x
            dfa_hidden = []
            for i, block in enumerate(self.transformer.h):
                x = self._apply_residual_mix(x, x0, i, prores_scales)
                ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
                x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
                x_avg = x_avg + mix_weights[i + 1] * x
                if targets is not None and i in self.config.dfa_layers:
                    dfa_hidden.append((i, x))
            x = x_avg
            info["avg_recurrence_depth"] = x.new_tensor(1.0)
            info["configured_recurrence_depth"] = x.new_tensor(1.0)
            info["recurrent_halted_fraction"] = x.new_zeros(())
            info["recurrent_active_fraction"] = x.new_tensor(1.0)
            info["ponder_gate_mean"] = x.new_zeros(())
            info["adaptive_exit_kl"] = x.new_zeros(())
            info["prores_scale_mean"] = x.new_tensor(1.0)

        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        if kv_cache is not None:
            kv_cache.advance(T)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=loss_reduction)
            if self.config.dfa_weight > 0 and len(dfa_hidden) > 0:
                valid = (targets >= 0)
                target_ids = targets.clamp_min(0)
                target_emb = self.transformer.wte(target_ids).to(x.dtype)
                target_emb = norm(target_emb)

                # dfa_weight_override: pre-computed by caller (avoids torch.compile recompilation
                # from progress float changing every step). Falls back to config.dfa_weight if None.
                if dfa_weight_override is not None:
                    effective_dfa_weight = dfa_weight_override
                elif progress is None:
                    effective_dfa_weight = self.config.dfa_weight
                elif progress <= self.config.dfa_start_frac:
                    effective_dfa_weight = self.config.dfa_weight
                elif progress >= self.config.dfa_end_frac:
                    effective_dfa_weight = 0.0
                else:
                    span = self.config.dfa_end_frac - self.config.dfa_start_frac
                    frac = (progress - self.config.dfa_start_frac) / span
                    effective_dfa_weight = self.config.dfa_weight * (1.0 - frac)

                aux_losses = []
                valid_f = valid.float()
                denom = valid_f.sum().clamp_min(1.0)
                for layer_idx, h in dfa_hidden:
                    h = norm(h)
                    cos_sim = F.cosine_similarity(h, target_emb, dim=-1)
                    aux = (1.0 - cos_sim) * valid_f
                    layer_scale = 1.0 - (layer_idx / max(self.config.n_layer - 1, 1))
                    aux_losses.append(layer_scale * aux.sum() / denom)
                if aux_losses:
                    loss = loss + effective_dfa_weight * torch.stack(aux_losses).mean()
            if self.use_recurrence:
                acttail_weight = self._resolve_acttail_weight(progress, acttail_weight_override)
                if acttail_weight > 0.0 and 'acttail_activations' in locals():
                    acttail_loss, acttail_active_frac, acttail_threshold = self._acttail_loss(acttail_activations)
                    loss = loss + acttail_weight * acttail_loss
                    info["acttail_loss"] = acttail_loss
                    info["acttail_active_fraction"] = acttail_active_frac
                    info["acttail_threshold"] = acttail_threshold
                else:
                    info["acttail_loss"] = loss.new_zeros(())
                    info["acttail_active_fraction"] = loss.new_zeros(())
                    info["acttail_threshold"] = loss.new_zeros(())
                if adaptive_training and ponder_lambda > 0.0 and ponder_terms:
                    ponder_loss = torch.stack(ponder_terms).mean()
                    loss = loss + ponder_lambda * ponder_loss
                    info["ponder_loss"] = ponder_loss
                    info["ponder_lambda"] = loss.new_tensor(ponder_lambda)
                else:
                    info["ponder_loss"] = loss.new_zeros(())
                    info["ponder_lambda"] = loss.new_tensor(float(ponder_lambda))
            if return_info:
                info["loss"] = loss.detach()
                return loss, info
            return loss
        else:
            if self.use_recurrence:
                info.setdefault("acttail_loss", logits.new_zeros(()))
                info.setdefault("acttail_active_fraction", logits.new_zeros(()))
                info.setdefault("acttail_threshold", logits.new_zeros(()))
                info.setdefault("ponder_loss", logits.new_zeros(()))
                info.setdefault("ponder_lambda", logits.new_zeros(()))
            if return_info:
                return logits, info
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42,
                 recurrence: Optional[int] = None, adaptive_exit_threshold: Optional[float] = None,
                 max_recurrence: Optional[int] = None):
        """
        Autoregressive inference. recurrence=None uses train_recurrence;
        pass recurrence=16 or 32 for deeper latent reasoning at inference.
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(
                ids, recurrence=recurrence,
                adaptive_exit_threshold=adaptive_exit_threshold,
                max_recurrence=max_recurrence,
            )
            logits = logits[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
