"""
Unified Flash Attention interface with conservative FA3/SDPA switching.

Exports `flash_attn` module that matches the FA3 API exactly, but falls back
to PyTorch SDPA when no working FA3 kernel is available for the current GPU.
On the public autoresearch path, Blackwell laptop GPUs use SDPA directly, so
nanochat mirrors that behavior by default and only probes FA3 there when
explicitly overridden.

Usage (drop-in replacement for FA3):
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import os
import subprocess
import sys
import textwrap
import warnings

import torch
import torch.nn.functional as F

_FA3_SELECTION_NOTE = None


# =============================================================================
# Detection: Try to load FA3 on the best kernel path for the current GPU
# =============================================================================
def _select_fa3_repo():
    """Choose the most appropriate FA3 kernel repo for the current CUDA device."""
    global _FA3_SELECTION_NOTE
    _FA3_SELECTION_NOTE = None
    if not torch.cuda.is_available():
        return None
    override_repo = os.environ.get("NANOCHAT_FA3_REPO")
    if override_repo:
        _FA3_SELECTION_NOTE = f"explicit override via NANOCHAT_FA3_REPO={override_repo}"
        return override_repo
    major, minor = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name(0).lower()
    # Match autoresearch: Hopper uses varunneal, non-Hopper CUDA GPUs use
    # kernels-community's build, which supports some non-Hopper variants.
    #
    # However, the public autoresearch train.py never actually calls FA3 on this
    # RTX Pro 500 Blackwell laptop path; it uses SDPA directly. Default to that
    # conservative behavior here unless the repo is explicitly overridden.
    if "rtx pro 500 blackwell" in gpu_name:
        _FA3_SELECTION_NOTE = "disabled by default on RTX Pro 500 Blackwell to match autoresearch's public SDPA path"
        return None
    if (major, minor) == (9, 0):
        return "varunneal/flash-attention-3"
    return "kernels-community/flash-attn3"


def _load_flash_attention_3():
    """Try to load Flash Attention 3 and return (interface, repo_name)."""
    repo = _select_fa3_repo()
    if repo is None:
        return None, None
    try:
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel(repo).flash_attn_interface, repo
    except Exception:
        return None, repo


_fa3, FA3_KERNEL_REPO = _load_flash_attention_3()
_fa3_probe_ran = False
_fa3_probe_ok = False
FA3_UNAVAILABLE_REASON = _FA3_SELECTION_NOTE
FA3_MODULE_FILE = None
FA3_BUILD_VARIANT = None


def _probe_fa3_runtime():
    """Verify the selected FA3 backend can actually launch on this GPU."""
    global _fa3_probe_ran, _fa3_probe_ok, FA3_UNAVAILABLE_REASON, FA3_MODULE_FILE, FA3_BUILD_VARIANT
    if _fa3_probe_ran:
        return _fa3_probe_ok
    _fa3_probe_ran = True

    if _fa3 is None:
        FA3_UNAVAILABLE_REASON = _FA3_SELECTION_NOTE or "kernel repo could not be loaded"
        _fa3_probe_ok = False
        return False
    if not torch.cuda.is_available():
        FA3_UNAVAILABLE_REASON = "CUDA is not available"
        _fa3_probe_ok = False
        return False
    FA3_MODULE_FILE = getattr(_fa3, "__file__", None)
    try:
        import kernels.utils as _kernels_utils
        FA3_BUILD_VARIANT = _kernels_utils.build_variant()
    except Exception:
        FA3_BUILD_VARIANT = None

    # Run the probe in a subprocess because some FA3 launch failures terminate the
    # current process rather than raising a clean Python exception.
    probe_code = textwrap.dedent(
        """
        import json
        import torch
        import kernels.utils as ku
        from kernels import get_kernel

        repo = __import__("os").environ["NANOCHAT_FA3_REPO"]
        mod = get_kernel(repo)
        target = getattr(mod, "flash_attn_interface", mod)
        q = torch.randn(1, 128, 4, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 128, 4, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 128, 4, 128, device="cuda", dtype=torch.bfloat16)
        y = target.flash_attn_func(q, k, v, causal=True, window_size=(64, 0))
        torch.cuda.synchronize()
        print(json.dumps({
            "module_file": getattr(target, "__file__", None),
            "build_variant": ku.build_variant(),
            "shape": list(y.shape),
        }))
        """
    )
    try:
        env = os.environ.copy()
        env["NANOCHAT_FA3_REPO"] = FA3_KERNEL_REPO
        proc = subprocess.run(
            [sys.executable, "-c", probe_code],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            tail_msg = tail[-1] if tail else f"probe subprocess exited with code {proc.returncode}"
            raise RuntimeError(tail_msg)
        import json
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        FA3_MODULE_FILE = result.get("module_file")
        FA3_BUILD_VARIANT = result.get("build_variant")
        _fa3_probe_ok = True
        return True
    except Exception as exc:
        FA3_UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"
        _fa3_probe_ok = False
        warnings.warn(
            f"Flash Attention 3 backend '{FA3_KERNEL_REPO}' failed runtime probe; "
            f"falling back to SDPA. Reason: {FA3_UNAVAILABLE_REASON}"
        )
        return False


HAS_FA3 = _fa3 is not None and _probe_fa3_runtime()

# Override for testing: set to 'fa3', 'sdpa', or None (auto)
_override_impl = None


def _resolve_use_fa3():
    """Decide once whether to use FA3, based on availability, override, and dtype."""
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return True
    if _override_impl == 'sdpa':
        return False
    if HAS_FA3:
        # FA3 Hopper kernels only support bf16 and fp8; fp16/fp32 must use SDPA fallback
        from nanochat.common import COMPUTE_DTYPE
        if COMPUTE_DTYPE == torch.bfloat16:
            return True
        return False
    return False

USE_FA3 = _resolve_use_fa3()


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
