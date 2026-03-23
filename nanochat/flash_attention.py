"""
Unified attention backend interface.

On this SM120 Blackwell laptop, BF16 PyTorch SDPA is the primary path by
design. Hopper keeps an FA3 path when a working kernel is available. All other
call sites use the same `flash_attn` shim regardless of backend.
"""
import os
import subprocess
import sys
import textwrap
import warnings
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from nanochat.common import COMPUTE_DTYPE


REGIME_FULL = "full_context_causal_training"
REGIME_SLIDING = "masked_sliding_window_training"
REGIME_DECODE = "single_token_decode"
REGIME_DECODE_GQA = "single_token_decode_gqa"
ALL_SDPA_REGIMES = (REGIME_FULL, REGIME_SLIDING, REGIME_DECODE, REGIME_DECODE_GQA)

SDPA_BACKEND_ENUMS = {
    "sdpa_cudnn": SDPBackend.CUDNN_ATTENTION,
    "sdpa_flash": SDPBackend.FLASH_ATTENTION,
    "sdpa_efficient": SDPBackend.EFFICIENT_ATTENTION,
    "sdpa_math": SDPBackend.MATH,
}
SDPA_AUTO_PRIORITY = {
    REGIME_FULL: ("sdpa_cudnn", "sdpa_flash", "sdpa_efficient", "sdpa_math"),
    REGIME_SLIDING: ("sdpa_cudnn", "sdpa_efficient", "sdpa_math"),
    REGIME_DECODE: ("sdpa_flash", "sdpa_cudnn", "sdpa_efficient", "sdpa_math"),
    REGIME_DECODE_GQA: ("sdpa_cudnn", "sdpa_flash", "sdpa_math"),
}

_FA3_SELECTION_NOTE = None
_SDPA_PROBE_CACHE = {}
_SDPA_AUTO_SELECTIONS = {}


def _device_capability():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()


def _is_sm120():
    return _device_capability() == (12, 0)


def _normalize_backend_mode(raw_mode):
    mode = (raw_mode or "auto").strip().lower()
    aliases = {
        "sdpa": "auto_sdpa",
        "pytorch_sdpa": "auto_sdpa",
        "bf16_sdpa": "auto_sdpa",
    }
    mode = aliases.get(mode, mode)
    allowed = {"auto", "auto_sdpa", "fa3", *SDPA_BACKEND_ENUMS.keys()}
    if mode not in allowed:
        raise ValueError(
            f"Invalid attention backend '{raw_mode}'. Expected one of: "
            f"{', '.join(sorted(allowed))}"
        )
    return mode


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
    capability = _device_capability()
    if capability == (12, 0):
        _FA3_SELECTION_NOTE = "SM120 uses BF16 SDPA as the primary attention path"
        return None
    if capability == (9, 0):
        return "varunneal/flash-attention-3"
    return "kernels-community/flash-attn3"


def _load_flash_attention_3():
    """Try to load Flash Attention 3 and return (interface, repo_name)."""
    repo = _select_fa3_repo()
    if repo is None:
        return None, None
    try:
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
            f"BF16 SDPA policy remains active. Reason: {FA3_UNAVAILABLE_REASON}"
        )
        return False


HAS_FA3 = _fa3 is not None and _probe_fa3_runtime()

# Backward-compat override used by tests. Accepted values are:
# None, "auto", "sdpa", "auto_sdpa", "fa3", and explicit "sdpa_*" backend names.
_override_impl = None


def _resolve_backend_mode():
    if _override_impl is not None:
        return _normalize_backend_mode(_override_impl)
    return _normalize_backend_mode(os.environ.get("NANOCHAT_ATTN_BACKEND", "auto"))


ATTENTION_BACKEND_MODE = _resolve_backend_mode()


def _use_fa3_mode():
    if ATTENTION_BACKEND_MODE == "fa3":
        if not HAS_FA3:
            raise RuntimeError(
                f"NANOCHAT_ATTN_BACKEND=fa3 requested, but FA3 is unavailable: {FA3_UNAVAILABLE_REASON}"
            )
        if COMPUTE_DTYPE != torch.bfloat16:
            raise RuntimeError(
                f"NANOCHAT_ATTN_BACKEND=fa3 requires bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}"
            )
        return True
    if ATTENTION_BACKEND_MODE in {"auto_sdpa", *SDPA_BACKEND_ENUMS.keys()}:
        return False
    if ATTENTION_BACKEND_MODE == "auto":
        if _is_sm120():
            return False
        return HAS_FA3 and COMPUTE_DTYPE == torch.bfloat16
    return False


USE_FA3 = _use_fa3_mode()
ATTENTION_BACKEND_POLICY = (
    "fa3_primary"
    if USE_FA3
    else "bf16_sdpa_primary_sm120"
    if _is_sm120()
    else "sdpa_primary"
)


def _classify_sdpa_regime(q, k, window_size, enable_gqa):
    tq = q.size(2)
    tk = k.size(2)
    window = window_size[0]
    if tq == 1:
        return REGIME_DECODE_GQA if enable_gqa else REGIME_DECODE
    if (window < 0 or window >= tq) and tq == tk:
        return REGIME_FULL
    return REGIME_SLIDING


def _make_sdpa_probe_args(regime):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if regime == REGIME_FULL:
        q = torch.randn(1, 4, 128, 128, device=device, dtype=dtype)
        k = torch.randn(1, 4, 128, 128, device=device, dtype=dtype)
        v = torch.randn(1, 4, 128, 128, device=device, dtype=dtype)
        return dict(query=q, key=k, value=v, is_causal=True, enable_gqa=False)
    if regime == REGIME_SLIDING:
        tq = 128
        tk = 128
        q = torch.randn(1, 4, tq, 128, device=device, dtype=dtype)
        k = torch.randn(1, 4, tk, 128, device=device, dtype=dtype)
        v = torch.randn(1, 4, tk, 128, device=device, dtype=dtype)
        row_idx = torch.arange(tq, device=device).unsqueeze(1)
        col_idx = torch.arange(tk, device=device).unsqueeze(0)
        mask = (col_idx <= row_idx) & ((row_idx - col_idx) <= 256)
        mask = mask.unsqueeze(0).unsqueeze(0)
        return dict(query=q, key=k, value=v, attn_mask=mask, enable_gqa=False)
    if regime == REGIME_DECODE:
        q = torch.randn(1, 4, 1, 128, device=device, dtype=dtype)
        k = torch.randn(1, 4, 128, 128, device=device, dtype=dtype)
        v = torch.randn(1, 4, 128, 128, device=device, dtype=dtype)
        return dict(query=q, key=k, value=v, is_causal=False, enable_gqa=False)
    if regime == REGIME_DECODE_GQA:
        q = torch.randn(1, 8, 1, 128, device=device, dtype=dtype)
        k = torch.randn(1, 4, 128, 128, device=device, dtype=dtype)
        v = torch.randn(1, 4, 128, 128, device=device, dtype=dtype)
        return dict(query=q, key=k, value=v, is_causal=False, enable_gqa=True)
    raise ValueError(f"Unknown SDPA regime: {regime}")


def _probe_sdpa_backend(backend_name, regime):
    key = (backend_name, regime)
    if key in _SDPA_PROBE_CACHE:
        return _SDPA_PROBE_CACHE[key]
    if backend_name not in SDPA_BACKEND_ENUMS:
        raise ValueError(f"Unknown SDPA backend: {backend_name}")
    if not torch.cuda.is_available() and backend_name != "sdpa_math":
        _SDPA_PROBE_CACHE[key] = False
        return False
    try:
        kwargs = _make_sdpa_probe_args(regime)
        with sdpa_kernel(SDPA_BACKEND_ENUMS[backend_name]):
            F.scaled_dot_product_attention(**kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ok = True
    except Exception:
        ok = False
    _SDPA_PROBE_CACHE[key] = ok
    return ok


def _resolve_auto_sdpa_selections():
    selections = {}
    for regime, candidates in SDPA_AUTO_PRIORITY.items():
        selected = None
        for backend_name in candidates:
            if _probe_sdpa_backend(backend_name, regime):
                selected = backend_name
                break
        if selected is None:
            raise RuntimeError(
                f"No supported SDPA backend found for attention regime '{regime}'"
            )
        selections[regime] = selected
    return selections


if not USE_FA3:
    _SDPA_AUTO_SELECTIONS = _resolve_auto_sdpa_selections()


def _resolve_use_fa3():
    """Backward-compatible boolean for existing tests/imports."""
    return _use_fa3_mode()


def _selected_sdpa_backend(regime):
    if ATTENTION_BACKEND_MODE in SDPA_BACKEND_ENUMS:
        backend_name = ATTENTION_BACKEND_MODE
        if not _probe_sdpa_backend(backend_name, regime):
            raise RuntimeError(
                f"Attention backend '{backend_name}' was forced, but it is unsupported "
                f"for regime '{regime}' on this machine."
            )
        return backend_name
    return _SDPA_AUTO_SELECTIONS[regime]


ATTENTION_REGIME_BACKENDS = (
    {regime: "fa3" for regime in ALL_SDPA_REGIMES}
    if USE_FA3
    else {regime: _selected_sdpa_backend(regime) for regime in ALL_SDPA_REGIMES}
)


def describe_attention_backends():
    lines = [
        f"Attention backend mode: {ATTENTION_BACKEND_MODE}",
        f"Attention backend policy: {ATTENTION_BACKEND_POLICY}",
    ]
    if USE_FA3:
        lines.append(f"Selected backend: fa3 via {FA3_KERNEL_REPO}")
        if FA3_BUILD_VARIANT is not None:
            lines.append(f"FA3 build variant: {FA3_BUILD_VARIANT}")
        if FA3_MODULE_FILE is not None:
            lines.append(f"FA3 module: {FA3_MODULE_FILE}")
    else:
        lines.append("Selected SDPA backends by regime:")
        for regime in ALL_SDPA_REGIMES:
            lines.append(f"  {regime}: {ATTENTION_REGIME_BACKENDS[regime]}")
        if FA3_KERNEL_REPO is not None and FA3_UNAVAILABLE_REASON is not None:
            lines.append(
                f"FA3 status: repo '{FA3_KERNEL_REPO}' is not active on this policy "
                f"({FA3_UNAVAILABLE_REASON})"
            )
    return lines


def _sdpa_context(backend_name):
    if backend_name is None:
        return nullcontext()
    return sdpa_kernel(SDPA_BACKEND_ENUMS[backend_name])


def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    tq = q.size(2)
    tk = k.size(2)
    window = window_size[0]
    regime = _classify_sdpa_regime(q, k, window_size, enable_gqa)
    backend_name = _selected_sdpa_backend(regime)

    # Full context, same length
    if (window < 0 or window >= tq) and tq == tk:
        with _sdpa_context(backend_name):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if tq == 1:
        if window >= 0 and window < tk:
            start = max(0, tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        with _sdpa_context(backend_name):
            return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Chunked/sliding attention needs an explicit causal mask.
    device = q.device
    row_idx = (tk - tq) + torch.arange(tq, device=device).unsqueeze(1)
    col_idx = torch.arange(tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx
    if window >= 0 and window < tk:
        mask = mask & ((row_idx - col_idx) <= window)
    with _sdpa_context(backend_name):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


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

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA path mirrors that behavior.
    """
    if USE_FA3:
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    _, t_new, _, _ = q.shape
    pos = cache_seqlens[0].item()

    if k is not None and v is not None:
        k_cache[:, pos:pos + t_new, :, :] = k
        v_cache[:, pos:pos + t_new, :, :] = v

    end_pos = pos + t_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)
    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)
    return y_sdpa.transpose(1, 2)


from types import SimpleNamespace

flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
