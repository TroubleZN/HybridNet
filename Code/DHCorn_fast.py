import torch

from Code.DHCorn import dhcorn as _dhcorn


_COMPILED_CACHE = {}
_FALLBACK_WARNED = set()


def _enable_cuda_runtime_hints():
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


def _get_compiled_dhcorn(device: str):
    cache_key = str(device)
    if cache_key in _COMPILED_CACHE:
        return _COMPILED_CACHE[cache_key]

    _enable_cuda_runtime_hints()

    def _wrapped(envi, mg, geno, pheno=0, device=device):
        return _dhcorn(envi, mg, geno, pheno=pheno, device=device)

    compiled = _wrapped
    if hasattr(torch, "compile") and str(device).startswith("cuda"):
        try:
            compiled = torch.compile(_wrapped, mode="reduce-overhead", fullgraph=False)
        except Exception:
            compiled = _wrapped

    _COMPILED_CACHE[cache_key] = compiled
    return compiled


def dhcorn_fast(envi, mg, geno, pheno=0, device="cpu", use_compile=True):
    if use_compile:
        fn = _get_compiled_dhcorn(device)
        try:
            return fn(envi, mg, geno, pheno=pheno, device=device)
        except Exception as exc:
            key = str(device)
            _COMPILED_CACHE[key] = _dhcorn
            if key not in _FALLBACK_WARNED:
                print(f"[DHCorn_fast] compile fallback on {device}: {exc.__class__.__name__}")
                _FALLBACK_WARNED.add(key)
            return _dhcorn(envi, mg, geno, pheno=pheno, device=device)
    return _dhcorn(envi, mg, geno, pheno=pheno, device=device)
