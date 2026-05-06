#!/usr/bin/env python3
"""
Debug script — run this on the compute node to diagnose env issues.
Usage: python scripts/debug_env.py
"""

import os
import sys
import subprocess
from pathlib import Path

PASS = "✓"
FAIL = "✗"
WARN = "!"

def check(label, ok, detail=""):
    sym = PASS if ok else FAIL
    print(f"  [{sym}] {label}" + (f": {detail}" if detail else ""))
    return ok

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

section("Python")
check("executable", True, sys.executable)
check("version", sys.version_info >= (3, 10), sys.version)
check("prefix", True, sys.prefix)

section("Environment Variables")
check("WORLDSCORE_ROOT set", bool(os.environ.get("WORLDSCORE_ROOT")),
      os.environ.get("WORLDSCORE_ROOT", "(not set)"))
check("CUDA_VISIBLE_DEVICES", True,
      os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"))
ldpath = os.environ.get("LD_LIBRARY_PATH", "")
conda_env_lib = Path(sys.prefix) / "lib"
check("LD_LIBRARY_PATH contains conda lib",
      str(conda_env_lib) in ldpath,
      str(conda_env_lib) + (" ← MISSING from LD_LIBRARY_PATH" if str(conda_env_lib) not in ldpath else ""))

section("Shared Libraries")
def check_lib(name):
    result = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True)
    found_ldconfig = name in result.stdout
    # Also check conda env lib directly
    conda_lib = Path(sys.prefix) / "lib"
    found_conda = any(conda_lib.glob(f"{name}*"))
    check(f"{name} in conda env", found_conda,
          str(list(conda_lib.glob(f"{name}*"))[:1] or "(not found)"))
    check(f"{name} findable by linker", found_ldconfig or found_conda)

check_lib("libbz2")
check_lib("libexpat")
check_lib("libpython3.10")

section("libexpat symbol check")
try:
    import ctypes
    libexpat_path = next(
        (str(p) for p in (Path(sys.prefix) / "lib").glob("libexpat.so*")
         if not p.is_symlink() or p.resolve().exists()),
        None
    )
    if libexpat_path:
        lib = ctypes.CDLL(libexpat_path)
        has_sym = hasattr(lib, "XML_SetAllocTrackerActivationThreshold")
        check("XML_SetAllocTrackerActivationThreshold in conda libexpat",
              has_sym, libexpat_path)
    else:
        check("libexpat found in conda env", False)
except Exception as e:
    check("libexpat check", False, str(e))

section("pyexpat (stdlib)")
try:
    import pyexpat
    check("pyexpat import", True, pyexpat.__file__)
except Exception as e:
    check("pyexpat import", False, str(e))

section("WorldScore Package")
worldscore_root = os.environ.get("WORLDSCORE_ROOT", "")
ws_root = Path(worldscore_root) if worldscore_root else None

if ws_root:
    check("WORLDSCORE_ROOT exists", ws_root.exists(), str(ws_root))
    check("worldscore package in root", (ws_root / "worldscore").exists())
    check("benchmark/metrics exists", (ws_root / "worldscore/benchmark/metrics").exists())
    check("third_party exists", (ws_root / "worldscore/benchmark/metrics/third_party").exists())

    ckpt_dir = ws_root / "worldscore/benchmark/metrics/checkpoints"
    check("checkpoints dir exists", ckpt_dir.exists(), str(ckpt_dir))
    for ckpt in ["droid.pth", "VFIMamba.pkl", "Tartan-C-T-TSKH-spring540x960-M.pth"]:
        check(f"  checkpoint: {ckpt}", (ckpt_dir / ckpt).exists())
else:
    check("WORLDSCORE_ROOT set", False, "cannot check checkpoints")

try:
    import worldscore
    ws_file = worldscore.__file__
    from_conda = "/site-packages/" in ws_file
    from_local = worldscore_root and worldscore_root in ws_file
    check("worldscore imported from local repo (not pip)", not from_conda or from_local, ws_file)
except Exception as e:
    check("worldscore import", False, str(e))

try:
    from worldscore.benchmark import metrics
    check("worldscore.benchmark.metrics importable", True)
except Exception as e:
    check("worldscore.benchmark.metrics importable", False, str(e))

section("Core Packages")
for pkg in ["torch", "torchvision", "numpy", "PIL", "cv2", "pyiqa", "tqdm"]:
    try:
        mod = __import__(pkg if pkg != "PIL" else "PIL.Image")
        ver = getattr(mod, "__version__", "?")
        check(pkg, True, ver)
    except Exception as e:
        check(pkg, False, str(e))

section("CUDA")
try:
    import torch
    check("torch.cuda.is_available()", torch.cuda.is_available())
    if torch.cuda.is_available():
        check("GPU name", True, torch.cuda.get_device_name(0))
        check("GPU count", True, str(torch.cuda.device_count()))
except Exception as e:
    check("torch", False, str(e))

section("Third-party sys.path")
if ws_root:
    third_party = ws_root / "worldscore/benchmark/metrics/third_party"
    for sub in ["droid_slam", "groundingdino", "sam2", "VFIMamba", "SEA-RAFT"]:
        p = str(third_party / sub)
        check(f"{sub} in sys.path", p in sys.path)

print(f"\n{'='*60}")
print("  Done")
print(f"{'='*60}\n")
