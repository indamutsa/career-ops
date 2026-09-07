"""Run this once before B0. It tells you whether the machine is ready, and nothing else.

    python3 code/check_env.py

Exit code 0 = ready. Anything printed with FAIL must be fixed before starting.
"""
import platform
import sys

ok = True


def check(label, condition, detail=""):
    global ok
    mark = "PASS" if condition else "FAIL"
    if not condition:
        ok = False
    print(f"  [{mark}] {label}{('  — ' + detail) if detail else ''}")
    return condition


print(f"\npython {platform.python_version()} on {platform.system()} {platform.machine()}\n")

check("Python >= 3.10", sys.version_info >= (3, 10), f"found {platform.python_version()}")

try:
    import torch
except ImportError:
    print("  [FAIL] torch not installed  — pip install torch")
    sys.exit(1)

check("torch installed", True, torch.__version__)

mps = torch.backends.mps.is_available()
cuda = torch.cuda.is_available()
device = "mps" if mps else "cuda" if cuda else "cpu"
check("GPU backend available", mps or cuda, f"using device='{device}'")
if not (mps or cuda):
    print("       CPU-only is fine for B0-B2. B6 will be slow but will still finish.")

# A real forward+backward on the target device catches driver problems that
# is_available() does not.
try:
    x = torch.randn(64, 64, device=device, requires_grad=True)
    y = (x @ x.T).sum()
    y.backward()
    check("forward + backward on device", x.grad is not None)
except Exception as e:  # noqa: BLE001
    check("forward + backward on device", False, str(e)[:80])

if device == "mps":
    try:
        with torch.autocast("mps", dtype=torch.bfloat16):
            z = torch.randn(32, 32, device="mps") @ torch.randn(32, 32, device="mps")
        check("bf16 autocast on mps", z.dtype in (torch.bfloat16, torch.float32), str(z.dtype))
    except Exception as e:  # noqa: BLE001
        check("bf16 autocast on mps", False, str(e)[:80])
    check("torch.mps.synchronize exists",
          hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"),
          "required before ANY timing measurement")

for name, hint in [("numpy", "pip install numpy"),
                   ("matplotlib", "pip install matplotlib  (plots in B6/B8)"),
                   ("regex", "pip install regex  (B2 pre-tokenization)"),
                   ("datasets", "pip install datasets  (B3 TinyStories)")]:
    try:
        __import__(name)
        check(name, True)
    except ImportError:
        check(name, False, hint)

print()
if ok:
    print("Ready. Open modules/build/00-foundations.md and start.\n")
else:
    print("Fix the FAIL lines above, then run this again.")
    print("To install everything at once:  pip install -r requirements.txt\n")
sys.exit(0 if ok else 1)
