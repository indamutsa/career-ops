# `code/` — your lab code

One file per lesson, named in the lesson's header block: `b0_autograd.py` … `b10_reasoning.py`.
The full table is in [`START-HERE.md`](../START-HERE.md).

`check_env.py` is the only file here not written during a lesson — run it before B0.

## Conventions

- **Seed at the top of every file.** `torch.manual_seed(1337)`. You will be comparing runs
  constantly, and an unseeded comparison is not a comparison.
- **Device once, at the top.** Set `device` from `torch.backends.mps.is_available()` and use that
  variable everywhere; never hardcode `"mps"` inside a function.
- **`torch.mps.synchronize()` before reading any clock.** MPS dispatch is asynchronous; without it
  every timing you record is fiction.
- **No file imports another.** Each lesson stands alone, so a broken B4 never blocks B5. Where a
  lesson needs something you wrote earlier it says so and you retype it — by B5 you will have typed
  attention twice, which is the point.
- **Numbers go in `notes/`, not in comments.** The next run overwrites comments; notes persist.
