"""Plain runner for the XQA mixed-page conformance matrix (no test framework).

Runs the 32 register-expansion cases and the 2 native block-FP8 cases on
sm90 / sm12x.  Exit code is the number of failing cases.
"""

import importlib.util
import pathlib
import sys
import traceback

_spec = importlib.util.spec_from_file_location(
    "_xqa_mixed_test", pathlib.Path(__file__).with_name("test_xqa_mixed_page_transport.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

SPANS = [(1, "NHD"), (4, "NHD"), (64, "NHD"), (4, "HND")]
PAGE_MODES = ["a16", "fp8", "fp4", "a16_fp8_runs", "a16_fp8", "a16_fp4", "fp8_fp4", "mixed"]


def _run(name, fn, *args):
    try:
        fn(*args)
        print(f"PASS {name}", flush=True)
        return 0
    except BaseException as exc:  # noqa: BLE001
        if type(exc).__name__.endswith("Skipped"):
            print(f"SKIP {name} {exc}", flush=True)
            return 0
        print(f"FAIL {name}", flush=True)
        traceback.print_exc()
        return 1


def main() -> int:
    # Optional `--shard i/n`: run only the cases whose index % n == i, so the
    # per-module JIT compiles can be spread over several processes.
    shard_idx, shard_n = 0, 1
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "--shard":
        shard_idx, shard_n = (int(x) for x in args[1].split("/"))
    cases = []
    for mode in PAGE_MODES:
        for q_len, layout in SPANS:
            cases.append((f"[{mode}-{layout}-{q_len}]",
                          _mod.test_xqa_mixed_page_transport_matches_register_expansion,
                          (q_len, layout, mode)))
    for q_len in (1, 64):
        cases.append((f"[native_fp8-{q_len}]",
                      _mod.test_xqa_native_block_fp8_matches_a16_expansion, (q_len,)))
    failures = 0
    n = 0
    for i, (name, fn, fn_args) in enumerate(cases):
        if i % shard_n != shard_idx:
            continue
        n += 1
        failures += _run(name, fn, *fn_args)
    print(f"{n - failures} passed, {failures} failed")
    return failures


if __name__ == "__main__":
    sys.exit(main())
