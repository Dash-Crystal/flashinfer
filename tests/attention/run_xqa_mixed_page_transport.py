"""Plain runner for the XQA mixed-page conformance matrix (no test framework).

Runs the 32 register-expansion cases, the 2 native block-FP8 cases and the 18
tail / value-range cases (short and odd sequence tails, 35-tile CTAs, E4M3
subnormal payloads and maximal block scales) on sm90 / sm12x.  Exit code is the
number of failing cases.
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
    failures = 0
    n = 0
    for mode in PAGE_MODES:
        for q_len, layout in SPANS:
            n += 1
            failures += _run(f"[{mode}-{layout}-{q_len}]",
                             _mod.test_xqa_mixed_page_transport_matches_register_expansion,
                             q_len, layout, mode)
    for q_len in (1, 64):
        n += 1
        failures += _run(f"[native_fp8-{q_len}]",
                         _mod.test_xqa_native_block_fp8_matches_a16_expansion, q_len)
    # Tails, 35-tile CTAs and extreme E4M3 value ranges (q=1, the sm90 GMMA host).
    for seq_len, nb_sub_seq, regime in _mod.TAIL_CASES:
        for mode in ("fp8", "fp4", "mixed"):
            n += 1
            failures += _run(f"[tail-{mode}-{seq_len}-sub{nb_sub_seq}-{regime}]",
                             _mod.test_xqa_mixed_page_transport_tails_and_value_ranges,
                             mode, seq_len, nb_sub_seq, regime)
    print(f"{n - failures} passed, {failures} failed")
    return failures


if __name__ == "__main__":
    sys.exit(main())
