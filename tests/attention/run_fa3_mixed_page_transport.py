"""Plain runner for the FA3 mixed-page conformance matrix (no test framework).

Usage: python tests/attention/run_fa3_mixed_page_transport.py [--first]
  --first  run a single case (fp4, NHD, q_len 4) so a build error surfaces once.
Exit code is the number of failing cases.
"""

import importlib.util
import pathlib
import sys
import traceback

_spec = importlib.util.spec_from_file_location(
    "_fa3_mixed_test", pathlib.Path(__file__).with_name("test_fa3_mixed_page_transport.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

Q_LENS = [1, 4, 64, 130]
LAYOUTS = ["NHD", "HND"]
PAGE_MODES = ["a16", "fp8", "fp4", "a16_fp8_runs", "a16_fp8", "a16_fp4", "fp8_fp4", "mixed"]


def main() -> int:
    cases = [(4, "NHD", "fp4")] if "--first" in sys.argv else [
        (q, l, m) for m in PAGE_MODES for l in LAYOUTS for q in Q_LENS
    ]
    failures = 0
    for q_len, layout, mode in cases:
        name = f"[{mode}-{layout}-{q_len}]"
        try:
            _mod.test_fa3_mixed_page_transport_matches_a16_expansion(q_len, layout, mode)
            print(f"PASS {name}", flush=True)
        except BaseException as exc:  # noqa: BLE001 - report every case
            if type(exc).__name__.endswith("Skipped"):
                print(f"SKIP {name} {exc}", flush=True)
                continue
            failures += 1
            print(f"FAIL {name}", flush=True)
            traceback.print_exc()
            if "--first" in sys.argv:
                break
    print(f"{len(cases) - failures} passed, {failures} failed")
    return failures


if __name__ == "__main__":
    sys.exit(main())
