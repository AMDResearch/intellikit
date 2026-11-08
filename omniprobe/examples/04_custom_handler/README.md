# Example 04: A bespoke message handler

When no stock analyzer counts what you need, write a message handler: a C++ class
Omniprobe calls for every message an instrumented kernel sends. `access_counter.cc`
(about 90 lines) counts memory-instruction executions per kernel by address space
and kind, the lanes involved, and executions by partially active waves.

`analyze_custom.py`:

1. Builds the handler with the flags from `handler_build_flags()` /
   `omniprobe-ik handler-flags`: `hipcc` against the `dh_comms` headers Omniprobe
   installs (`<prefix>/omniprobe/include/dh_comms`; the source tree is the fallback),
   linked against `libdh_comms`.
2. Builds a kernel whose counts are known in advance (1000 threads, blocks of 256:
   16 waves, the last with 40 active lanes) with `instrumentation_flags("memory")`.
3. Runs it with `run_analyzer("/abs/path/libaccess_counter.so", "./app")` and checks
   the counts: 16 load and 16 store executions, 2000 lanes, 2 partial-wave executions.

Test a handler the same way: run it on a kernel whose answer you can compute by hand
before trusting it on a real application.

```bash
python3 analyze_custom.py
```
