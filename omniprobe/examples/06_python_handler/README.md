# Example 06: A message handler written in Python

`line_counter.py` is a Python module for Omniprobe's `PythonHandler` (Omniprobe b9492cd or later
for the per-wave call it uses). It prints, per dispatch and per source location, access type and
address space: wave executions, lane accesses, executions by partially active waves, and distinct
128-byte lines.

Run it with `omniprobe-ik run ./line_counter.py ./app` (Python: `run_analyzer("line_counter.py",
"./app")`; MCP `run_analyzer`) on an application built with the **memory** plugin and `-g`. What
`handleReport` prints is in the run's `stdout`.

What Omniprobe calls (all functions optional, at least one of the first three required):

| Function | Called | `msg` keys |
|---|---|---|
| `handleAddressWave(msg)` | once per wave-level memory instruction | `addresses` (one int per active lane), `lanes`, `exec`, `access_type` (`"read"`, `"write"`, `"readwrite"`), `memory_space` (`"global"`, `"shared"`, `"constant"`, `"scratch"`, `"flat"`, ...), `access_size` (bytes), `source_file`, `line`, `column`, `block_idx`, `wave_num`, `cu_id`, `se_id`, `xcc_id`, `timestamp` (`s_memtime` ticks), `kernel`, `dispatch_id` |
| `handleAddressMessage(msg)` | once per active lane | `address`, `kernel`, `dispatch_id` |
| `handleTimeMessage(msg)` | per time-interval message | the wave keys, `start`, `stop`, `elapsed_time`, `block_id` |
| `handleReport(msg)` | once per dispatch | `kernel`, `dispatch_id` |

- About 10 µs per call, so one `handleAddressWave` call per wave is up to 64 times cheaper than
  per-lane calls. Narrow with `--kernel` and `--dispatches 1` on large applications.
- Module state persists across dispatches (one interpreter per process); calls never overlap.
- Print with `flush=True`: unflushed output is lost when the run ends.
- A module that cannot be imported, or defines none of the three message functions, is an error
  (Omniprobe exits 1 before the application runs).
- Every wave also loads its kernel arguments (`memory_space == "constant"`); filter on
  `memory_space` as needed.
- Untested: a Python handler inside an application that is itself a Python process (e.g.
  Triton); it may deadlock on the GIL.

`analyze_python_handler.py` checks the handler on a kernel whose answer is known (copy_n, 1000
threads in blocks of 256 = 16 waves, the last with 40 active lanes: for line 5, 16 wave
executions, 1000 lane accesses, 1 partial wave and 32 distinct lines for both the read and the
write). Test any handler that way before trusting it on a real application.
