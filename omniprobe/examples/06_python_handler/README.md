# Example 06: A message handler written in Python

`address_counter.py` is a Python module for Omniprobe's `PythonHandler` (Omniprobe 13ca707 or
later). Omniprobe calls `handleAddressMessage(msg)` once per active lane of every instrumented
memory instruction and `handleReport(msg)` once per dispatch; this one prints, per dispatch, the
lane accesses, distinct addresses and distinct 128-byte lines.

Run it with `omniprobe-ik run ./address_counter.py ./app` (Python: `run_analyzer("address_counter.py",
"./app")`; MCP `run_analyzer`) on an application built with the **memory** plugin. What
`handleReport` prints is in the run's `stdout`.

What a Python handler can and cannot see:

- Each call gets only `address`, `kernel` and `dispatch_id` (and `handleTimeMessage` gets
  `elapsed_time`). There is **no source line, read/write flag, address space, access size,
  exec mask or wave id**, so it suits address-level questions: which buffers or pages are
  touched, how many distinct addresses or lines, address histograms per dispatch.
- Kernel-argument loads are included (every lane of every wave), indistinguishable from the
  application's accesses except by their address.
- About 10 µs per call: narrow with `--kernel` and `--dispatches 1`.
- For per-line or per-wave questions, or high message rates, write a C++ handler (example 04).
- Untested: a Python handler inside an application that is itself a Python process (e.g.
  Triton); it may deadlock on the GIL.

`analyze_python_handler.py` checks the handler on a kernel whose answer is known (16 waves,
2000 global lane accesses plus 16 × 64 kernel-argument lane loads = 3024). Test any handler that
way before trusting it on a real application.
