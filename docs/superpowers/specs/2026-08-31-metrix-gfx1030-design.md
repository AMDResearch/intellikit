# metrix gfx1030 (RDNA2) support — design

**Date:** 2026-08-31
**Branch:** `gfx1030-support`
**Hardware:** AMD Radeon RX 6800 XT (gfx1030), rocprofv3

## Problem

metrix nominally supports gfx1030 — `GFX1030Backend` exists, `get_backend()` maps
`gfx1030/1031/1032`, and `counter_defs.yaml` gates 26 metric definitions on `gfx1030`.
All of it was written without an RDNA2 card (PR #109, "RDNA backend support").

Validated against a real gfx1030, four things are broken.

### 1. Three metrics reference counters the hardware does not expose

`rocprofv3 --list-avail` on gfx1030 reports 148 counters. Three metrics gated on
`gfx1030` are not among them. Each was profiled individually and each crashes:

| Metric | Available on gfx1030 |
| --- | --- |
| `MeanOccupancyPerActiveCU` | No — only `MeanOccupancyPerCU` |
| `TA_BUFFER_LOAD_WAVEFRONTS_sum` | No — only `TA_FLAT_LOAD_WAVEFRONTS[_sum]` |
| `TA_BUFFER_STORE_WAVEFRONTS_sum` | No — only `TA_FLAT_STORE_WAVEFRONTS[_sum]` |

The remaining 23 gfx1030 metrics profile cleanly.

Failure mode is opaque: rocprofv3 exits without writing output and metrix surfaces
`RuntimeError: No output CSV found in /tmp/metrix_xxxx`
(`profiler/rocprof_wrapper.py:386`).

### 2. Three of five built-in profiles hard-fail on all RDNA parts

`METRIC_PROFILES` (`metrics/catalog.py:22`) is a hardcoded, architecture-agnostic
list containing CDNA-only metrics.

The CLI expands it verbatim, so `get_required_counters()` receives metrics the
backend has never heard of and raises `ValueError: Unknown metric ...`
(`backends/base.py:324`). The Python API path (`api.py`) already filtered
unavailable metrics and warned, so only `metrix profile --profile ...` crashes.
Routing both through one resolver removes the divergence.

| Profile | Missing on gfx1030 / gfx1100 / gfx1151 / gfx1201 |
| --- | --- |
| `memory` | `memory.l1_hit_rate`, `memory.coalescing_efficiency`, `memory.global_load_efficiency`, `memory.global_store_efficiency`, `memory.atomic_latency` |
| `memory_cache` | `memory.l1_hit_rate`, `memory.coalescing_efficiency` |
| `compute` | all five `compute.*` metrics |

`quick` and `memory_bandwidth` work and return plausible values on gfx1030
(HBM utilisation 29–55%, L2 hit rate 33–50% on a stream kernel).

This is not gfx1030-specific. RDNA has never had a working `memory` or `compute`
profile.

### 3. One unit test fails on RDNA hardware

`tests/unit/test_mcp_server.py:88` asserts every metric name contains `.`. RDNA
passthrough counters (`ALUStalledByLDS`, `VALUInsts`, `GPUBusy`, …) are
deliberately exposed under their native rocprofiler names and have no category
prefix. Baseline on this card: 330 passed, 1 failed, 33 skipped.

### 4. Documentation is stale

`AGENTS.md` and `metrix/README.md` state RDNA support is "gfx1151/gfx1201",
contradicting `get_backend()`.

## Design

### A. Architecture-aware profile resolution

Add a pure function to `metrix/metrics/catalog.py`:

```python
def resolve_profile_metrics(
    profile_name: str,
    available: Iterable[str],
    arch: str,
    unsupported: Iterable[str] = (),
) -> tuple[list[str], list[str]]:
    """Return (metrics to collect, metrics dropped as unknown on this arch)."""
```

- Takes the metric sets as arguments; imports no backend. Pure and directly
  unit-testable.
- `arch` is used only for messages. `unsupported` names the metrics the backend
  rejects with an explicit reason (`unsupported_reason` in the YAML); those are
  *kept* in the selection so the caller's existing handler can report the reason.
  `get_available_metrics()` omits them, so filtering on `available` alone would
  turn "TCC_EA_ATOMIC_LEVEL_sum is broken on MI200" into a bare "not available".
- Raises `ValueError` for an unknown profile name (preserving today's behaviour).
- Raises `ValueError` naming the architecture when *no* metric in the profile is
  available, replacing the internal `Unknown metric` traceback.
- Preserves the profile's declared metric order.

Both existing expansion sites call it, removing the current duplication:
`api.py:129` and `cli/profile_cmd.py:52`. Dropped metrics are reported once via
`logger.warning`, listed by name.

This changes one Python API behaviour: `Metrix.profile(profile=...)` on an
architecture supporting none of the profile's metrics now raises `ValueError`
instead of degrading to time-only mode. Silently timing a kernel when the user
asked for compute metrics hides the problem rather than reporting it.

Resulting behaviour on gfx1030: `memory` collects 8 of 13 metrics, `memory_cache`
2 of 4, `memory_bandwidth` and `quick` unchanged. `compute` still errors — all
five of its metrics are CDNA-only — but with a message naming the architecture.
The same fix applies to gfx1100, gfx1151 and gfx1201.

### B. counter_defs.yaml corrections

- Remove `gfx1030` from `MeanOccupancyPerActiveCU`. `MeanOccupancyPerCU` is
  already gated on `gfx1030` and covers the same ground.
- Remove `gfx1030` from `TA_BUFFER_LOAD_WAVEFRONTS_sum` and
  `TA_BUFFER_STORE_WAVEFRONTS_sum`.
- Add `TA_FLAT_LOAD_WAVEFRONTS_sum` and `TA_FLAT_STORE_WAVEFRONTS_sum` gated on
  `gfx1030`. HIP kernels emit flat/global memory instructions rather than buffer
  instructions, so these are the meaningful texture-addresser counters on this
  part. Both verified present on the card.

`gfx1100` remains on the `TA_BUFFER_*` definitions. No RDNA3 hardware is
available to check it, and unverified edits are what produced this bug. Recorded
here as suspect for whoever has an RX 7000 card.

### C. MCP test

Relax the assertion at `tests/unit/test_mcp_server.py:88` to accept bare
passthrough counter names alongside `category.metric` names. The unprefixed names
are the intended public surface for SDK passthrough counters on every RDNA
backend; renaming them would be a breaking change across gfx1100/1151/1201 and is
out of scope.

### D. Tests

**Unit — `resolve_profile_metrics`:**
- drops metrics absent from `available`
- preserves declared order of the survivors
- returns the dropped list for reporting
- raises `ValueError` on an unknown profile name
- raises `ValueError` naming the arch when nothing survives

**Unit — gfx1030 metric set** (extends `tests/unit/backends/test_backend_metrics.py`):
- contains `TA_FLAT_LOAD_WAVEFRONTS_sum` and `TA_FLAT_STORE_WAVEFRONTS_sum`
- excludes `TA_BUFFER_LOAD_WAVEFRONTS_sum`, `TA_BUFFER_STORE_WAVEFRONTS_sum`,
  `MeanOccupancyPerActiveCU`

**Integration — every built-in profile, on whatever GPU is present** (new test in
`tests/integration/`): profile a HIP kernel once per entry in `METRIC_PROFILES`
and assert each either produces metric values or raises the clean
"no metrics available on this architecture" error. Deliberately *not* gated on
`requires_arch("gfx1030")` — the absence of an architecture-generic profile test
is why this shipped broken, and gating it on one card would repeat the mistake.

### E. Documentation

Update the RDNA support statement in `AGENTS.md`, `metrix/README.md`, and
`docs/tools/metrix.md` to match `get_backend()`: gfx1030/1031/1032 (RDNA2),
gfx1100–1103 (RDNA3), gfx1151 (RDNA 3.5), gfx1201 (RDNA4). Note that gfx1030 is
hardware-validated and that per-architecture metric coverage varies.

## Out of scope

Flagged, not fixed:

- **`--metrics` with an unsupported metric.** Now handled: the CLI checks
  explicitly requested metrics against the backend before profiling, matching
  what `api.py` already did. Was previously a `traceback.print_exc()`.
- **Opaque rocprofv3 failure.** When rocprofv3 rejects a counter it writes no
  output and metrix reports `No output CSV found`. A pre-flight check against
  `--list-avail`, or capturing rocprofv3's stderr, would turn this into an
  actionable message. Affects all architectures.
- **Integration coverage on non-CDNA hardware.** 29 of 44 integration tests skip
  on gfx1030, gated on `requires_cdna()` or on metrics RDNA lacks. Item D adds
  one architecture-generic test; broadening the rest is a larger effort.
- **`gfx1100` TA_BUFFER_* definitions.** Suspect, unverifiable without RDNA3
  hardware. See section B.

## Success criteria

1. `metrix profile --profile <p>` succeeds on gfx1030 for `quick`,
   `memory_bandwidth`, `memory`, and `memory_cache`; `compute` fails with a
   message naming the architecture rather than a traceback.
2. Each of the 23 working gfx1030 metrics still profiles, and
   `TA_FLAT_LOAD_WAVEFRONTS_sum` / `TA_FLAT_STORE_WAVEFRONTS_sum` produce values.
3. No gfx1030 metric references a counter absent from `rocprofv3 --list-avail`.
4. `pytest tests/unit` passes with zero failures on gfx1030 (baseline: 1 failure).
5. `pytest tests/integration` shows no new failures; the two existing failures are
   `PATH` artifacts of running outside the venv, not gfx1030 defects.
6. Docs state the supported architecture list that `get_backend()` implements.
