<div align="center">

<img src="./docs/public/intellikit.svg" alt="IntelliKit" width="200" />

# IntelliKit

**Agent-first tooling for AMD hardware**

</div>

IntelliKit is a set of Python tools for **AMD-focused** performance and validation. Most of the stack targets **GPUs through ROCm**, turning hardware counters, traces, and dispatch data into **clear APIs** you can use from Python. **`uprof_mcp`** adds **AMD uProf** for **host-side CPU** hotspot analysis in the same toolbox. For LLM-style workflows you also get **Model Context Protocol (MCP)** servers (profiling, HIP compile, HIP docs, **rocminfo**, …) and **agent skills** — installable `SKILL.md` playbooks for Kerncap, Metrix, Linex, Nexus, and Accordo (`install/skills/install.sh`). Use the stack from a notebook, a script, an MCP client, or Cursor / Claude / Codex.

---

## What’s in the box

Rough workflow: **isolate** a kernel → **profile** it (counters and/or source lines) → lean on **other** helpers (what actually ran, **MCP**, **skills**, CPU profiling) → **validate** that changes are still correct.

| Tool | What it’s for | Docs |
|------|----------------|------|
| **[Kerncap](kerncap/)** | **Isolate** — capture dispatches and build **standalone reproducers** (HIP, Triton). | [README](kerncap/README.md) · [examples](kerncap/examples/) |
| **[Metrix](metrix/)** | **Profile** — **human-readable** metrics from hardware counters (bandwidth, cache, etc.). | [README](metrix/README.md) · [examples](metrix/examples/) |
| **[Linex](linex/)** | **Profile** — **source-line** timing and stalls (compile with `-g` for file:line mapping). | [README](linex/README.md) · [examples](linex/examples/) |
| **[Nexus](nexus/)** | **Inspect** — from **HSA packets**, see what ran: source and assembly. | [README](nexus/README.md) · [examples](nexus/examples/) |
| **[rocm_mcp](rocm_mcp/)** | **MCP** — HIP compile, HIP docs, **rocminfo**, and related servers for agents. | [README](rocm_mcp/README.md) · [examples](rocm_mcp/examples/) |
| **[uprof_mcp](uprof_mcp/)** | **CPU** — MCP bridge to **AMD uProf** for host-side hotspots. | [README](uprof_mcp/README.md) · [examples](uprof_mcp/examples/) |
| **[Accordo](accordo/)** | **Validate** — prove an optimized kernel still matches a reference. | [README](accordo/README.md) · [examples](accordo/examples/) |

**Idea in one line:** pull a kernel out with Kerncap, understand it with Metrix and Linex, dig into execution with Nexus, wire agents with **MCP** and **skills**, add **uProf** when you care about the host, then lock in correctness with Accordo.

---

## Quick start

**Tools** — every package from Git via `pip` (`install/tools/install.sh`; no metapackage at the repo root):

```bash
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/tools/install.sh | bash
```

**Skills** — agent skill files for Kerncap, Metrix, Linex, Nexus, and Accordo (`install/skills/install.sh`):

```bash
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/skills/install.sh | bash
```

**Clone?** Use `./install/tools/install.sh` and `./install/skills/install.sh`. **Pipe from curl?** Put flags after `bash -s --` (example: `… | bash -s -- --tools metrix,linex`). **`--help`** on either script lists the rest.

---

## Requirements

| Requirement | Notes |
|-------------|--------|
| Python | 3.10 or newer |
| ROCm | 6.0+ for GPU packages (use **7.0+** for Linex); skip if you only use host-side tools like `uprof_mcp` |
| GPU | **MI300+** for the full GPU experience; some pieces vary by tool — see each package’s README |
| uProf | **AMD uProf** on **x86** for `uprof_mcp` only — see that README |

For **development** on a subset of packages only, use editable installs (nothing to install at the monorepo root):

```bash
pip install -e metrix/
pip install -e linex/
```

## Try it

Try Metrix on your app (see [Metrix docs](metrix/README.md) and [examples](metrix/examples/)):

```python
from metrix import Metrix

profiler = Metrix()
results = profiler.profile("./your_app", metrics=["memory.hbm_bandwidth_utilization"])

for kernel in results.kernels:
    print(f"{kernel.name}: {kernel.duration_us.avg:.2f} μs")
```

---

## MCP quick config

With **uv** and a clone of this repo, you can point an MCP client at each package directory (adjust `/path/to/intellikit/...`):

```json
{
  "mcpServers": {
    "metrix-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/intellikit/metrix", "metrix-mcp"]
    },
    "kerncap-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/intellikit/kerncap", "kerncap-mcp"]
    },
    "hip-compiler-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/intellikit/rocm_mcp", "hip-compiler-mcp"]
    }
  }
}
```

If you installed with `pip` / `install.sh` instead, use the console script names (`metrix-mcp`, …) on your `PATH`, or the full path under your venv.

More servers and examples: [rocm_mcp/README.md](rocm_mcp/README.md), [AGENTS.md](AGENTS.md) (contributor-oriented, includes all entry point names).

<details>
<summary><strong>More install options</strong> (pip command, branch, dry-run, per-package Git URLs)</summary>

**Tools script** ([`install/tools/install.sh`](install/tools/install.sh))

- Default **`pip3`**; the script checks that pip’s Python is **3.10+** before installing (override with `--pip-cmd` if needed).
- Subset only: `--tools metrix,linex,nexus`
- Custom pip:  
  `curl -sSL .../install/tools/install.sh | bash -s -- --pip-cmd pip3.12`  
  or `--pip-cmd "python3.12 -m pip"`
- Branch/tag: `--ref my-branch`
- Different repo: `--repo-url https://github.com/you/fork.git`
- Preview: `--dry-run`

**Skills script** ([`install/skills/install.sh`](install/skills/install.sh))

- `--target cursor` | `claude` | `codex` | `agents` — where skills are written  
- `--global` — e.g. `~/.cursor/skills/` for Cursor  
- `--dry-run`

**Individual packages from Git**

```bash
pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=metrix"
# accordo, kerncap, linex, nexus, rocm_mcp, uprof_mcp — same pattern
```

**Editable from clone**

```bash
git clone https://github.com/AMDResearch/intellikit.git
cd intellikit
pip install -e ./accordo
pip install -e ./kerncap
# …any subset
```

</details>

---

## Example: profile → inspect → validate

```python
from metrix import Metrix
from nexus import Nexus
from accordo import Accordo

# 1) Baseline metrics
profiler = Metrix()
baseline = profiler.profile(
    "./app_baseline",
    metrics=["memory.hbm_bandwidth_utilization"],
)
baseline_bw = baseline.kernels[0].metrics["memory.hbm_bandwidth_utilization"].avg

# 2) See what ran on the GPU
trace = Nexus().run(["./app_baseline"])
for kernel in trace:
    print(kernel.name, len(kernel.assembly), "instructions")

# 3) After you optimize — check correctness
validator = Accordo(binary="./app_baseline", kernel_name="my_kernel")
ref = validator.capture_snapshot(binary="./app_baseline")
opt = validator.capture_snapshot(binary="./app_opt")
result = validator.compare_snapshots(ref, opt, tolerance=1e-6)

if result.is_valid:
    opt_results = profiler.profile(
        "./app_opt",
        metrics=["memory.hbm_bandwidth_utilization"],
    )
    opt_bw = opt_results.kernels[0].metrics["memory.hbm_bandwidth_utilization"].avg
    print(f"PASS — {result.num_arrays_validated} arrays matched; BW delta {opt_bw - baseline_bw:.1f}%")
```

---

## Contributing & support

We welcome issues and pull requests on [GitHub](https://github.com/AMDResearch/intellikit).

- **Bugs / ideas:** [Issues](https://github.com/AMDResearch/intellikit/issues)
- **License:** [MIT](LICENSE) — Copyright © 2025–2026 Advanced Micro Devices, Inc.

---

*Made for the next generation of GPU development — with or without an LLM in the loop.*
