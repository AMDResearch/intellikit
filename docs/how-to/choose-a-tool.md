---
myst:
    html_meta:
        "description": "Choose the IntelliKit tool that answers your GPU performance question: Metrix, Linex, Omniprobe, Nexus, Kerncap, or Accordo."
        "keywords": "IntelliKit, choose tool, Metrix, Linex, Omniprobe, Nexus, Kerncap, Accordo, coalescing, bank conflicts"
---

# Choose a tool

Each IntelliKit tool answers a different question. Start with the cheapest tool that answers yours,
and move to a more specific one only when you need its detail.

| Question | Tool | Cost |
|----------|------|------|
| How long does a kernel take? What bandwidth, cache hit rate or FLOPS does it achieve? | Metrix | Runs the existing binary |
| Is memory coalescing poor, or are there LDS bank conflicts, in this kernel overall? | Metrix (`memory.coalescing_efficiency`, `memory.lds_bank_conflicts`) | Runs the existing binary |
| **Which access (file:line) is uncoalesced or causes the bank conflicts, and by how much?** | **Omniprobe** memory analysis | Rebuild with the Omniprobe plugin |
| Which source lines stall the most cycles, and on what? | Linex | Rebuild with `-g` |
| What kernels ran, with which arguments, and what assembly? | Nexus | Runs the existing binary |
| Isolate one kernel into a standalone reproducer | Kerncap | Runs the existing binary |
| Is an optimized kernel still correct? | Accordo | Two builds to compare |

## Memory inefficiency: Metrix, then Omniprobe

Metrix tells you *that* a kernel coalesces poorly or has bank conflicts, from hardware counters,
without a rebuild. Omniprobe tells you *where*: it instruments every memory instruction and reports
each source location's wasted cache lines and LDS conflict cycles, ranked worst first. Use Metrix to
decide whether to look, Omniprobe to find and to verify the fix. See [Omniprobe](../tools/omniprobe.md)
for how to build and how to read its numbers, including the accesses it cannot see.
