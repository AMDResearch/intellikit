---
title: IntelliKit
description: A Python toolkit for AMD GPU performance analysis and validation.
template: splash
hero:
  tagline: Agent-first tooling for AMD hardware.
  actions:
    - text: Get started
      link: /intellikit/getting-started/installation/
      icon: right-arrow
      variant: primary
    - text: GitHub
      link: https://github.com/AMDResearch/intellikit
      icon: external
---

IntelliKit is a set of Python tools for AMD GPU performance analysis and validation — profiling kernels, inspecting assembly, mapping source lines, and validating optimizations with ROCm.

| Tool | Role | Description |
|------|------|-------------|
| **Kerncap** | Isolate | Capture kernel dispatches, build standalone reproducers for HIP and Triton |
| **Metrix** | Profile | Human-readable metrics from hardware counters: bandwidth, cache, compute |
| **Linex** | Profile | Source-line timing and stall analysis |
| **Nexus** | Inspect | Intercept HSA packets to see what ran on the GPU |
| **Accordo** | Validate | Prove an optimized kernel matches a reference implementation |
| **ROCm MCP** | MCP | HIP compiler, HIP docs, and rocminfo servers for LLM agents |
| **uProf MCP** | CPU | MCP bridge to AMD uProf for CPU hotspot analysis |
