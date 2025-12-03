# GPU Benchmark Results - AMD Instinct MI300X

## Reproduction

To reproduce all results:
```bash
cd atomic_latnecy && ../bench.sh high.hip && ../bench.sh low.hip && cd ..
cd coalescing && ../bench.sh coalesced.hip && ../bench.sh uncoalesced.hip && cd ..
cd bank_conflict && ../bench.sh conflict.hip && ../bench.sh no_conflict.hip && cd ..
cd l2_cache && ../bench.sh high_hit.hip && ../bench.sh low_hit.hip && cd ..
```

---

## 1. Atomic Contention (`atomic_latnecy/`)

**Metric:** `17.2.11 Atomic Latency` (Cycles)

| Benchmark | Value |
|-----------|-------|
| High Contention (high.hip) | 1936.96 |
| Low Contention (low.hip) | 381.68 |

```
│ 17.2.11     │ Atomic Latency                    │ 1936.96 │ 1936.96 │ 1936.96 │ Cycles         │
│ 17.2.11     │ Atomic Latency                    │ 381.68  │ 381.68  │ 381.68  │ Cycles         │
```

---

## 2. Memory Coalescing (`coalescing/`)

**Metric:** `16.1.3 Coalescing` (Pct of peak)

| Benchmark | Value |
|-----------|-------|
| Coalesced (coalesced.hip) | 100.00 |
| Uncoalesced (uncoalesced.hip) | 25.00 |

```
│ 16.1.3      │ Coalescing  │ 100.00 │ Pct of peak │
│ 16.1.3      │ Coalescing  │  25.00 │ Pct of peak │
```

---

## 3. Bank Conflicts (`bank_conflict/`)

**Metric:** `2.1.17 LDS Bank Conflicts/Access` (Conflicts/access)

| Benchmark | Value |
|-----------|-------|
| With Conflicts (conflict.hip) | 15.50 |
| No Conflicts (no_conflict.hip) | 0.00 |

```
│ 2.1.17      │ LDS Bank Conflicts/Access │   15.50 │ Conflicts/access │ 32.0      │ 48.44         │
│ 2.1.17      │ LDS Bank Conflicts/Access │    0.00 │ Conflicts/access │ 32.0      │ 0.0           │
```

---

## 4. L2 Cache Hit Rate (`l2_cache/`)

**Metric:** `2.1.20 L2 Cache Hit Rate` (Pct)

| Benchmark | Value |
|-----------|-------|
| High Hit Rate (high_hit.hip) | 37.22 |
| Low Hit Rate (low_hit.hip) | 21.79 |

```
│ 2.1.20      │ L2 Cache Hit Rate         │ 37.22    │ Pct              │ 100.0     │ 37.22         │
│ 2.1.20      │ L2 Cache Hit Rate         │ 21.79   │ Pct              │ 100.0     │ 21.79         │
```

---

## Raw Hardware Counters and Computed Metrics

### 1. Average Atomic Latency

**Formula:** `atomic_lat = TCC_EA0_ATOMIC_LEVEL_sum / TCC_EA0_ATOMIC_sum`

| Benchmark | TCC_EA0_ATOMIC_LEVEL_sum | TCC_EA0_ATOMIC_sum | Computed Value | Reported Value |
|-----------|--------------------------|-------------------|----------------|----------------|
| High Contention | 247931.0 | 128.0 | 1936.96 cycles | 1936.96 cycles |
| Low Contention | 195422.0 | 512.0 | 381.68 cycles | 381.68 cycles |

### 2. Coalescing

**Formula:** `coal = ((TA_TOTAL_WAVEFRONTS_sum * 64) * 100) / (TCP_TOTAL_ACCESSES_sum * 4)`

| Benchmark | TA_TOTAL_WAVEFRONTS_sum | TCP_TOTAL_ACCESSES_sum | Computed Value | Reported Value |
|-----------|-------------------------|------------------------|----------------|----------------|
| Coalesced | 2048.0 | 32768.0 | 100.00% | 100.00% |
| Uncoalesced | 2048.0 | 131072.0 | 25.00% | 25.00% |


### 3. Bank Conflicts

**Formula:** `conflicts_per_access = SQ_LDS_BANK_CONFLICT / (SQ_LDS_IDX_ACTIVE - SQ_LDS_BANK_CONFLICT)`

| Benchmark | SQ_LDS_IDX_ACTIVE | SQ_LDS_BANK_CONFLICT | quo (act - conf) | Computed Value | Reported Value |
|-----------|-------------------|---------------------|------------------|----------------|----------------|
| With Conflicts | 1081344 | 1015808 | 65536 | 15.50 | 15.50 |
| No Conflicts | 65536 | 0 | 65536 | 0.00 | 0.00 |

### 4. L2 Cache Hit Rate

**Formula:** `hr = (100 * TCC_HIT_sum) / (TCC_HIT_sum + TCC_MISS_sum)`

| Benchmark | TCC_HIT_sum | TCC_MISS_sum | Computed Value | Reported Value |
|-----------|-------------|--------------|----------------|----------------|
| High Hit Rate | 19496.0 | 32888.0 | 37.22% | 37.22% |
| Low Hit Rate | 11464.0 | 41138.0 | 21.79% | 21.79% |

---

## References

- [Coalescing Documentation](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/vector-l1-cache.html)
- [Atomic Latency Documentation](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/l2-cache.html)
- [Bank Conflicts Documentation](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/local-data-share.html)
- [L2 Hit Rate Documentation](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/l2-cache.html)

