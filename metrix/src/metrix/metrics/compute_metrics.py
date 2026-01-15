"""
Compute-focused metric definitions (FLOPS, Arithmetic Intensity)
Based on Omnipilot's calculate_hbm_arithmetic_intensity() implementation

NOTE: The `derived_from` field contains CONCEPTUAL counter names for documentation.
Actual hardware counter names vary by architecture (e.g., TCC_EA_* vs TCC_EA0_*).
For architecture-specific counter names, see the backend implementations in
metrix/backends/gfx942.py, gfx1201.py, etc.
"""

from .categories import MetricCategory

# ═══════════════════════════════════════════════════════════════════
# COMPUTE THROUGHPUT METRICS
# ═══════════════════════════════════════════════════════════════════

COMPUTE_THROUGHPUT_METRICS = {
    "compute.total_flops": {
        "name": "Total FLOPS",
        "description": "Total floating-point operations performed by the kernel",
        "unit": "FLOPS",
        "category": MetricCategory.COMPUTE,
        # NOTE: HBM counters are architecture-specific:
        # - MI300 (gfx942): TCC_EA0_RDREQ_sum, TCC_EA0_WRREQ_sum, etc.
        # - MI200 (gfx90a): TCC_EA_RDREQ_sum, TCC_EA_WRREQ_sum, etc.
        "derived_from": [
            # FP16 instructions
            "SQ_INSTS_VALU_ADD_F16",
            "SQ_INSTS_VALU_MUL_F16",
            "SQ_INSTS_VALU_TRANS_F16",
            "SQ_INSTS_VALU_FMA_F16",
            # FP32 instructions
            "SQ_INSTS_VALU_ADD_F32",
            "SQ_INSTS_VALU_MUL_F32",
            "SQ_INSTS_VALU_TRANS_F32",
            "SQ_INSTS_VALU_FMA_F32",
            # FP64 instructions
            "SQ_INSTS_VALU_ADD_F64",
            "SQ_INSTS_VALU_MUL_F64",
            "SQ_INSTS_VALU_TRANS_F64",
            "SQ_INSTS_VALU_FMA_F64",
            # MFMA instructions (Matrix FMA)
            "SQ_INSTS_VALU_MFMA_MOPS_F16",
            "SQ_INSTS_VALU_MFMA_MOPS_BF16",
            "SQ_INSTS_VALU_MFMA_MOPS_F32",
            "SQ_INSTS_VALU_MFMA_MOPS_F64",
        ],
        "formula": """
            # 64 operations per wave (wavefront size = 64)
            # FMA counts as 2 operations (multiply + add)
            # MFMA instructions produce 512 operations per instruction

            fops = 64 * (
                (
                    SQ_INSTS_VALU_ADD_F16 +
                    SQ_INSTS_VALU_MUL_F16 +
                    SQ_INSTS_VALU_TRANS_F16 +
                    SQ_INSTS_VALU_FMA_F16 * 2
                ) +
                (
                    SQ_INSTS_VALU_ADD_F32 +
                    SQ_INSTS_VALU_MUL_F32 +
                    SQ_INSTS_VALU_TRANS_F32 +
                    SQ_INSTS_VALU_FMA_F32 * 2
                ) +
                (
                    SQ_INSTS_VALU_ADD_F64 +
                    SQ_INSTS_VALU_MUL_F64 +
                    SQ_INSTS_VALU_TRANS_F64 +
                    SQ_INSTS_VALU_FMA_F64 * 2
                )
            ) + 512 * (
                SQ_INSTS_VALU_MFMA_MOPS_F16 +
                SQ_INSTS_VALU_MFMA_MOPS_BF16 +
                SQ_INSTS_VALU_MFMA_MOPS_F32 +
                SQ_INSTS_VALU_MFMA_MOPS_F64
            )

            return fops
        """,
    },
    "compute.hbm_gflops": {
        "name": "HBM Compute Throughput",
        "description": "Compute throughput (GFLOPS) normalized by kernel execution time",
        "unit": "GFLOPS",
        "category": MetricCategory.COMPUTE,
        "derived_from": [
            # All FLOPS counters
            "SQ_INSTS_VALU_ADD_F16",
            "SQ_INSTS_VALU_MUL_F16",
            "SQ_INSTS_VALU_TRANS_F16",
            "SQ_INSTS_VALU_FMA_F16",
            "SQ_INSTS_VALU_ADD_F32",
            "SQ_INSTS_VALU_MUL_F32",
            "SQ_INSTS_VALU_TRANS_F32",
            "SQ_INSTS_VALU_FMA_F32",
            "SQ_INSTS_VALU_ADD_F64",
            "SQ_INSTS_VALU_MUL_F64",
            "SQ_INSTS_VALU_TRANS_F64",
            "SQ_INSTS_VALU_FMA_F64",
            "SQ_INSTS_VALU_MFMA_MOPS_F16",
            "SQ_INSTS_VALU_MFMA_MOPS_BF16",
            "SQ_INSTS_VALU_MFMA_MOPS_F32",
            "SQ_INSTS_VALU_MFMA_MOPS_F64",
            "GRBM_GUI_ACTIVE",
        ],
        "formula": """
            # Calculate total FLOPS (same as compute.total_flops)
            fops = 64 * (
                (
                    SQ_INSTS_VALU_ADD_F16 +
                    SQ_INSTS_VALU_MUL_F16 +
                    SQ_INSTS_VALU_TRANS_F16 +
                    SQ_INSTS_VALU_FMA_F16 * 2
                ) +
                (
                    SQ_INSTS_VALU_ADD_F32 +
                    SQ_INSTS_VALU_MUL_F32 +
                    SQ_INSTS_VALU_TRANS_F32 +
                    SQ_INSTS_VALU_FMA_F32 * 2
                ) +
                (
                    SQ_INSTS_VALU_ADD_F64 +
                    SQ_INSTS_VALU_MUL_F64 +
                    SQ_INSTS_VALU_TRANS_F64 +
                    SQ_INSTS_VALU_FMA_F64 * 2
                )
            ) + 512 * (
                SQ_INSTS_VALU_MFMA_MOPS_F16 +
                SQ_INSTS_VALU_MFMA_MOPS_BF16 +
                SQ_INSTS_VALU_MFMA_MOPS_F32 +
                SQ_INSTS_VALU_MFMA_MOPS_F64
            )

            # Convert to GFLOPS
            time_seconds = GRBM_GUI_ACTIVE / (gpu_freq_mhz * 1e6)
            gflops = (fops / 1e9) / time_seconds if time_seconds > 0 else 0

            return gflops
        """,
        "device_specific": True,
    },
}

# ═══════════════════════════════════════════════════════════════════
# ARITHMETIC INTENSITY METRICS
# ═══════════════════════════════════════════════════════════════════

ARITHMETIC_INTENSITY_METRICS = {
    "compute.hbm_arithmetic_intensity": {
        "name": "HBM Arithmetic Intensity",
        "description": "Ratio of floating-point operations to HBM bytes transferred (FLOP/byte)",
        "unit": "FLOP/byte",
        "category": MetricCategory.COMPUTE,
        "derived_from": [
            # FLOPS counters (same across architectures)
            "SQ_INSTS_VALU_ADD_F16",
            "SQ_INSTS_VALU_MUL_F16",
            "SQ_INSTS_VALU_TRANS_F16",
            "SQ_INSTS_VALU_FMA_F16",
            "SQ_INSTS_VALU_ADD_F32",
            "SQ_INSTS_VALU_MUL_F32",
            "SQ_INSTS_VALU_TRANS_F32",
            "SQ_INSTS_VALU_FMA_F32",
            "SQ_INSTS_VALU_ADD_F64",
            "SQ_INSTS_VALU_MUL_F64",
            "SQ_INSTS_VALU_TRANS_F64",
            "SQ_INSTS_VALU_FMA_F64",
            "SQ_INSTS_VALU_MFMA_MOPS_F16",
            "SQ_INSTS_VALU_MFMA_MOPS_BF16",
            "SQ_INSTS_VALU_MFMA_MOPS_F32",
            "SQ_INSTS_VALU_MFMA_MOPS_F64",
            # HBM bandwidth counters - conceptual names (actual names vary by arch)
            "TCC_EA_RDREQ_32B_sum",
            "TCC_EA_RDREQ_sum",
            "TCC_BUBBLE_sum",
            "TCC_EA_WRREQ_64B_sum",
            "TCC_EA_WRREQ_sum",
        ],
        "formula": """
            # Calculate total FLOPS
            fops = 64 * (
                (
                    SQ_INSTS_VALU_ADD_F16 +
                    SQ_INSTS_VALU_MUL_F16 +
                    SQ_INSTS_VALU_TRANS_F16 +
                    SQ_INSTS_VALU_FMA_F16 * 2
                ) +
                (
                    SQ_INSTS_VALU_ADD_F32 +
                    SQ_INSTS_VALU_MUL_F32 +
                    SQ_INSTS_VALU_TRANS_F32 +
                    SQ_INSTS_VALU_FMA_F32 * 2
                ) +
                (
                    SQ_INSTS_VALU_ADD_F64 +
                    SQ_INSTS_VALU_MUL_F64 +
                    SQ_INSTS_VALU_TRANS_F64 +
                    SQ_INSTS_VALU_FMA_F64 * 2
                )
            ) + 512 * (
                SQ_INSTS_VALU_MFMA_MOPS_F16 +
                SQ_INSTS_VALU_MFMA_MOPS_BF16 +
                SQ_INSTS_VALU_MFMA_MOPS_F32 +
                SQ_INSTS_VALU_MFMA_MOPS_F64
            )

            # Calculate HBM bytes (with 32B/64B/128B distinction)
            # Note: TCC_BUBBLE_sum counts 128B read requests on MI300
            hbm_rd = (TCC_BUBBLE_sum * 128 +
                     (TCC_EA_RDREQ_sum - TCC_BUBBLE_sum - TCC_EA_RDREQ_32B_sum) * 64 +
                     TCC_EA_RDREQ_32B_sum * 32)
            hbm_wr = (TCC_EA_WRREQ_64B_sum * 64 +
                     (TCC_EA_WRREQ_sum - TCC_EA_WRREQ_64B_sum) * 32)
            hbm_bytes = hbm_rd + hbm_wr

            # Arithmetic intensity = FLOP / byte
            ai_hbm = fops / hbm_bytes if hbm_bytes > 0 else 0

            return ai_hbm
        """,
        "interpretation": {
            "excellent": (10, float("inf"), "Compute bound - excellent FLOP/byte ratio"),
            "good": (5, 10, "Good balance between compute and memory"),
            "fair": (1, 5, "Memory bound - moderate FLOP/byte ratio"),
            "poor": (0, 1, "Heavily memory bound - low FLOP/byte ratio"),
        },
    },
    "compute.l2_arithmetic_intensity": {
        "name": "L2 Arithmetic Intensity",
        "description": "Ratio of floating-point operations to L2 cache bytes accessed (FLOP/byte)",
        "unit": "FLOP/byte",
        "category": MetricCategory.COMPUTE,
        "derived_from": [
            # FLOPS counters
            "SQ_INSTS_VALU_ADD_F16",
            "SQ_INSTS_VALU_MUL_F16",
            "SQ_INSTS_VALU_TRANS_F16",
            "SQ_INSTS_VALU_FMA_F16",
            "SQ_INSTS_VALU_ADD_F32",
            "SQ_INSTS_VALU_MUL_F32",
            "SQ_INSTS_VALU_TRANS_F32",
            "SQ_INSTS_VALU_FMA_F32",
            "SQ_INSTS_VALU_ADD_F64",
            "SQ_INSTS_VALU_MUL_F64",
            "SQ_INSTS_VALU_TRANS_F64",
            "SQ_INSTS_VALU_FMA_F64",
            "SQ_INSTS_VALU_MFMA_MOPS_F16",
            "SQ_INSTS_VALU_MFMA_MOPS_BF16",
            "SQ_INSTS_VALU_MFMA_MOPS_F32",
            "SQ_INSTS_VALU_MFMA_MOPS_F64",
            # L2 cache counters
            "TCC_REQ_sum",
        ],
        "formula": """
            # Calculate total FLOPS
            fops = 64 * (
                (
                    SQ_INSTS_VALU_ADD_F16 +
                    SQ_INSTS_VALU_MUL_F16 +
                    SQ_INSTS_VALU_TRANS_F16 +
                    SQ_INSTS_VALU_FMA_F16 * 2
                ) +
                (
                    SQ_INSTS_VALU_ADD_F32 +
                    SQ_INSTS_VALU_MUL_F32 +
                    SQ_INSTS_VALU_TRANS_F32 +
                    SQ_INSTS_VALU_FMA_F32 * 2
                ) +
                (
                    SQ_INSTS_VALU_ADD_F64 +
                    SQ_INSTS_VALU_MUL_F64 +
                    SQ_INSTS_VALU_TRANS_F64 +
                    SQ_INSTS_VALU_FMA_F64 * 2
                )
            ) + 512 * (
                SQ_INSTS_VALU_MFMA_MOPS_F16 +
                SQ_INSTS_VALU_MFMA_MOPS_BF16 +
                SQ_INSTS_VALU_MFMA_MOPS_F32 +
                SQ_INSTS_VALU_MFMA_MOPS_F64
            )

            # Calculate L2 bytes (L2 cache line is 128 bytes)
            l2_bytes = TCC_REQ_sum * 128

            # Arithmetic intensity = FLOP / byte
            ai_l2 = fops / l2_bytes if l2_bytes > 0 else 0

            return ai_l2
        """,
    },
    "compute.l1_arithmetic_intensity": {
        "name": "L1 Arithmetic Intensity",
        "description": "Ratio of floating-point operations to L1 cache bytes accessed (FLOP/byte)",
        "unit": "FLOP/byte",
        "category": MetricCategory.COMPUTE,
        "derived_from": [
            # FLOPS counters
            "SQ_INSTS_VALU_ADD_F16",
            "SQ_INSTS_VALU_MUL_F16",
            "SQ_INSTS_VALU_TRANS_F16",
            "SQ_INSTS_VALU_FMA_F16",
            "SQ_INSTS_VALU_ADD_F32",
            "SQ_INSTS_VALU_MUL_F32",
            "SQ_INSTS_VALU_TRANS_F32",
            "SQ_INSTS_VALU_FMA_F32",
            "SQ_INSTS_VALU_ADD_F64",
            "SQ_INSTS_VALU_MUL_F64",
            "SQ_INSTS_VALU_TRANS_F64",
            "SQ_INSTS_VALU_FMA_F64",
            "SQ_INSTS_VALU_MFMA_MOPS_F16",
            "SQ_INSTS_VALU_MFMA_MOPS_BF16",
            "SQ_INSTS_VALU_MFMA_MOPS_F32",
            "SQ_INSTS_VALU_MFMA_MOPS_F64",
            # L1 cache counters
            "TCP_TOTAL_CACHE_ACCESSES_sum",
        ],
        "formula": """
            # Calculate total FLOPS
            fops = 64 * (
                (
                    SQ_INSTS_VALU_ADD_F16 +
                    SQ_INSTS_VALU_MUL_F16 +
                    SQ_INSTS_VALU_TRANS_F16 +
                    SQ_INSTS_VALU_FMA_F16 * 2
                ) +
                (
                    SQ_INSTS_VALU_ADD_F32 +
                    SQ_INSTS_VALU_MUL_F32 +
                    SQ_INSTS_VALU_TRANS_F32 +
                    SQ_INSTS_VALU_FMA_F32 * 2
                ) +
                (
                    SQ_INSTS_VALU_ADD_F64 +
                    SQ_INSTS_VALU_MUL_F64 +
                    SQ_INSTS_VALU_TRANS_F64 +
                    SQ_INSTS_VALU_FMA_F64 * 2
                )
            ) + 512 * (
                SQ_INSTS_VALU_MFMA_MOPS_F16 +
                SQ_INSTS_VALU_MFMA_MOPS_BF16 +
                SQ_INSTS_VALU_MFMA_MOPS_F32 +
                SQ_INSTS_VALU_MFMA_MOPS_F64
            )

            # Calculate L1 bytes (L1 cache line is 64 bytes)
            l1_bytes = TCP_TOTAL_CACHE_ACCESSES_sum * 64

            # Arithmetic intensity = FLOP / byte
            ai_l1 = fops / l1_bytes if l1_bytes > 0 else 0

            return ai_l1
        """,
    },
}

# ═══════════════════════════════════════════════════════════════════
# COMBINED COMPUTE METRIC CATALOG
# ═══════════════════════════════════════════════════════════════════

COMPUTE_METRICS = {**COMPUTE_THROUGHPUT_METRICS, **ARITHMETIC_INTENSITY_METRICS}
