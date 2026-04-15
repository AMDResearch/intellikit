/******************************************************************************
 * MIT License
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <map>
#include <vector>
#include <mutex>

#include <hsa/hsa.h>

#define PROBOSCIS_PREFIX "__proboscis_"
#define PROBOSCIS_HIDDEN_ARG "hidden_proboscis_ctx"

/// Probe buffer header — first 24 bytes of the GPU-visible probe buffer.
/// The instrumented kernel atomically increments write_offset to claim a slot,
/// then writes the record at header_size + slot * record_size.
struct ProbeBufferHeader {
    uint64_t write_offset;   // atomic counter — next slot index
    uint64_t max_records;    // capacity
    uint64_t record_size;    // bytes per record
};

/// Argument descriptor for kernarg repacking.
/// Mirrors omniprobe's arg_descriptor_t.
struct ArgDescriptor {
    uint32_t explicit_args_length;
    uint32_t hidden_args_length;
    uint32_t kernarg_length;            // total size of original kernarg
    uint32_t instrumented_kernarg_length; // total size with hidden_proboscis_ctx
    uint32_t probe_ctx_offset;          // offset of hidden_proboscis_ctx in instrumented layout
};

/// Per-kernel metadata collected during code object loading.
struct KernelInfo {
    std::string name;
    std::string symbol;
    hsa_agent_t agent;
    uint64_t kernel_object;
    uint32_t kernarg_size;
    ArgDescriptor arg_desc;
};

/// Configuration loaded from PROBOSCIS_CONFIG JSON.
struct ProbeConfig {
    std::string probe_type;       // "memory_trace", "block_count", "register_snapshot"
    std::string target_kernel;    // kernel name pattern (empty = all)
    int sample_rate;
    int max_records;
    int log_level;
    std::string results_path;     // PROBOSCIS_RESULTS env var
};

