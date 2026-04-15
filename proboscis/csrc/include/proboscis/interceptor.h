/******************************************************************************
 * MIT License
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/
#pragma once

#include "proboscis.h"

#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

// Forward declarations from hsa_api_trace.h (can't include directly due to
// broken relative includes in ROCm 6.x/7.x installed headers).
typedef void (*hsa_amd_queue_intercept_packet_writer)(const void* pkts, uint64_t pkt_count);

#ifndef PUBLIC_API
#define PUBLIC_API __attribute__((visibility("default")))
#endif

/// Per-dispatch metadata recorded by the interceptor.
struct DispatchRecord {
    uint64_t kernel_object;
    uint32_t grid_size[3];
    uint32_t workgroup_size[3];
    uint64_t kernarg_address;
};

/// Hash/compare for hsa_executable_symbol_t (opaque handle with .handle member).
struct SymbolHash {
    size_t operator()(hsa_executable_symbol_t s) const noexcept {
        return std::hash<uint64_t>{}(s.handle);
    }
};
struct SymbolEqual {
    bool operator()(hsa_executable_symbol_t a, hsa_executable_symbol_t b) const noexcept {
        return a.handle == b.handle;
    }
};

/// Static instruction analysis per kernel.
struct InstructionAnalysis {
    uint32_t global_loads = 0;
    uint32_t global_stores = 0;
    uint32_t flat_loads = 0;
    uint32_t flat_stores = 0;
    uint32_t buffer_loads = 0;
    uint32_t buffer_stores = 0;
    uint32_t ds_reads = 0;
    uint32_t ds_writes = 0;
    uint32_t atomics = 0;

    // By size
    uint32_t loads_1b = 0;
    uint32_t loads_2b = 0;
    uint32_t loads_4b = 0;
    uint32_t loads_8b = 0;
    uint32_t loads_12b = 0;
    uint32_t loads_16b = 0;
    uint32_t stores_1b = 0;
    uint32_t stores_2b = 0;
    uint32_t stores_4b = 0;
    uint32_t stores_8b = 0;
    uint32_t stores_12b = 0;
    uint32_t stores_16b = 0;

    uint32_t total_loads() const {
        return global_loads + flat_loads + buffer_loads + ds_reads;
    }
    uint32_t total_stores() const {
        return global_stores + flat_stores + buffer_stores + ds_writes;
    }
    uint32_t loads_gt_4b() const {
        return loads_8b + loads_12b + loads_16b;
    }
    uint32_t stores_gt_4b() const {
        return stores_8b + stores_12b + stores_16b;
    }
};

/// Proboscis HSA interceptor — hooks queue creation and kernel dispatch
/// to inject probe instrumentation via the hidden-argument ABI trick.
class ProboscisInterceptor {
public:
    static ProboscisInterceptor* getInstance();
    static void cleanup();

    void loadConfig();

    // HSA hook callbacks
    static hsa_status_t hsa_queue_create(
        hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
        void (*callback)(hsa_status_t, hsa_queue_t*, void*),
        void* data, uint32_t private_segment_size,
        uint32_t group_segment_size, hsa_queue_t** queue);

    static hsa_status_t hsa_executable_symbol_get_info(
        hsa_executable_symbol_t executable_symbol,
        hsa_executable_symbol_info_t attribute,
        void* value);

    static hsa_status_t hsa_executable_get_symbol_by_name(
        hsa_executable_t executable, const char* symbol_name,
        const hsa_agent_t* agent, hsa_executable_symbol_t* symbol);

    static hsa_status_t hsa_code_object_reader_create_from_memory(
        const void* code_object, size_t size,
        hsa_code_object_reader_t* code_object_reader);

    static void OnSubmitPackets(
        const void* in_packets, uint64_t count,
        uint64_t user_que_idx, void* data,
        hsa_amd_queue_intercept_packet_writer writer);

    // Probe buffer management
    void* allocateProbeBuffer(hsa_agent_t agent, size_t size);
    void freeProbeBuffer(void* ptr);
    void writeResults();

    bool shouldInstrument(const std::string& kernel_name) const;
    std::string getKernelName(uint64_t kernel_object);

private:
    ProboscisInterceptor() = default;
    ~ProboscisInterceptor();

    void addQueue(hsa_queue_t* queue, hsa_agent_t agent);
    void doPackets(hsa_queue_t* queue, const hsa_kernel_dispatch_packet_t* packets,
                   uint64_t count, hsa_amd_queue_intercept_packet_writer writer);

    // Code object patching
    bool patchCodeObject(const void* code_object, size_t size,
                         std::vector<uint8_t>& patched);
    void analyzeInstructions(const std::string& kernel_name,
                             const void* code_object, size_t size);

    static ProboscisInterceptor* singleton_;
    static std::mutex singleton_mutex_;

    ProbeConfig config_;
    std::map<hsa_queue_t*, hsa_agent_t> queues_;
    std::vector<void*> probe_buffers_;
    std::mutex mutex_;
    std::atomic<bool> shutting_down_{false};

    // Kernel name resolution (nexus pattern: two-level lookup)
    std::unordered_map<uint64_t, hsa_executable_symbol_t> handles_symbols_;
    std::unordered_map<hsa_executable_symbol_t, std::string, SymbolHash, SymbolEqual> symbols_names_;

    // Dispatch recording
    std::map<std::string, std::vector<DispatchRecord>> dispatch_records_;

    // Per-kernel metadata from code object patching
    std::map<std::string, ArgDescriptor> kernel_arg_descs_;

    // Static instruction analysis per kernel
    std::map<std::string, InstructionAnalysis> instruction_analysis_;

    // Patched code objects (keep alive so HSA can read them)
    std::vector<std::vector<uint8_t>> patched_code_objects_;

    // Per-dispatch probe buffers (for readback)
    struct ProbeBufferInfo {
        void* buffer;
        size_t size;
        uint64_t max_records;
        uint64_t record_size;
        hsa_agent_t agent;
    };
    std::map<std::string, std::vector<ProbeBufferInfo>> probe_buffer_records_;
};
