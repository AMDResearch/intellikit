/******************************************************************************
 * MIT License
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Proboscis HSA interceptor — loaded via HSA_TOOLS_LIB.
 *
 * Hooks:
 *   - hsa_queue_create → creates intercept queues
 *   - OnSubmitPackets  → inspects dispatch packets, repacks kernargs
 *
 * This is the C++ runtime that makes the hidden-argument ABI trick work
 * at dispatch time. The Python layer generates the probe config and
 * the ELF surgery; this layer handles the live interception.
 ******************************************************************************/

#include "proboscis/interceptor.h"
#include "proboscis/proboscis.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>

// ─── Declarations from hsa_api_trace.h ─────────────────────────────────────
// The installed hsa_api_trace.h in ROCm 6.x/7.x has broken relative includes
// (inc/hsa_ext_image.h) that prevent compilation. We reproduce the minimal
// declarations needed: intercept API functions, callback typedef, and the
// API table structs with ABI-compatible layout verified against rocprofiler-sdk
// ROCP_SDK_ENFORCE_ABI static assertions.

// Intercept queue callback and API functions (from hsa_api_trace.h)
typedef void (*hsa_amd_queue_intercept_packet_writer)(const void* pkts, uint64_t pkt_count);
typedef void (*hsa_amd_queue_intercept_handler)(const void* pkts, uint64_t pkt_count,
                                                uint64_t user_pkt_index, void* data,
                                                hsa_amd_queue_intercept_packet_writer writer);
hsa_status_t hsa_amd_queue_intercept_register(hsa_queue_t* queue,
                                              hsa_amd_queue_intercept_handler callback,
                                              void* user_data);
hsa_status_t hsa_amd_queue_intercept_create(
    hsa_agent_t agent_handle, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data), void* data,
    uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t** queue);

struct ApiTableVersion {
    uint32_t major_id;
    uint32_t minor_id;
    uint32_t step_id;
    uint32_t reserved;
};

// CoreApiTable: version + 7 fn ptrs before hsa_queue_create_fn (field 8).
// We need fields 8 (hsa_queue_create_fn), 10 (hsa_queue_destroy_fn),
// and 94 (hsa_executable_symbol_get_info_fn).
struct CoreApiTable {
    ApiTableVersion version;                                        // offset 0
    decltype(hsa_init)* hsa_init_fn;                                // 1
    decltype(hsa_shut_down)* hsa_shut_down_fn;                      // 2
    decltype(hsa_system_get_info)* hsa_system_get_info_fn;          // 3
    void* _pad_4;                                                   // 4
    void* _pad_5;                                                   // 5
    decltype(hsa_iterate_agents)* hsa_iterate_agents_fn;            // 6
    decltype(hsa_agent_get_info)* hsa_agent_get_info_fn;            // 7
    decltype(hsa_queue_create)* hsa_queue_create_fn;                // 8
    decltype(hsa_soft_queue_create)* hsa_soft_queue_create_fn;      // 9
    decltype(hsa_queue_destroy)* hsa_queue_destroy_fn;              // 10
    // Fields 11-93: we don't need to enumerate them all, but the struct layout
    // must place hsa_executable_symbol_get_info_fn at field 94.
    // 94 - 11 = 83 padding slots.
    void* _core_pad[83];                                            // 11-93
    decltype(hsa_executable_symbol_get_info)* hsa_executable_symbol_get_info_fn; // 94
    // There are more fields after 94 but we don't access them.
};

// AmdExtTable: ROCP_SDK_ENFORCE_ABI uses member-index counting (0-indexed):
// member 0 = version, members 1-37 = 37 function pointers, member 38 = intercept_create.
struct AmdExtTable {
    ApiTableVersion version;                                        // member 0
    void* _amd_pad[37];                                             // members 1-37
    decltype(hsa_amd_queue_intercept_create)* hsa_amd_queue_intercept_create_fn;   // member 38
    decltype(hsa_amd_queue_intercept_register)* hsa_amd_queue_intercept_register_fn; // member 39
};

struct HsaApiTable {
    ApiTableVersion version;
    CoreApiTable* core_;
    AmdExtTable* amd_ext_;
    // finalizer_ext_, image_ext_, tools_, pc_sampling_ext_ follow but we don't use them.
};

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

// Saved original HSA function pointers
static decltype(hsa_queue_create)* orig_hsa_queue_create = nullptr;
static decltype(hsa_queue_destroy)* orig_hsa_queue_destroy = nullptr;
static decltype(hsa_amd_queue_intercept_create)* orig_intercept_create = nullptr;
static decltype(hsa_amd_queue_intercept_register)* orig_intercept_register = nullptr;
static decltype(hsa_executable_symbol_get_info)* orig_symbol_get_info = nullptr;

// Singleton
ProboscisInterceptor* ProboscisInterceptor::singleton_ = nullptr;
std::mutex ProboscisInterceptor::singleton_mutex_;

ProboscisInterceptor* ProboscisInterceptor::getInstance() {
    std::lock_guard<std::mutex> lock(singleton_mutex_);
    if (!singleton_) {
        singleton_ = new ProboscisInterceptor();
        singleton_->loadConfig();
    }
    return singleton_;
}

void ProboscisInterceptor::cleanup() {
    std::lock_guard<std::mutex> lock(singleton_mutex_);
    if (singleton_) {
        singleton_->writeResults();
        delete singleton_;
        singleton_ = nullptr;
    }
}

ProboscisInterceptor::~ProboscisInterceptor() {
    shutting_down_.store(true);
    // Free all probe buffers
    for (auto* buf : probe_buffers_) {
        if (buf) hsa_memory_free(buf);
    }
}

void ProboscisInterceptor::loadConfig() {
    const char* config_path = std::getenv("PROBOSCIS_CONFIG");
    const char* results_path = std::getenv("PROBOSCIS_RESULTS");
    const char* log_level = std::getenv("PROBOSCIS_LOG_LEVEL");

    config_.log_level = log_level ? std::atoi(log_level) : 1;
    config_.results_path = results_path ? results_path : "/tmp/proboscis_results.json";

    if (config_path) {
        // Read JSON config file
        std::ifstream f(config_path);
        if (f.good()) {
            std::ostringstream ss;
            ss << f.rdbuf();
            // Simple JSON parsing for probe_spec fields
            // In production, use a proper JSON parser
            std::string content = ss.str();
            // Extract probe_type
            auto extract = [&content](const std::string& key) -> std::string {
                auto pos = content.find("\"" + key + "\"");
                if (pos == std::string::npos) return "";
                pos = content.find("\"", pos + key.size() + 2);
                if (pos == std::string::npos) return "";
                auto end = content.find("\"", pos + 1);
                if (end == std::string::npos) return "";
                return content.substr(pos + 1, end - pos - 1);
            };
            config_.probe_type = extract("probe_type");
            config_.target_kernel = extract("target_kernel");

            if (config_.log_level >= 2) {
                std::cerr << "[proboscis] Config loaded: probe_type=" << config_.probe_type
                          << " target=" << config_.target_kernel << std::endl;
            }
        }
    }
}

bool ProboscisInterceptor::shouldInstrument(const std::string& kernel_name) const {
    if (config_.target_kernel.empty()) return true;
    return kernel_name.find(config_.target_kernel) != std::string::npos;
}

void ProboscisInterceptor::addQueue(hsa_queue_t* queue, hsa_agent_t agent) {
    std::lock_guard<std::mutex> lock(mutex_);
    queues_[queue] = agent;
}

void ProboscisInterceptor::OnSubmitPackets(
    const void* in_packets, uint64_t count,
    uint64_t /*user_que_idx*/, void* data,
    hsa_amd_queue_intercept_packet_writer writer)
{
    auto* self = getInstance();
    if (self && !self->shutting_down_.load()) {
        auto* queue = reinterpret_cast<hsa_queue_t*>(data);
        self->doPackets(
            queue,
            static_cast<const hsa_kernel_dispatch_packet_t*>(in_packets),
            count, writer);
    } else {
        writer(in_packets, count);
    }
}

void ProboscisInterceptor::doPackets(
    hsa_queue_t* /*queue*/,
    const hsa_kernel_dispatch_packet_t* packets,
    uint64_t count,
    hsa_amd_queue_intercept_packet_writer writer)
{
    // For now, pass through all packets unmodified.
    // The kernarg repacking logic will be added here once
    // the ELF surgery pipeline is validated end-to-end.
    writer(packets, count);
}

void* ProboscisInterceptor::allocateProbeBuffer(hsa_agent_t agent, size_t size) {
    void* ptr = nullptr;
    // Find a memory pool that supports kernel agent access
    hsa_amd_memory_pool_t pool{};
    hsa_amd_agent_iterate_memory_pools(agent,
        [](hsa_amd_memory_pool_t pool, void* data) -> hsa_status_t {
            hsa_amd_segment_t segment;
            hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
            if (segment == HSA_AMD_SEGMENT_GLOBAL) {
                bool accessible = false;
                hsa_amd_memory_pool_get_info(pool,
                    HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &accessible);
                if (accessible) {
                    *static_cast<hsa_amd_memory_pool_t*>(data) = pool;
                    return HSA_STATUS_INFO_BREAK;
                }
            }
            return HSA_STATUS_SUCCESS;
        }, &pool);

    if (hsa_amd_memory_pool_allocate(pool, size, 0, &ptr) == HSA_STATUS_SUCCESS) {
        std::memset(ptr, 0, size);
        std::lock_guard<std::mutex> lock(mutex_);
        probe_buffers_.push_back(ptr);
        return ptr;
    }
    return nullptr;
}

void ProboscisInterceptor::freeProbeBuffer(void* ptr) {
    if (ptr) {
        hsa_memory_free(ptr);
        std::lock_guard<std::mutex> lock(mutex_);
        probe_buffers_.erase(
            std::remove(probe_buffers_.begin(), probe_buffers_.end(), ptr),
            probe_buffers_.end());
    }
}

void ProboscisInterceptor::writeResults() {
    if (config_.results_path.empty()) return;

    std::ofstream f(config_.results_path);
    if (!f.good()) {
        std::cerr << "[proboscis] Failed to write results to " << config_.results_path << std::endl;
        return;
    }

    // Write collected results as JSON
    f << "{\"kernels\": {}}" << std::endl;

    if (config_.log_level >= 1) {
        std::cerr << "[proboscis] Results written to " << config_.results_path << std::endl;
    }
}

// ─── HSA Queue Hook ──────────────────────────────────────────────────────────

hsa_status_t ProboscisInterceptor::hsa_queue_create(
    hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t, hsa_queue_t*, void*),
    void* data, uint32_t private_segment_size,
    uint32_t group_segment_size, hsa_queue_t** queue)
{
    hsa_status_t result = orig_intercept_create(
        agent, size, type, callback, data,
        private_segment_size, group_segment_size, queue);

    if (result == HSA_STATUS_SUCCESS) {
        auto* self = getInstance();
        self->addQueue(*queue, agent);
        orig_intercept_register(*queue, OnSubmitPackets, reinterpret_cast<void*>(*queue));

        if (self->config_.log_level >= 2) {
            std::cerr << "[proboscis] Intercepted queue creation" << std::endl;
        }
    }
    return result;
}

// ─── HSA_TOOLS_LIB Entry Points ─────────────────────────────────────────────

extern "C" {

PUBLIC_API bool OnLoad(
    HsaApiTable* table,
    uint64_t /*runtime_version*/,
    uint64_t /*failed_tool_count*/,
    const char* const* /*failed_tool_names*/)
{
    if (!table) {
        std::cerr << "[proboscis] ERROR: HSA API table is NULL" << std::endl;
        return false;
    }

    // Save original function pointers
    orig_hsa_queue_create = table->core_->hsa_queue_create_fn;
    orig_hsa_queue_destroy = table->core_->hsa_queue_destroy_fn;
    orig_intercept_create = table->amd_ext_->hsa_amd_queue_intercept_create_fn;
    orig_intercept_register = table->amd_ext_->hsa_amd_queue_intercept_register_fn;
    orig_symbol_get_info = table->core_->hsa_executable_symbol_get_info_fn;

    // Hook queue creation
    table->core_->hsa_queue_create_fn = ProboscisInterceptor::hsa_queue_create;

    // Initialize the interceptor
    ProboscisInterceptor::getInstance();

    std::cerr << "[proboscis] Tool loaded — intercepting GPU dispatches" << std::endl;
    return true;
}

PUBLIC_API void OnUnload() {
    ProboscisInterceptor::cleanup();
    std::cerr << "[proboscis] Tool unloaded" << std::endl;
}

} // extern "C"
