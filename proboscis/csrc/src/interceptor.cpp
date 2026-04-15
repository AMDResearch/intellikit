/******************************************************************************
 * MIT License
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Proboscis HSA interceptor — loaded via HSA_TOOLS_LIB.
 *
 * Hooks:
 *   - hsa_queue_create               -> creates intercept queues
 *   - hsa_executable_symbol_get_info  -> captures kernel_object -> symbol
 *   - hsa_executable_get_symbol_by_name -> captures symbol -> name
 *   - OnSubmitPackets                 -> records dispatch metadata
 ******************************************************************************/

#include "proboscis/interceptor.h"
#include "proboscis/proboscis.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cxxabi.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

// ---- Declarations from hsa_api_trace.h ------------------------------------
// The installed hsa_api_trace.h has broken relative includes in ROCm 6.x/7.x.
// We reproduce the minimal declarations needed.

typedef void (*hsa_amd_queue_intercept_handler)(
    const void* pkts, uint64_t pkt_count,
    uint64_t user_pkt_index, void* data,
    hsa_amd_queue_intercept_packet_writer writer);

hsa_status_t hsa_amd_queue_intercept_register(
    hsa_queue_t* queue, hsa_amd_queue_intercept_handler callback, void* user_data);

hsa_status_t hsa_amd_queue_intercept_create(
    hsa_agent_t agent_handle, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
    void* data, uint32_t private_segment_size,
    uint32_t group_segment_size, hsa_queue_t** queue);

// ---- Minimal HsaApiTable ABI-compatible definitions -----------------------
// Offsets verified against rocprofiler-sdk ROCP_SDK_ENFORCE_ABI.
// compute_table_offset(N) = N * sizeof(void*) + sizeof(size_t).
// Field N byte offset = N*8+8 on 64-bit. Pointer index from struct base = N+1.

struct ApiTableVersion {
    uint32_t major_id;
    uint32_t minor_id;
    uint32_t step_id;
    uint32_t reserved;
};

// We only define fields we access by name. Field 123 (hsa_executable_get_symbol_by_name)
// is accessed via core_table_field_ptr() helper to avoid a massive struct.
struct CoreApiTable {
    ApiTableVersion version;
    decltype(hsa_init)* hsa_init_fn;                                // field 1
    decltype(hsa_shut_down)* hsa_shut_down_fn;                      // 2
    decltype(hsa_system_get_info)* hsa_system_get_info_fn;          // 3
    void* _pad4;                                                    // 4
    void* _pad5;                                                    // 5
    decltype(hsa_iterate_agents)* hsa_iterate_agents_fn;            // 6
    decltype(hsa_agent_get_info)* hsa_agent_get_info_fn;            // 7
    decltype(hsa_queue_create)* hsa_queue_create_fn;                // 8
    decltype(hsa_soft_queue_create)* hsa_soft_queue_create_fn;      // 9
    decltype(hsa_queue_destroy)* hsa_queue_destroy_fn;              // 10
    void* _pad11_93[83];                                            // 11-93
    decltype(hsa_executable_symbol_get_info)* hsa_executable_symbol_get_info_fn; // 94
    // Fields 95+ accessed via core_table_field_ptr() below.
};

struct AmdExtTable {
    ApiTableVersion version;
    void* _pad[37];                                                 // fields 1-37
    decltype(hsa_amd_queue_intercept_create)* hsa_amd_queue_intercept_create_fn;   // 38
    decltype(hsa_amd_queue_intercept_register)* hsa_amd_queue_intercept_register_fn; // 39
};

struct HsaApiTable {
    ApiTableVersion version;
    CoreApiTable* core_;
    AmdExtTable* amd_ext_;
};

// Access CoreApiTable fields beyond what the struct defines.
// Field N is at pointer index (N+1) from the struct base.
static inline void** core_field(CoreApiTable* t, size_t n) {
    return reinterpret_cast<void**>(t) + n + 1;
}
static constexpr size_t FIELD_GET_SYMBOL_BY_NAME = 123;

// ---- Saved original HSA function pointers ---------------------------------

static decltype(hsa_queue_create)* orig_hsa_queue_create = nullptr;
static decltype(hsa_queue_destroy)* orig_hsa_queue_destroy = nullptr;
static decltype(hsa_amd_queue_intercept_create)* orig_intercept_create = nullptr;
static decltype(hsa_amd_queue_intercept_register)* orig_intercept_register = nullptr;
static decltype(hsa_executable_symbol_get_info)* orig_symbol_get_info = nullptr;
static decltype(hsa_executable_get_symbol_by_name)* orig_get_symbol_by_name = nullptr;

// ---- Singleton ------------------------------------------------------------

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
    for (auto* buf : probe_buffers_) {
        if (buf) hsa_memory_free(buf);
    }
}

// ---- Config ---------------------------------------------------------------

void ProboscisInterceptor::loadConfig() {
    const char* config_path = std::getenv("PROBOSCIS_CONFIG");
    const char* results_path = std::getenv("PROBOSCIS_RESULTS");
    const char* log_level_str = std::getenv("PROBOSCIS_LOG_LEVEL");

    config_.log_level = log_level_str ? std::atoi(log_level_str) : 1;
    config_.results_path = results_path ? results_path : "/tmp/proboscis_results.json";

    if (config_path) {
        std::ifstream f(config_path);
        if (f.good()) {
            std::ostringstream ss;
            ss << f.rdbuf();
            std::string content = ss.str();

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
                std::cerr << "[proboscis] Config: probe_type=" << config_.probe_type
                          << " target=" << config_.target_kernel << std::endl;
            }
        }
    }
}

// ---- Kernel name resolution -----------------------------------------------

static std::string demangle(const char* mangled) {
    int status = 0;
    char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
    if (status == 0 && demangled) {
        std::string result(demangled);
        std::free(demangled);
        auto pos = result.find(" [clone");
        if (pos != std::string::npos) result.resize(pos);
        return result;
    }
    return mangled;
}

std::string ProboscisInterceptor::getKernelName(uint64_t kernel_object) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto h = handles_symbols_.find(kernel_object);
    if (h == handles_symbols_.end()) {
        std::ostringstream s;
        s << "kernel_0x" << std::hex << kernel_object;
        return s.str();
    }

    auto n = symbols_names_.find(h->second);
    if (n == symbols_names_.end()) {
        std::ostringstream s;
        s << "kernel_0x" << std::hex << kernel_object;
        return s.str();
    }

    return demangle(n->second.c_str());
}

bool ProboscisInterceptor::shouldInstrument(const std::string& kernel_name) const {
    if (config_.target_kernel.empty()) return true;
    return kernel_name.find(config_.target_kernel) != std::string::npos;
}

// ---- Queue management -----------------------------------------------------

void ProboscisInterceptor::addQueue(hsa_queue_t* queue, hsa_agent_t agent) {
    std::lock_guard<std::mutex> lock(mutex_);
    queues_[queue] = agent;
}

// ---- Dispatch interception ------------------------------------------------

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
    for (uint64_t i = 0; i < count; i++) {
        const auto& pkt = packets[i];
        uint8_t pkt_type = pkt.header & 0xFF;
        if (pkt_type != HSA_PACKET_TYPE_KERNEL_DISPATCH) continue;

        std::string name = getKernelName(pkt.kernel_object);
        if (!shouldInstrument(name)) continue;

        DispatchRecord rec{};
        rec.kernel_object = pkt.kernel_object;
        rec.grid_size[0] = pkt.grid_size_x;
        rec.grid_size[1] = pkt.grid_size_y;
        rec.grid_size[2] = pkt.grid_size_z;
        rec.workgroup_size[0] = pkt.workgroup_size_x;
        rec.workgroup_size[1] = pkt.workgroup_size_y;
        rec.workgroup_size[2] = pkt.workgroup_size_z;
        rec.kernarg_address = reinterpret_cast<uint64_t>(pkt.kernarg_address);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            dispatch_records_[name].push_back(rec);
        }

        if (config_.log_level >= 2) {
            std::cerr << "[proboscis] Dispatch: " << name
                      << " grid=[" << pkt.grid_size_x << "," << pkt.grid_size_y
                      << "," << pkt.grid_size_z << "]"
                      << " wg=[" << pkt.workgroup_size_x << "," << pkt.workgroup_size_y
                      << "," << pkt.workgroup_size_z << "]" << std::endl;
        }
    }

    writer(packets, count);
}

// ---- Probe buffer management ----------------------------------------------

void* ProboscisInterceptor::allocateProbeBuffer(hsa_agent_t agent, size_t size) {
    void* ptr = nullptr;
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

// ---- Results output -------------------------------------------------------

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            default:   out += c;
        }
    }
    return out;
}

void ProboscisInterceptor::writeResults() {
    if (config_.results_path.empty()) return;

    std::ofstream f(config_.results_path);
    if (!f.good()) {
        std::cerr << "[proboscis] Failed to write results to "
                  << config_.results_path << std::endl;
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    f << "{\"kernels\": {";
    bool first_kernel = true;
    for (const auto& kv : dispatch_records_) {
        const auto& kernel_name = kv.first;
        const auto& records = kv.second;

        if (!first_kernel) f << ", ";
        first_kernel = false;

        uint64_t total_threads = 0;
        f << "\"" << json_escape(kernel_name) << "\": {\"records\": [";

        for (size_t j = 0; j < records.size(); j++) {
            const auto& rec = records[j];
            uint64_t threads = (uint64_t)rec.grid_size[0] * rec.grid_size[1] * rec.grid_size[2];
            total_threads += threads;

            if (j > 0) f << ", ";
            f << "{\"dispatch_id\": " << j
              << ", \"grid_size\": [" << rec.grid_size[0] << ", " << rec.grid_size[1]
              << ", " << rec.grid_size[2] << "]"
              << ", \"workgroup_size\": [" << rec.workgroup_size[0] << ", "
              << rec.workgroup_size[1] << ", " << rec.workgroup_size[2] << "]"
              << ", \"threads\": " << threads << "}";
        }

        f << "], \"summary\": {\"dispatch_count\": " << records.size()
          << ", \"total_threads\": " << total_threads << "}}";
    }
    f << "}}" << std::endl;

    if (config_.log_level >= 1) {
        size_t total = 0;
        for (const auto& kv : dispatch_records_) total += kv.second.size();
        std::cerr << "[proboscis] Results: " << dispatch_records_.size()
                  << " kernels, " << total << " dispatches -> "
                  << config_.results_path << std::endl;
    }
}

// ---- HSA Hooks ------------------------------------------------------------

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
        if (self->config_.log_level >= 2)
            std::cerr << "[proboscis] Intercepted queue creation" << std::endl;
    }
    return result;
}

hsa_status_t ProboscisInterceptor::hsa_executable_symbol_get_info(
    hsa_executable_symbol_t executable_symbol,
    hsa_executable_symbol_info_t attribute,
    void* value)
{
    hsa_status_t result = orig_symbol_get_info(executable_symbol, attribute, value);
    if (result == HSA_STATUS_SUCCESS &&
        attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT) {
        auto* self = getInstance();
        uint64_t ko = *static_cast<uint64_t*>(value);
        std::lock_guard<std::mutex> lock(self->mutex_);
        self->handles_symbols_[ko] = executable_symbol;
    }
    return result;
}

hsa_status_t ProboscisInterceptor::hsa_executable_get_symbol_by_name(
    hsa_executable_t executable, const char* symbol_name,
    const hsa_agent_t* agent, hsa_executable_symbol_t* symbol)
{
    hsa_status_t result = orig_get_symbol_by_name(executable, symbol_name, agent, symbol);
    if (result == HSA_STATUS_SUCCESS && symbol_name && symbol) {
        auto* self = getInstance();
        std::lock_guard<std::mutex> lock(self->mutex_);
        self->symbols_names_[*symbol] = std::string(symbol_name);
    }
    return result;
}

// ---- HSA_TOOLS_LIB Entry Points ------------------------------------------

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

    // Save originals
    orig_hsa_queue_create = table->core_->hsa_queue_create_fn;
    orig_hsa_queue_destroy = table->core_->hsa_queue_destroy_fn;
    orig_intercept_create = table->amd_ext_->hsa_amd_queue_intercept_create_fn;
    orig_intercept_register = table->amd_ext_->hsa_amd_queue_intercept_register_fn;
    orig_symbol_get_info = table->core_->hsa_executable_symbol_get_info_fn;
    orig_get_symbol_by_name = reinterpret_cast<decltype(orig_get_symbol_by_name)>(
        *core_field(table->core_, FIELD_GET_SYMBOL_BY_NAME));

    // Install hooks
    table->core_->hsa_queue_create_fn = ProboscisInterceptor::hsa_queue_create;
    table->core_->hsa_executable_symbol_get_info_fn =
        ProboscisInterceptor::hsa_executable_symbol_get_info;
    *core_field(table->core_, FIELD_GET_SYMBOL_BY_NAME) =
        reinterpret_cast<void*>(ProboscisInterceptor::hsa_executable_get_symbol_by_name);

    ProboscisInterceptor::getInstance();
    std::cerr << "[proboscis] Tool loaded" << std::endl;
    return true;
}

PUBLIC_API void OnUnload() {
    ProboscisInterceptor::cleanup();
    std::cerr << "[proboscis] Tool unloaded" << std::endl;
}

} // extern "C"
