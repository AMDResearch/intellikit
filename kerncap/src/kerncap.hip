/**
 * kerncap.hip — HSA interception tool for kernel argument capture
 *
 * Loaded via LD_PRELOAD (rocprofiler-sdk registration). Intercepts kernel
 * dispatches, captures grid/block dimensions, kernel arguments (scalars and
 * device buffers), and writes them to an output directory for reproducer
 * generation.
 *
 * Adapted from Accordo (AMD Research, MIT License).
 */

#define __HIP_PLATFORM_AMD__

#include "kerncap.hpp"

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/intercept_table.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cxxabi.h>
#include <fcntl.h>
#include <iomanip>
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

// SHA-256 implementation (compact public-domain).  We avoid pulling in a
// full crypto dependency for what is essentially a content-hash lookup.
namespace {

struct Sha256 {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t buf[64];
    uint32_t buflen;

    static constexpr uint32_t K[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
    };

    static uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

    void init() {
        static const uint32_t H0[8] = {
            0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
            0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19,
        };
        std::memcpy(state, H0, sizeof(state));
        bitlen = 0;
        buflen = 0;
    }

    void transform(const uint8_t* d) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(d[4*i])   << 24) |
                   (uint32_t(d[4*i+1]) << 16) |
                   (uint32_t(d[4*i+2]) <<  8) |
                   (uint32_t(d[4*i+3])      );
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
            uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=state[0],b=state[1],c=state[2],d2=state[3],
                 e=state[4],f=state[5],g=state[6],h=state[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = h + S1 + ch + K[i] + w[i];
            uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
            uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + mj;
            h=g; g=f; f=e; e=d2+t1; d2=c; c=b; b=a; a=t1+t2;
        }
        state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d2;
        state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
    }

    void update(const uint8_t* data, size_t len) {
        for (size_t i = 0; i < len; ++i) {
            buf[buflen++] = data[i];
            if (buflen == 64) {
                transform(buf);
                bitlen += 512;
                buflen = 0;
            }
        }
    }

    std::string hex_finalize() {
        uint64_t total_bits = bitlen + uint64_t(buflen) * 8;
        buf[buflen++] = 0x80;
        if (buflen > 56) {
            while (buflen < 64) buf[buflen++] = 0;
            transform(buf);
            buflen = 0;
        }
        while (buflen < 56) buf[buflen++] = 0;
        for (int i = 7; i >= 0; --i) buf[buflen++] = uint8_t(total_bits >> (8 * i));
        transform(buf);
        std::ostringstream os;
        os << std::hex << std::setfill('0');
        for (int i = 0; i < 8; ++i) os << std::setw(8) << state[i];
        return os.str();
    }
};

std::string sha256_hex(const void* data, size_t bytes) {
    Sha256 h;
    h.init();
    h.update(static_cast<const uint8_t*>(data), bytes);
    return h.hex_finalize();
}

}  // namespace

namespace kerncap {

// ---------------------------------------------------------------------------
// Static members
// ---------------------------------------------------------------------------
CaptureState* CaptureState::singleton_{nullptr};
std::shared_mutex CaptureState::mutex_{};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::string demangle(const char* mangled) {
    int status = 0;
    std::unique_ptr<char, void(*)(void*)> result(
        abi::__cxa_demangle(mangled, nullptr, nullptr, &status), std::free);
    return (status == 0) ? result.get() : mangled;
}

static const char* safe_getenv(const char* name, const char* fallback = "") {
    const char* val = std::getenv(name);
    return val ? val : fallback;
}

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------
CaptureState* CaptureState::get_instance(
        HsaApiTable* table, uint64_t runtime_version,
        uint64_t failed_tool_count, const char* const* failed_tool_names) {
    const std::lock_guard lock(mutex_);
    if (!singleton_ && table) {
        singleton_ = new CaptureState(
            table, runtime_version, failed_tool_count, failed_tool_names);
    }
    return singleton_;
}

// ---------------------------------------------------------------------------
// Constructor — save API, install hooks, read config
// ---------------------------------------------------------------------------
CaptureState::CaptureState(
        HsaApiTable* table, uint64_t /*runtime_version*/,
        uint64_t /*failed_tool_count*/, const char* const* /*failed_tool_names*/)
    : api_table_{table} {

    // Read configuration from environment
    target_kernel_  = safe_getenv("KERNCAP_KERNEL");
    output_dir_     = safe_getenv("KERNCAP_OUTPUT", "/tmp/kerncap_output");

    const char* disp_env = std::getenv("KERNCAP_DISPATCH");
    target_dispatch_ = disp_env ? std::atoi(disp_env) : -1;
    const char* trace_env = std::getenv("KERNCAP_TRACE_DISPATCHES");
    trace_dispatch_limit_ = trace_env ? std::strtoull(trace_env, nullptr, 10) : 0;
    const char* trace_per_queue_env = std::getenv("KERNCAP_TRACE_DISPATCHES_PER_QUEUE");
    trace_dispatch_per_queue_limit_ =
        trace_per_queue_env ? std::strtoull(trace_per_queue_env, nullptr, 10) : 0;
    intercept_first_queue_only_ =
        (std::getenv("KERNCAP_INTERCEPT_FIRST_QUEUE_ONLY") != nullptr);
    bypass_after_capture_ =
        (std::getenv("KERNCAP_BYPASS_AFTER_CAPTURE") != nullptr);
    disable_code_object_hooks_ =
        (std::getenv("KERNCAP_DISABLE_CODE_OBJECT_HOOKS") != nullptr);

    triton_name_map_path_ = safe_getenv("KERNCAP_TRITON_NAME_MAP");
    if (!triton_name_map_path_.empty()) {
        KERNCAP_INFO("Triton HSA capture mode: name_map=%s",
                     triton_name_map_path_.c_str());
    }

    // Fork safety: record PID and check for multi-process mode
    initial_pid_ = getpid();
    capture_child_mode_ = (std::getenv("KERNCAP_CAPTURE_CHILD") != nullptr);

    if (capture_child_mode_) {
        if (pthread_atfork(nullptr, CaptureState::atfork_parent_handler, nullptr) != 0) {
            KERNCAP_WARN("pthread_atfork failed — disabling capture_child_mode");
            capture_child_mode_ = false;
        }
    }

    KERNCAP_INFO("Process pid=%d, capture_child_mode=%s",
                 initial_pid_, capture_child_mode_ ? "true" : "false");

    if (target_kernel_.empty()) {
        KERNCAP_WARN("KERNCAP_KERNEL not set — will not capture any kernel");
    } else {
        KERNCAP_INFO("Target kernel: '%s', dispatch: %s, output: %s",
                     target_kernel_.c_str(),
                     target_dispatch_ < 0 ? "first match" : std::to_string(target_dispatch_).c_str(),
                     output_dir_.c_str());
    }
    if (trace_dispatch_limit_ > 0) {
        KERNCAP_INFO("Dispatch tracing enabled for first %" PRIu64 " dispatches",
                     trace_dispatch_limit_);
    }
    if (trace_dispatch_per_queue_limit_ > 0) {
        KERNCAP_INFO("Per-queue dispatch tracing enabled for first %" PRIu64
                     " dispatches on each queue",
                     trace_dispatch_per_queue_limit_);
    }
    if (intercept_first_queue_only_) {
        KERNCAP_INFO("Queue policy: intercept first queue only");
    }
    if (bypass_after_capture_) {
        KERNCAP_INFO("Packet policy: bypass all interception after first capture");
    }
    if (disable_code_object_hooks_) {
        KERNCAP_INFO("Code object policy: disable code object interception hooks");
    }

    // Create output directory
    mkdir(output_dir_.c_str(), 0755);

    save_hsa_api();
    hook_api();
}

CaptureState::~CaptureState() {
    delete saved_api_.core_;
    delete saved_api_.amd_ext_;
    delete saved_api_.finalizer_ext_;
    delete saved_api_.image_ext_;
}

// ---------------------------------------------------------------------------
// Fork safety — multi-process support
// ---------------------------------------------------------------------------
void CaptureState::atfork_parent_handler() {
    auto* inst = get_instance();
    if (inst) {
        inst->fork_detected_.store(true, std::memory_order_relaxed);
        KERNCAP_INFO("fork() detected in parent (pid=%d) — "
                     "parent will become passive, child will capture",
                     getpid());
    }
}

void CaptureState::reset_inherited_state() {
    if (child_state_reset_.exchange(true, std::memory_order_relaxed)) return;

    KERNCAP_INFO("Child process (pid=%d) resetting inherited state "
                 "from parent (pid=%d)",
                 getpid(), initial_pid_);

    pointer_sizes_.clear();
    vmem_tracked_.clear();
    symbol_names_.clear();
    handle_to_symbol_.clear();
    symbol_to_executable_.clear();
    pending_reader_blobs_.clear();
    executable_blobs_.clear();
    kernel_hsaco_.clear();
    executable_kernargs_.clear();
    executable_blob_sha256_.clear();
    queue_agents_.clear();
    queue_dispatch_seen_.clear();
    target_dispatch_count_ = 0;
    dispatch_seen_count_.store(0, std::memory_order_relaxed);
    captured_.store(false, std::memory_order_relaxed);
    gpu_arch_.clear();
}

bool CaptureState::is_active_process() {
    if (!capture_child_mode_) return true;
    pid_t current = getpid();
    if (current != initial_pid_) {
        reset_inherited_state();
        return true;
    }
    return !fork_detected_.load(std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// API table save / hook
// ---------------------------------------------------------------------------
void CaptureState::save_hsa_api() {
    saved_api_.core_          = new CoreApiTable();
    saved_api_.amd_ext_       = new AmdExtTable();
    saved_api_.finalizer_ext_ = new FinalizerExtTable();
    saved_api_.image_ext_     = new ImageExtTable();

    std::memcpy(saved_api_.core_,          api_table_->core_,          sizeof(CoreApiTable));
    std::memcpy(saved_api_.amd_ext_,       api_table_->amd_ext_,       sizeof(AmdExtTable));
    std::memcpy(saved_api_.finalizer_ext_, api_table_->finalizer_ext_, sizeof(FinalizerExtTable));
    std::memcpy(saved_api_.image_ext_,     api_table_->image_ext_,     sizeof(ImageExtTable));
}

void CaptureState::hook_api() {
    api_table_->core_->hsa_queue_create_fn =
        CaptureState::hsa_queue_create;
    api_table_->core_->hsa_queue_destroy_fn =
        CaptureState::hsa_queue_destroy;
    api_table_->amd_ext_->hsa_amd_memory_pool_allocate_fn =
        CaptureState::hsa_amd_memory_pool_allocate;
    api_table_->amd_ext_->hsa_amd_memory_pool_free_fn =
        CaptureState::hsa_amd_memory_pool_free;
    api_table_->core_->hsa_memory_allocate_fn =
        CaptureState::hsa_memory_allocate;
    api_table_->core_->hsa_executable_get_symbol_by_name_fn =
        CaptureState::hsa_executable_get_symbol_by_name;
    api_table_->core_->hsa_executable_symbol_get_info_fn =
        CaptureState::hsa_executable_symbol_get_info;

    // VMEM tracking hooks
    api_table_->amd_ext_->hsa_amd_vmem_address_reserve_fn =
        CaptureState::hsa_amd_vmem_address_reserve;
    api_table_->amd_ext_->hsa_amd_vmem_address_free_fn =
        CaptureState::hsa_amd_vmem_address_free;
    api_table_->amd_ext_->hsa_amd_vmem_map_fn =
        CaptureState::hsa_amd_vmem_map;
    api_table_->amd_ext_->hsa_amd_vmem_unmap_fn =
        CaptureState::hsa_amd_vmem_unmap;

    // Code object interception for .hsaco capture
    if (!disable_code_object_hooks_) {
        api_table_->core_->hsa_code_object_reader_create_from_memory_fn =
            CaptureState::hsa_code_object_reader_create_from_memory;
        api_table_->core_->hsa_executable_load_agent_code_object_fn =
            CaptureState::hsa_executable_load_agent_code_object;
    }
}

// ---------------------------------------------------------------------------
// Symbol tracking hooks
// ---------------------------------------------------------------------------
hsa_status_t CaptureState::hsa_executable_get_symbol_by_name(
        hsa_executable_t executable, const char* symbol_name,
        const hsa_agent_t* agent, hsa_executable_symbol_t* symbol) {
    auto* inst = get_instance();
    if (!inst->is_active_process()) {
        return hsa_core_call(inst, hsa_executable_get_symbol_by_name,
                             executable, symbol_name, agent, symbol);
    }
    auto result = hsa_core_call(inst, hsa_executable_get_symbol_by_name,
                                executable, symbol_name, agent, symbol);
    if (result == HSA_STATUS_SUCCESS) {
        std::lock_guard g(mutex_);
        inst->symbol_names_[*symbol] = std::string(symbol_name);
        inst->symbol_to_executable_[*symbol] = executable.handle;
    }
    return result;
}

hsa_status_t CaptureState::hsa_executable_symbol_get_info(
        hsa_executable_symbol_t symbol,
        hsa_executable_symbol_info_t attribute, void* value) {
    auto* inst = get_instance();
    if (!inst->is_active_process()) {
        return hsa_core_call(inst, hsa_executable_symbol_get_info,
                             symbol, attribute, value);
    }
    auto result = hsa_core_call(inst, hsa_executable_symbol_get_info,
                                symbol, attribute, value);
    if (result == HSA_STATUS_SUCCESS &&
        attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT) {
        uint64_t kernel_object = *static_cast<uint64_t*>(value);
        std::lock_guard g(mutex_);
        inst->handle_to_symbol_[kernel_object] = symbol;

        // Lazily associate kernel_object with its executable so we can
        // retrieve the .hsaco blob at capture time. We deliberately do NOT
        // copy the blob here: many kernels can share one executable, and
        // duplicating a multi-MB blob per kernel symbol blows up memory
        // (e.g. rocBLAS/Tensile loads ~15 MB executables containing
        // hundreds of kernels).
        auto exec_it = inst->symbol_to_executable_.find(symbol);
        if (exec_it != inst->symbol_to_executable_.end()) {
            std::lock_guard co_lock(inst->code_object_mutex_);
            inst->kernel_hsaco_[kernel_object] = exec_it->second;
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Code object interception (for .hsaco capture)
// ---------------------------------------------------------------------------
hsa_status_t CaptureState::hsa_code_object_reader_create_from_memory(
        const void* code_object, size_t size,
        hsa_code_object_reader_t* code_object_reader) {
    auto* inst = get_instance();
    if (!inst->is_active_process()) {
        return hsa_core_call(inst, hsa_code_object_reader_create_from_memory,
                             code_object, size, code_object_reader);
    }
    auto result = hsa_core_call(inst, hsa_code_object_reader_create_from_memory,
                                code_object, size, code_object_reader);
    if (result == HSA_STATUS_SUCCESS && code_object_reader) {
        std::lock_guard lock(inst->code_object_mutex_);
        auto* bytes = static_cast<const uint8_t*>(code_object);
        inst->pending_reader_blobs_[code_object_reader->handle] =
            std::vector<uint8_t>(bytes, bytes + size);
        KERNCAP_INFO("Code object reader: handle=0x%lx (%zu bytes)",
                     static_cast<unsigned long>(code_object_reader->handle), size);
    }
    return result;
}

hsa_status_t CaptureState::hsa_executable_load_agent_code_object(
        hsa_executable_t executable, hsa_agent_t agent,
        hsa_code_object_reader_t code_object_reader,
        const char* options, hsa_loaded_code_object_t* loaded_code_object) {
    auto* inst = get_instance();
    if (!inst->is_active_process()) {
        return hsa_core_call(inst, hsa_executable_load_agent_code_object,
                             executable, agent, code_object_reader,
                             options, loaded_code_object);
    }
    auto result = hsa_core_call(inst, hsa_executable_load_agent_code_object,
                                executable, agent, code_object_reader,
                                options, loaded_code_object);
    if (result == HSA_STATUS_SUCCESS) {
        std::lock_guard lock(inst->code_object_mutex_);

        // Store blob keyed by executable handle. The kernel_object -> blob
        // association is built lazily in hsa_executable_symbol_get_info
        // when the runtime queries KERNEL_OBJECT, avoiding a deadlock
        // from calling hsa_executable_iterate_symbols while the runtime
        // holds its internal executable lock.
        auto it = inst->pending_reader_blobs_.find(code_object_reader.handle);
        if (it != inst->pending_reader_blobs_.end()) {
            const auto& blob = it->second;

            // Parse kernarg metadata up front (cheap; ~ms per HSACO) so
            // dispatch-time has zero comgr cost in the hot path.
            auto kernels = parse_kernarg_metadata(blob.data(), blob.size());
            if (!kernels.empty()) {
                KERNCAP_INFO(
                    "Parsed %zu kernel(s) from code object on executable 0x%lx",
                    kernels.size(), executable.handle);
                inst->executable_kernargs_[executable.handle] = std::move(kernels);
            }

            inst->executable_blob_sha256_[executable.handle] =
                sha256_hex(blob.data(), blob.size());
            inst->executable_blobs_[executable.handle] = std::move(it->second);
            inst->pending_reader_blobs_.erase(it);
            KERNCAP_INFO("Stored code object blob for executable 0x%lx (sha256=%.16s...)",
                         executable.handle,
                         inst->executable_blob_sha256_[executable.handle].c_str());
            // Refresh the Triton name map so the user-name lookup can hit.
            inst->maybe_reload_triton_name_map();
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Triton name_map.json — populated by the compile-hook wrapper, consumed
// here at dispatch time to resolve loaded-blob SHA-256 -> user name.
// ---------------------------------------------------------------------------
static int64_t mtime_ns(const std::string& path) {
    struct stat st{};
    if (stat(path.c_str(), &st) != 0) return 0;
    return int64_t(st.st_mtim.tv_sec) * 1000000000 + int64_t(st.st_mtim.tv_nsec);
}

void CaptureState::maybe_reload_triton_name_map() {
    std::lock_guard lock(triton_name_map_mutex_);
    if (triton_name_map_path_.empty()) return;
    auto cur = mtime_ns(triton_name_map_path_);
    if (cur == 0 || cur == triton_name_map_mtime_ns_) return;

    std::ifstream f(triton_name_map_path_);
    if (!f) return;
    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        KERNCAP_WARN("name_map.json parse error: %s", e.what());
        return;
    }
    triton_name_map_.clear();
    if (!j.is_array()) return;
    for (const auto& row : j) {
        TritonNameMapRow r;
        r.user_name    = row.value("user_name", "");
        r.hsaco_sha256 = row.value("hsaco_sha256", "");
        r.hsaco_path   = row.value("hsaco_path", "");
        if (row.contains("constexpr_values")) r.constexpr_values = row["constexpr_values"];
        triton_name_map_.push_back(std::move(r));
    }
    triton_name_map_mtime_ns_ = cur;
    KERNCAP_INFO("Reloaded triton name_map (%zu rows) from %s",
                 triton_name_map_.size(), triton_name_map_path_.c_str());
}

std::optional<CaptureState::TritonNameMapRow>
CaptureState::lookup_triton_user_name(const std::string& blob_sha256) {
    std::lock_guard lock(triton_name_map_mutex_);
    for (const auto& r : triton_name_map_) {
        if (r.hsaco_sha256 == blob_sha256) return r;
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// Memory allocation tracking
// ---------------------------------------------------------------------------
hsa_status_t CaptureState::hsa_amd_memory_pool_allocate(
        hsa_amd_memory_pool_t pool, size_t size, uint32_t flags, void** ptr) {
    auto* inst = get_instance();
    if (!inst->is_active_process()) {
        return hsa_ext_call(inst, hsa_amd_memory_pool_allocate,
                            pool, size, flags, ptr);
    }
    auto result = hsa_ext_call(inst, hsa_amd_memory_pool_allocate,
                               pool, size, flags, ptr);
    if (result == HSA_STATUS_SUCCESS && *ptr) {
        std::lock_guard lock(inst->ptr_mutex_);
        inst->pointer_sizes_[*ptr] = size;
    }
    return result;
}

hsa_status_t CaptureState::hsa_amd_memory_pool_free(void* ptr) {
    auto* inst = get_instance();
    if (!inst->is_active_process()) {
        return hsa_ext_call(inst, hsa_amd_memory_pool_free, ptr);
    }
    auto result = hsa_ext_call(inst, hsa_amd_memory_pool_free, ptr);
    if (result == HSA_STATUS_SUCCESS && ptr) {
        std::lock_guard lock(inst->ptr_mutex_);
        inst->pointer_sizes_.erase(ptr);
    }
    return result;
}

hsa_status_t CaptureState::hsa_memory_allocate(
        hsa_region_t region, size_t size, void** ptr) {
    auto* inst = get_instance();
    if (!inst->is_active_process()) {
        return hsa_core_call(inst, hsa_memory_allocate, region, size, ptr);
    }
    auto result = hsa_core_call(inst, hsa_memory_allocate, region, size, ptr);
    if (result == HSA_STATUS_SUCCESS && *ptr) {
        std::lock_guard lock(inst->ptr_mutex_);
        inst->pointer_sizes_[*ptr] = size;
    }
    return result;
}

// ---------------------------------------------------------------------------
// VMEM tracking hooks
// ---------------------------------------------------------------------------
hsa_status_t CaptureState::hsa_amd_vmem_address_reserve(
        void** va, size_t size, uint64_t address, uint64_t flags) {
    auto* inst = get_instance();
    if (!inst->is_active_process()) {
        return hsa_ext_call(inst, hsa_amd_vmem_address_reserve,
                            va, size, address, flags);
    }
    auto result = hsa_ext_call(inst, hsa_amd_vmem_address_reserve,
                               va, size, address, flags);
    if (result == HSA_STATUS_SUCCESS && va && *va) {
        std::lock_guard lock(inst->ptr_mutex_);
        inst->pointer_sizes_[*va] = size;
        inst->vmem_tracked_.insert(*va);
    }
    return result;
}

hsa_status_t CaptureState::hsa_amd_vmem_address_free(
        void* va, size_t size) {
    auto* inst = get_instance();
    if (!inst->is_active_process()) {
        return hsa_ext_call(inst, hsa_amd_vmem_address_free, va, size);
    }
    auto result = hsa_ext_call(inst, hsa_amd_vmem_address_free, va, size);
    if (result == HSA_STATUS_SUCCESS && va) {
        std::lock_guard lock(inst->ptr_mutex_);
        inst->pointer_sizes_.erase(va);
        inst->vmem_tracked_.erase(va);
    }
    return result;
}

hsa_status_t CaptureState::hsa_amd_vmem_map(
        void* va, size_t size, size_t in_offset,
        hsa_amd_vmem_alloc_handle_t memory_handle, uint64_t flags) {
    auto* inst = get_instance();
    if (!inst->is_active_process()) {
        return hsa_ext_call(inst, hsa_amd_vmem_map,
                            va, size, in_offset, memory_handle, flags);
    }
    auto result = hsa_ext_call(inst, hsa_amd_vmem_map,
                               va, size, in_offset, memory_handle, flags);
    if (result == HSA_STATUS_SUCCESS && va) {
        std::lock_guard lock(inst->ptr_mutex_);
        inst->pointer_sizes_[va] = size;
        inst->vmem_tracked_.insert(va);
    }
    return result;
}

hsa_status_t CaptureState::hsa_amd_vmem_unmap(
        void* va, size_t size) {
    auto* inst = get_instance();
    if (!inst->is_active_process()) {
        return hsa_ext_call(inst, hsa_amd_vmem_unmap, va, size);
    }
    auto result = hsa_ext_call(inst, hsa_amd_vmem_unmap, va, size);
    if (result == HSA_STATUS_SUCCESS && va) {
        std::lock_guard lock(inst->ptr_mutex_);
        inst->pointer_sizes_.erase(va);
        inst->vmem_tracked_.erase(va);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Queue interception
// ---------------------------------------------------------------------------
hsa_status_t CaptureState::hsa_queue_create(
        hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
        void (*callback)(hsa_status_t, hsa_queue_t*, void*),
        void* data, uint32_t private_segment_size,
        uint32_t group_segment_size, hsa_queue_t** queue) {
    auto* inst = get_instance();
    bool intercept_queue = true;

    if (!inst->is_active_process()) {
        KERNCAP_INFO("Passive mode (pid=%d): passing through hsa_queue_create",
                     getpid());
        return hsa_core_call(inst, hsa_queue_create, agent, size, type,
                             callback, data, private_segment_size,
                             group_segment_size, queue);
    }

    if (inst->intercept_first_queue_only_) {
        std::shared_lock g(mutex_);
        intercept_queue = inst->queue_agents_.empty();
    }
    if (!intercept_queue) {
        KERNCAP_INFO("First-queue-only mode (pid=%d): passing through hsa_queue_create",
                     getpid());
        return hsa_core_call(inst, hsa_queue_create, agent, size, type,
                             callback, data, private_segment_size,
                             group_segment_size, queue);
    }

    KERNCAP_INFO("Active mode (pid=%d): intercepting hsa_queue_create "
                 "(size=%u, type=%u)", getpid(), size, type);

    auto result = hsa_ext_call(inst, hsa_amd_queue_intercept_create,
                               agent, size, type, callback, data,
                               private_segment_size, group_segment_size, queue);
    if (result == HSA_STATUS_SUCCESS) {
        KERNCAP_INFO("hsa_amd_queue_intercept_create success, queue=0x%lx", (*queue)->id);
        {
            std::lock_guard g(mutex_);
            inst->queue_agents_[*queue] = agent;
        }
        result = hsa_ext_call(inst, hsa_amd_queue_intercept_register,
                              *queue, CaptureState::on_submit_packet,
                              reinterpret_cast<void*>(*queue));
        if (result != HSA_STATUS_SUCCESS) {
            KERNCAP_ERROR("Failed to register intercept callback: %d",
                         static_cast<int>(result));
        } else {
            KERNCAP_INFO("hsa_amd_queue_intercept_register success");
        }
    } else {
        KERNCAP_ERROR("hsa_amd_queue_intercept_create failed: %d", static_cast<int>(result));
    }
    return result;
}

hsa_status_t CaptureState::hsa_queue_destroy(hsa_queue_t* queue) {
    auto* inst = get_instance();
    if (!inst->is_active_process()) {
        return hsa_core_call(inst, hsa_queue_destroy, queue);
    }
    {
        std::lock_guard g(mutex_);
        inst->queue_agents_.erase(queue);
    }
    return hsa_core_call(inst, hsa_queue_destroy, queue);
}

// ---------------------------------------------------------------------------
// Kernel name resolution
// ---------------------------------------------------------------------------
std::string CaptureState::get_kernel_name(uint64_t kernel_object) {
    std::shared_lock g(mutex_);
    auto hit = handle_to_symbol_.find(kernel_object);
    if (hit == handle_to_symbol_.end()) return "<unknown>";
    auto sit = symbol_names_.find(hit->second);
    if (sit == symbol_names_.end()) return "<unknown>";
    return demangle(sit->second.c_str());
}

std::string CaptureState::get_kernel_symbol(uint64_t kernel_object) {
    std::shared_lock g(mutex_);
    auto hit = handle_to_symbol_.find(kernel_object);
    if (hit == handle_to_symbol_.end()) return "";
    auto sit = symbol_names_.find(hit->second);
    if (sit == symbol_names_.end()) return "";
    // Strip the ".kd" (kernel descriptor) suffix that HSA uses internally;
    // hipModuleGetFunction expects the bare mangled name.
    std::string sym = sit->second;
    if (sym.size() > 3 && sym.compare(sym.size() - 3, 3, ".kd") == 0)
        sym.resize(sym.size() - 3);
    return sym;
}

bool CaptureState::should_trace_dispatch(
        uint64_t queue_id,
        uint64_t dispatch_seq,
        uint64_t queue_dispatch_seq,
        bool is_target) const {
    if (is_target) return true;
    if (trace_dispatch_limit_ > 0 && dispatch_seq < trace_dispatch_limit_) {
        return true;
    }
    if (trace_dispatch_per_queue_limit_ > 0 &&
        queue_id != 0 &&
        queue_dispatch_seq < trace_dispatch_per_queue_limit_) {
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Capture kernel dispatch data
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Full device memory snapshot for VA-faithful replay
// ---------------------------------------------------------------------------
void CaptureState::snapshot_all_tracked_memory(
        const hsa_kernel_dispatch_packet_t* disp) {
    std::string memdir = output_dir_ + "/memory";
    mkdir(memdir.c_str(), 0755);

    struct TrackedAlloc {
        void* ptr;
        size_t size;
        bool is_vmem;
    };
    std::vector<TrackedAlloc> allocs;
    {
        std::lock_guard lock(ptr_mutex_);
        allocs.reserve(pointer_sizes_.size());
        for (const auto& [ptr, sz] : pointer_sizes_) {
            allocs.push_back({ptr, sz, vmem_tracked_.count(ptr) > 0});
        }
    }

    // Build the regions list and write memory_regions.json FIRST
    // (before any hsa_memory_copy that might crash on freed regions).
    nlohmann::json regions_json = nlohmann::json::array();

    for (const auto& a : allocs) {
        if (a.size == 0 || a.size > (1ULL << 40)) continue;

        uint64_t base = reinterpret_cast<uint64_t>(a.ptr);

        bool contains_kernarg = false;
        uint64_t ka = reinterpret_cast<uint64_t>(disp->kernarg_address);
        if (ka >= base && ka < base + a.size) {
            contains_kernarg = true;
        }

        regions_json.push_back({
            {"base", base},
            {"size", a.size},
            {"is_pool", !a.is_vmem},
            {"is_vmem", a.is_vmem},
            {"contains_kernarg", contains_kernarg},
            {"handle", 0},
            {"access", 0},
        });
    }

    {
        nlohmann::json regions_doc;
        regions_doc["regions"] = regions_json;

        std::string regions_path = output_dir_ + "/memory_regions.json";
        std::ofstream rout(regions_path);
        rout << regions_doc.dump(2);
        rout.close();
    }

    // Now do the D2H copies. If hsa_memory_copy crashes on a freed
    // region, at least memory_regions.json and dispatch.json are safe.
    //
    // Stream each region in fixed-size chunks via a single reusable
    // staging buffer so kerncap's host-RAM footprint is bounded by the
    // chunk size (default 64 MiB) regardless of how large any
    // individual GPU allocation is.  Without this, a multi-GiB region
    // (e.g. vLLM's KV-cache slabs) would force a full-region
    // ``std::vector<char>`` allocation and an equally large in-flight
    // D2H copy, easily pushing the host into OOM.
    //
    // Tunable via ``KERNCAP_SNAPSHOT_CHUNK_BYTES`` (bytes, min 64 KiB).
    constexpr size_t kDefaultChunkBytes = 64ULL << 20;  // 64 MiB
    constexpr size_t kMinChunkBytes     = 64ULL << 10;  // 64 KiB
    size_t chunk_bytes = kDefaultChunkBytes;
    if (const char* env = std::getenv("KERNCAP_SNAPSHOT_CHUNK_BYTES")) {
        try {
            long long v = std::stoll(env);
            if (v >= static_cast<long long>(kMinChunkBytes)) {
                chunk_bytes = static_cast<size_t>(v);
            } else {
                KERNCAP_WARN("KERNCAP_SNAPSHOT_CHUNK_BYTES=%s below minimum "
                             "(%zu); using default %zu",
                             env, kMinChunkBytes, kDefaultChunkBytes);
            }
        } catch (const std::exception&) {
            KERNCAP_WARN("KERNCAP_SNAPSHOT_CHUNK_BYTES=%s not parseable; "
                         "using default %zu", env, kDefaultChunkBytes);
        }
    }

    std::vector<char> buf(chunk_bytes);  // allocated ONCE, reused per region
    KERNCAP_INFO("Memory snapshot: streaming %zu region(s) in %zu-byte chunks",
                 allocs.size(), chunk_bytes);

    int ok_count = 0;
    for (const auto& a : allocs) {
        if (a.size == 0 || a.size > (1ULL << 40)) continue;

        uint64_t base = reinterpret_cast<uint64_t>(a.ptr);

        char filename[256];
        std::snprintf(filename, sizeof(filename),
                     "%s/region_%" PRIx64 ".bin", memdir.c_str(), base);

        std::ofstream out(filename, std::ios::binary);
        if (!out) {
            KERNCAP_WARN("snapshot: failed to open %s for writing", filename);
            continue;
        }

        bool region_ok = true;
        for (size_t off = 0; off < a.size; off += chunk_bytes) {
            size_t n = std::min(chunk_bytes, a.size - off);
            void* src = static_cast<char*>(a.ptr) + off;
            hsa_status_t st = hsa_memory_copy(buf.data(), src, n);
            if (st != HSA_STATUS_SUCCESS) {
                KERNCAP_WARN("snapshot: hsa_memory_copy failed for region 0x%"
                             PRIx64 " at offset %zu (chunk %zu bytes, err=%d)",
                             base, off, n, static_cast<int>(st));
                region_ok = false;
                break;
            }
            out.write(buf.data(), n);
            if (!out) {
                KERNCAP_WARN("snapshot: write to %s failed at offset %zu",
                             filename, off);
                region_ok = false;
                break;
            }
        }
        out.close();

        if (region_ok) {
            ok_count++;
        } else {
            // Avoid leaving truncated/garbage region files around.
            std::remove(filename);
        }
    }

    KERNCAP_INFO("Memory snapshot: %d/%zu regions captured",
                 ok_count, allocs.size());
}

// ---------------------------------------------------------------------------
// Module-variable snapshot (HSA_SYMBOL_KIND_VARIABLE) for VA-faithful replay
// ---------------------------------------------------------------------------
//
// Kernels launched via Kokkos's hip_parallel_launch_constant_memory<...>
// (used by LAMMPS TagPairEAMKernelC, SPARTA, ExaMPM, etc.) carry their View
// pointers and scalars in the HSACO module variable
// ``kokkos_impl_hip_constant_memory_buffer``, populated at runtime by
// ``hipMemcpyToSymbol``. The dynamic-allocation snapshot (above) does not
// see these bytes because they live in the executable's data segment, not
// a tracked allocation. Without restoring them at replay time, the kernel
// reads NULL ``View::m_data`` and faults at ``(nil)``.
//
// Iterate every observed executable's symbols, capture each variable's
// device bytes via the same chunked staging pattern as the memory
// snapshot, and write a ``module_variables.json`` manifest plus one
// ``module_variables/<exec_sha16>_<name>.bin`` blob per variable.
void CaptureState::snapshot_module_variables(
        const hsa_kernel_dispatch_packet_t* /*disp*/) {
    // Snapshot the (executable handle, sha256) pairs we currently know
    // about. Copy out under lock so the iteration runs without blocking
    // concurrent intercepted calls.
    std::vector<std::pair<uint64_t, std::string>> execs;
    {
        std::shared_lock g(mutex_);
        std::unordered_set<uint64_t> seen;
        for (const auto& [sym, exec_handle] : symbol_to_executable_) {
            (void)sym;
            if (!seen.insert(exec_handle).second) continue;
            std::string sha;
            auto sit = executable_blob_sha256_.find(exec_handle);
            if (sit != executable_blob_sha256_.end()) sha = sit->second;
            execs.push_back({exec_handle, std::move(sha)});
        }
    }

    if (execs.empty()) {
        KERNCAP_INFO("Module-variable snapshot: no executables observed; nothing to capture");
        return;
    }

    struct VarEntry {
        uint64_t exec_handle;
        std::string exec_sha256;   // first 16 hex chars are used in the blob filename
        std::string name;          // raw HSA symbol name (may include ':')
        std::string sanitized_name;
        uint64_t address;
        size_t size;
        std::string blob_relpath;  // module_variables/<exec16>_<name>.bin
    };

    std::vector<VarEntry> variables;

    auto sanitize = [](const std::string& s) {
        std::string out;
        out.reserve(s.size());
        for (char c : s) {
            if (c == ':' || c == '/' || c == '\\' || c == ' ') out += '_';
            else out += c;
        }
        return out;
    };

    // Pass 1: discover variable symbols, addresses, and sizes (no D2H yet).
    // Per-variable size cap of 1 GiB is applied inline in the callback below.
    for (const auto& [exec_handle, exec_sha] : execs) {
        struct IterCtx {
            std::vector<VarEntry>* out;
            uint64_t exec_handle;
            const std::string* exec_sha;
            decltype(sanitize)* sanitize_fn;
        } ctx{&variables, exec_handle, &exec_sha, &sanitize};

        hsa_executable_t exec{exec_handle};
        hsa_status_t it_st = hsa_executable_iterate_symbols(
            exec,
            [](hsa_executable_t /*e*/,
               hsa_executable_symbol_t sym,
               void* data) -> hsa_status_t {
                auto* c = static_cast<IterCtx*>(data);
                hsa_symbol_kind_t kind;
                if (hsa_executable_symbol_get_info(
                        sym, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind)
                    != HSA_STATUS_SUCCESS) {
                    return HSA_STATUS_SUCCESS;
                }
                if (kind != HSA_SYMBOL_KIND_VARIABLE) return HSA_STATUS_SUCCESS;

                uint32_t name_len = 0;
                if (hsa_executable_symbol_get_info(
                        sym, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &name_len)
                    != HSA_STATUS_SUCCESS || name_len == 0) {
                    return HSA_STATUS_SUCCESS;
                }
                std::string name(name_len, '\0');
                if (hsa_executable_symbol_get_info(
                        sym, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name.data())
                    != HSA_STATUS_SUCCESS) {
                    return HSA_STATUS_SUCCESS;
                }

                uint64_t addr = 0;
                size_t size = 0;
                if (hsa_executable_symbol_get_info(
                        sym, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &addr)
                    != HSA_STATUS_SUCCESS) {
                    return HSA_STATUS_SUCCESS;
                }
                if (hsa_executable_symbol_get_info(
                        sym, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE, &size)
                    != HSA_STATUS_SUCCESS) {
                    return HSA_STATUS_SUCCESS;
                }
                if (size == 0 || size > (1ULL << 30) || addr == 0) {
                    return HSA_STATUS_SUCCESS;
                }

                VarEntry e{};
                e.exec_handle = c->exec_handle;
                e.exec_sha256 = *c->exec_sha;
                e.name = name;
                e.sanitized_name = (*c->sanitize_fn)(name);
                e.address = addr;
                e.size = size;

                std::string sha16 = e.exec_sha256.substr(
                    0, std::min<size_t>(16, e.exec_sha256.size()));
                if (sha16.empty()) {
                    char buf[32];
                    std::snprintf(buf, sizeof(buf), "exec%lx", (unsigned long)c->exec_handle);
                    sha16 = buf;
                }
                e.blob_relpath =
                    std::string("module_variables/") + sha16 + "_" + e.sanitized_name + ".bin";

                c->out->push_back(std::move(e));
                return HSA_STATUS_SUCCESS;
            },
            &ctx);
        if (it_st != HSA_STATUS_SUCCESS) {
            KERNCAP_WARN("Module-variable snapshot: hsa_executable_iterate_symbols "
                         "failed for executable 0x%lx (err=%d)",
                         static_cast<unsigned long>(exec_handle),
                         static_cast<int>(it_st));
        }
    }

    // Build manifest JSON and write it FIRST, so a crash mid-snapshot
    // still leaves a parseable manifest (matches the memory-snapshot
    // defensive ordering above).
    nlohmann::json vars_arr = nlohmann::json::array();
    for (const auto& v : variables) {
        vars_arr.push_back({
            {"executable_sha256", v.exec_sha256},
            {"name", v.name},
            {"size", v.size},
            {"blob", v.blob_relpath},
        });
    }
    nlohmann::json doc;
    doc["variables"] = vars_arr;
    {
        std::string manifest_path = output_dir_ + "/module_variables.json";
        std::ofstream mout(manifest_path);
        mout << doc.dump(2);
        mout.close();
    }

    if (variables.empty()) {
        KERNCAP_INFO("Module-variable snapshot: 0 variables across %zu executable(s)",
                     execs.size());
        return;
    }

    // Make sure the blob directory exists.
    std::string vardir = output_dir_ + "/module_variables";
    mkdir(vardir.c_str(), 0755);

    // Reuse the same chunked staging-buffer pattern as
    // ``snapshot_all_tracked_memory`` so a single large variable
    // (rare but possible for big __device__ arrays) cannot blow up
    // host RAM. Default 64 MiB, tunable via KERNCAP_SNAPSHOT_CHUNK_BYTES.
    constexpr size_t kDefaultChunkBytes = 64ULL << 20;
    constexpr size_t kMinChunkBytes     = 64ULL << 10;
    size_t chunk_bytes = kDefaultChunkBytes;
    if (const char* env = std::getenv("KERNCAP_SNAPSHOT_CHUNK_BYTES")) {
        try {
            long long v = std::stoll(env);
            if (v >= static_cast<long long>(kMinChunkBytes)) {
                chunk_bytes = static_cast<size_t>(v);
            }
        } catch (const std::exception&) {
            // already warned about by snapshot_all_tracked_memory
        }
    }
    std::vector<char> buf(chunk_bytes);

    int ok = 0;
    for (const auto& v : variables) {
        std::string blob_path = output_dir_ + "/" + v.blob_relpath;
        std::ofstream out(blob_path, std::ios::binary);
        if (!out) {
            KERNCAP_WARN("Module-variable snapshot: failed to open %s for writing",
                         blob_path.c_str());
            continue;
        }
        bool var_ok = true;
        for (size_t off = 0; off < v.size; off += chunk_bytes) {
            size_t n = std::min(chunk_bytes, v.size - off);
            void* src = reinterpret_cast<void*>(v.address + off);
            hsa_status_t st = hsa_memory_copy(buf.data(), src, n);
            if (st != HSA_STATUS_SUCCESS) {
                KERNCAP_WARN("Module-variable snapshot: hsa_memory_copy failed "
                             "for '%s' at offset %zu (chunk %zu, err=%d)",
                             v.name.c_str(), off, n, static_cast<int>(st));
                var_ok = false;
                break;
            }
            out.write(buf.data(), n);
            if (!out) {
                KERNCAP_WARN("Module-variable snapshot: write to %s failed at offset %zu",
                             blob_path.c_str(), off);
                var_ok = false;
                break;
            }
        }
        out.close();
        if (var_ok) {
            ++ok;
        } else {
            std::remove(blob_path.c_str());
        }
    }

    KERNCAP_INFO("Module-variable snapshot: %d/%zu variables captured across %zu executable(s)",
                 ok, variables.size(), execs.size());
}


void CaptureState::capture_kernel(
        const hsa_kernel_dispatch_packet_t* disp,
        const std::string& kernel_name) {

    KERNCAP_INFO("Capturing kernel '%s' to %s", kernel_name.c_str(), output_dir_.c_str());

    // Query GPU architecture (deferred from constructor because HIP is
    // not yet initialized when OnLoad runs).
    if (gpu_arch_.empty()) {
        hipDeviceProp_t prop;
        if (hipGetDeviceProperties(&prop, 0) == hipSuccess) {
            gpu_arch_ = prop.gcnArchName;
            auto colon = gpu_arch_.find(':');
            if (colon != std::string::npos)
                gpu_arch_ = gpu_arch_.substr(0, colon);
            KERNCAP_INFO("GPU architecture: %s", gpu_arch_.c_str());
        } else {
            KERNCAP_WARN("Could not query GPU architecture");
        }
    }

    char* kernarg = static_cast<char*>(disp->kernarg_address);

    // Query the actual kernarg segment size from the kernel symbol so we
    // dump exactly the right number of bytes (not a fixed 4096 slab).
    uint32_t kernarg_segment_size = 0;
    {
        hsa_executable_symbol_t sym{};
        bool found = false;
        {
            std::shared_lock g(mutex_);
            auto hit = handle_to_symbol_.find(disp->kernel_object);
            if (hit != handle_to_symbol_.end()) {
                sym = hit->second;
                found = true;
            }
        }
        if (found) {
            hsa_core_call(this, hsa_executable_symbol_get_info,
                          sym,
                          HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                          &kernarg_segment_size);
        }
        if (kernarg_segment_size == 0) {
            KERNCAP_WARN("Could not query kernarg size; falling back to 4096");
            kernarg_segment_size = 4096;
        }
    }

    // Build metadata.json
    nlohmann::json meta;
    meta["kernel_name"] = kernel_name;
    meta["kernel_symbol"] = get_kernel_symbol(disp->kernel_object);
    if (!gpu_arch_.empty()) {
        meta["gpu_arch"] = gpu_arch_;
    }
    meta["grid"]  = {{"x", disp->grid_size_x},
                     {"y", disp->grid_size_y},
                     {"z", disp->grid_size_z}};
    meta["block"] = {{"x", disp->workgroup_size_x},
                     {"y", disp->workgroup_size_y},
                     {"z", disp->workgroup_size_z}};
    meta["lds_size"] = disp->group_segment_size;

    // ----- Kernarg slot table (parsed from AMDGPU metadata at load time) -----
    //
    // Two-phase lookup to keep the lock order ``mutex_ -> code_object_mutex_``
    // consistent with ``hsa_executable_symbol_get_info`` (the only other
    // place that takes both).  Phase 1 reads symbol-side state under
    // ``mutex_`` and copies the executable handle + symbol string out;
    // phase 2 reads code-object-side state under ``code_object_mutex_``
    // alone.  The previous code held ``code_object_mutex_`` first and
    // then took ``mutex_``, which is the AB-BA partner of
    // ``hsa_executable_symbol_get_info``'s ``mutex_ -> code_object_mutex_``
    // and could deadlock when Triton compiles new kernels concurrently
    // with active dispatches.
    std::optional<KernelKernargInfo> kinfo;
    std::string blob_sha;
    uint64_t exec_handle = 0;
    bool exec_found = false;
    std::string sym_str;
    {
        std::shared_lock g(mutex_);
        auto sym_it = handle_to_symbol_.find(disp->kernel_object);
        if (sym_it != handle_to_symbol_.end()) {
            auto exec_it = symbol_to_executable_.find(sym_it->second);
            if (exec_it != symbol_to_executable_.end()) {
                exec_handle = exec_it->second;
                exec_found = true;
            }
            auto sit = symbol_names_.find(sym_it->second);
            if (sit != symbol_names_.end()) sym_str = sit->second;
        }
    }
    if (exec_found) {
        std::lock_guard lock(code_object_mutex_);
        auto kit = executable_kernargs_.find(exec_handle);
        if (kit != executable_kernargs_.end()) {
            auto found = find_kernarg_metadata(kit->second, sym_str);
            if (found) kinfo = std::move(*found);
        }
        auto bit = executable_blob_sha256_.find(exec_handle);
        if (bit != executable_blob_sha256_.end()) blob_sha = bit->second;
    }

    if (kinfo) {
        nlohmann::json slots = nlohmann::json::array();
        for (const auto& s : kinfo->args) {
            slots.push_back({
                {"offset", s.offset},
                {"size",   s.size},
                {"value_kind", s.kind_str.empty() ? to_string(s.kind) : s.kind_str},
                {"value_type", s.value_type},
                {"address_space", s.address_space},
                {"name", s.name},
            });
        }
        meta["kernarg_slots"] = slots;
        meta["kernarg_segment_size_meta"] = kinfo->kernarg_segment_size;
        meta["sgpr_count"] = kinfo->sgpr_count;
        meta["vgpr_count"] = kinfo->vgpr_count;
    }

    // ----- Triton name_map.json lookup (by content hash) -----
    if (!blob_sha.empty()) {
        meta["hsaco_sha256"] = blob_sha;
        if (!triton_name_map_path_.empty()) {
            maybe_reload_triton_name_map();
            auto row = lookup_triton_user_name(blob_sha);
            if (row) {
                meta["triton_user_name"] = row->user_name;
                if (!row->constexpr_values.is_null())
                    meta["constexpr_values"] = row->constexpr_values;
            }
        }
    }

    // Dump the raw kernarg buffer
    {
        std::string blob_path = output_dir_ + "/kernarg_raw.bin";
        std::ofstream out(blob_path, std::ios::binary);
        if (out) {
            out.write(kernarg, kernarg_segment_size);
        }
        meta["kernarg_raw"] = "kernarg_raw.bin";
    }

    std::string meta_path = output_dir_ + "/metadata.json";
    std::ofstream meta_out(meta_path);
    meta_out << meta.dump(2);
    meta_out.close();

    // Write dispatch.json for kerncap-replay (BEFORE snapshot, in case
    // snapshot crashes on a freed region)
    {
        nlohmann::json dispatch_meta;
        dispatch_meta["mangled_name"] = get_kernel_symbol(disp->kernel_object);
        dispatch_meta["demangled_name"] = kernel_name;

        // Query agent/ISA info
        hsa_agent_t agent{};
        {
            std::shared_lock g(mutex_);
            if (!queue_agents_.empty()) {
                agent = queue_agents_.begin()->second;
            }
        }

        char agent_name[64] = {};
        char isa_name[128] = {};
        uint32_t wavefront_size = 64;

        if (agent.handle != 0) {
            hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, agent_name);

            hsa_isa_t isa{};
            if (hsa_agent_get_info(agent, HSA_AGENT_INFO_ISA, &isa) == HSA_STATUS_SUCCESS) {
                hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, isa_name);
            }
            hsa_agent_get_info(agent, HSA_AGENT_INFO_WAVEFRONT_SIZE, &wavefront_size);
        }

        dispatch_meta["agent_name"] = agent_name;
        dispatch_meta["isa_name"] = isa_name;
        dispatch_meta["wavefront_size"] = wavefront_size;

        dispatch_meta["grid"] = {disp->grid_size_x, disp->grid_size_y, disp->grid_size_z};
        dispatch_meta["block"] = {disp->workgroup_size_x, disp->workgroup_size_y, disp->workgroup_size_z};
        dispatch_meta["group_segment_size"] = disp->group_segment_size;
        dispatch_meta["private_segment_size"] = disp->private_segment_size;

        dispatch_meta["kernarg_size"] = kernarg_segment_size;
        dispatch_meta["dispatch_index"] = target_dispatch_count_ - 1;

        std::string dispatch_path = output_dir_ + "/dispatch.json";
        std::ofstream dout(dispatch_path);
        dout << dispatch_meta.dump(2);
        dout.close();

        // Also write kernarg.bin (copy of kernarg_raw.bin) for replay
        std::string kernarg_path = output_dir_ + "/kernarg.bin";
        if (access(kernarg_path.c_str(), F_OK) != 0) {
            std::string raw_path = output_dir_ + "/kernarg_raw.bin";
            std::ifstream src(raw_path, std::ios::binary);
            std::ofstream dst(kernarg_path, std::ios::binary);
            dst << src.rdbuf();
        }
    }

    // Write .hsaco if we captured the code object (before snapshot,
    // since snapshot's hsa_memory_copy may crash on freed regions).
    // kernel_hsaco_ stores kernel_object -> executable_handle; the actual
    // bytes live in executable_blobs_ (one copy per loaded executable).
    {
        std::lock_guard lock(code_object_mutex_);
        auto co_it = kernel_hsaco_.find(disp->kernel_object);
        if (co_it != kernel_hsaco_.end()) {
            auto blob_it = executable_blobs_.find(co_it->second);
            if (blob_it != executable_blobs_.end()) {
                std::string hsaco_path = output_dir_ + "/kernel.hsaco";
                std::ofstream out(hsaco_path, std::ios::binary);
                out.write(reinterpret_cast<const char*>(blob_it->second.data()),
                          blob_it->second.size());
                KERNCAP_INFO("Saved kernel.hsaco (%zu bytes)", blob_it->second.size());
            } else {
                KERNCAP_WARN("Kernel mapped to executable 0x%lx but no blob "
                             "stored for that executable",
                             static_cast<unsigned long>(co_it->second));
            }
        } else {
            KERNCAP_WARN("No .hsaco blob captured for this kernel "
                         "(code object interception may not have fired)");
        }
    }

    // Snapshot ALL tracked device memory for VA-faithful replay.
    // This creates memory_regions.json + memory/region_*.bin files
    // that kerncap-replay uses to restore device state at original VAs.
    // Done LAST because hsa_memory_copy may crash on freed regions.
    //
    // Diagnostic knob: ``KERNCAP_DISABLE_SNAPSHOT=1`` skips the D2H
    // copy entirely.  Useful for isolating whether D2H copies on
    // HMM-mapped regions are what triggers downstream GPU page faults
    // (replay artifacts will be incomplete in that case).
    if (std::getenv("KERNCAP_DISABLE_SNAPSHOT") != nullptr) {
        KERNCAP_INFO("Skipping memory snapshot (KERNCAP_DISABLE_SNAPSHOT=1)");
    } else {
        snapshot_all_tracked_memory(disp);
    }

    // Module-variable snapshot (HSA_SYMBOL_KIND_VARIABLE) for kernels that
    // read state populated via hipMemcpyToSymbol / __constant__ / __device__
    // globals. Without this, Kokkos's hip_parallel_launch_constant_memory<...>
    // launchers (LAMMPS TagPairEAMKernelC, SPARTA, ExaMPM) replay against a
    // zero-initialized constant buffer and the kernel faults reading NULL
    // View pointers.
    //
    // Escape hatch: ``KERNCAP_DISABLE_VARIABLE_SNAPSHOT=1`` skips this pass
    // (replay STAGE 4.5 will then no-op for back-compat with older captures).
    if (std::getenv("KERNCAP_DISABLE_VARIABLE_SNAPSHOT") != nullptr) {
        KERNCAP_INFO("Skipping module-variable snapshot "
                     "(KERNCAP_DISABLE_VARIABLE_SNAPSHOT=1)");
    } else {
        snapshot_module_variables(disp);
    }

    KERNCAP_INFO("Capture complete -> %s", meta_path.c_str());
    captured_.store(true, std::memory_order_relaxed);

    // Drop a sentinel file the orchestrator can poll on.  Once this
    // exists, every artifact the reproducer needs is already on disk,
    // and the orchestrator is free to terminate the child process --
    // continuing past this point only exposes us to whatever the host
    // app does next (vLLM warmup, async benchmark loops, etc.) which
    // doesn't benefit from kerncap's hooks and frequently OOMs or
    // hangs because of them.
    {
        std::string sentinel_path = output_dir_ + "/capture_complete";
        std::ofstream sentinel(sentinel_path);
        sentinel << "ok\n";
    }
}

// ---------------------------------------------------------------------------
// Packet interception callback
// ---------------------------------------------------------------------------
void CaptureState::on_submit_packet(
        const void* packets, uint64_t count, uint64_t /*user_que_idx*/,
        void* data, hsa_amd_queue_intercept_packet_writer writer) {
    auto* inst = get_instance();
    if (!inst || !inst->is_active_process()) {
        writer(packets, count);
        return;
    }
    if (inst->bypass_after_capture_ &&
        inst->captured_.load(std::memory_order_relaxed)) {
        writer(packets, count);
        return;
    }

    auto pkt_type = get_packet_type(packets);
    if (pkt_type == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
        inst->handle_packets(
            reinterpret_cast<hsa_queue_t*>(data),
            static_cast<const hsa_kernel_dispatch_packet_t*>(packets),
            count, writer);
    } else {
        writer(packets, count);
    }
}

void CaptureState::handle_packets(
        hsa_queue_t* queue,
        const hsa_kernel_dispatch_packet_t* disp,
        uint64_t count,
        hsa_amd_queue_intercept_packet_writer writer) {
    try {
        uint64_t dispatch_seq =
            dispatch_seen_count_.fetch_add(1, std::memory_order_relaxed);
        uint64_t queue_id = queue ? queue->id : 0;
        uint64_t queue_dispatch_seq = 0;
        {
            std::lock_guard g(mutex_);
            queue_dispatch_seq = queue_dispatch_seen_[queue_id]++;
        }

        // Resolve kernel name
        std::string kernel_name = get_kernel_name(disp->kernel_object);

        // Check if this is our target kernel.  In Triton-HSA mode the
        // user name lives in name_map.json keyed by content hash; fall
        // through to that lookup when the mangled symbol does not match.
        bool is_target = !target_kernel_.empty() &&
                         kernel_name.find(target_kernel_) != std::string::npos;
        if (!is_target && !target_kernel_.empty() &&
            !triton_name_map_path_.empty()) {
            // Two-phase lookup with consistent lock order
            // (``mutex_ -> code_object_mutex_``) to avoid the AB-BA
            // deadlock with ``hsa_executable_symbol_get_info``.  This
            // path is hot: it fires on every non-target dispatch for
            // the entire life of a Triton-HSA capture.
            uint64_t exec_handle = 0;
            bool exec_found = false;
            {
                std::shared_lock g(mutex_);
                auto sit = handle_to_symbol_.find(disp->kernel_object);
                if (sit != handle_to_symbol_.end()) {
                    auto eit = symbol_to_executable_.find(sit->second);
                    if (eit != symbol_to_executable_.end()) {
                        exec_handle = eit->second;
                        exec_found = true;
                    }
                }
            }
            std::string blob_sha;
            if (exec_found) {
                std::lock_guard lock(code_object_mutex_);
                auto bit = executable_blob_sha256_.find(exec_handle);
                if (bit != executable_blob_sha256_.end())
                    blob_sha = bit->second;
            }
            if (!blob_sha.empty()) {
                maybe_reload_triton_name_map();
                auto row = lookup_triton_user_name(blob_sha);
                if (row && row->user_name.find(target_kernel_) != std::string::npos) {
                    is_target = true;
                    KERNCAP_INFO("Matched Triton kernel '%s' via content-hash lookup",
                                 row->user_name.c_str());
                }
            }
        }

        // Dispatch tracing (after is_target is finalized so the
        // target_match field reflects the post-Triton-match value).
        bool trace_dispatch = should_trace_dispatch(
            queue_id, dispatch_seq, queue_dispatch_seq, is_target);
        if (trace_dispatch) {
            KERNCAP_INFO(
                "Dispatch[%" PRIu64 "] queue=0x%" PRIx64
                " kernel='%s' object=0x%" PRIx64
                " queue_dispatch=%" PRIu64
                " target_match=%s target_seen=%u"
                " grid=(%u,%u,%u) block=(%u,%u,%u)"
                " completion_signal=0x%" PRIx64 " count=%" PRIu64,
                dispatch_seq,
                queue_id,
                kernel_name.c_str(),
                static_cast<uint64_t>(disp->kernel_object),
                queue_dispatch_seq,
                is_target ? "true" : "false",
                target_dispatch_count_,
                disp->grid_size_x,
                disp->grid_size_y,
                disp->grid_size_z,
                disp->workgroup_size_x,
                disp->workgroup_size_y,
                disp->workgroup_size_z,
                static_cast<uint64_t>(disp->completion_signal.handle),
                count);
        }

        if (is_target && !captured_.load(std::memory_order_relaxed)) {
            // Check dispatch index filter
            bool should_capture = false;
            if (target_dispatch_ < 0) {
                should_capture = true;  // Capture first match
            } else if (static_cast<int>(target_dispatch_count_) == target_dispatch_) {
                should_capture = true;
            }
            target_dispatch_count_++;

            if (should_capture) {
                // Create a new signal so we can wait for the kernel to complete
                // before capturing the output buffers
                hsa_signal_t new_signal;
                auto status = hsa_core_call(this, hsa_signal_create,
                                            1, 0, nullptr, &new_signal);
                if (status != HSA_STATUS_SUCCESS) {
                    KERNCAP_ERROR("Failed to create signal");
                    writer(disp, count);
                    return;
                }

                // Capture input arguments BEFORE dispatch (for input pointers)
                // We make a copy of the packet, swap the signal, dispatch, wait,
                // then capture outputs.

                // First: capture all input (const) pointer args before dispatch
                // (they won't change after the kernel runs)

                hsa_kernel_dispatch_packet_t modified = *disp;
                hsa_signal_t old_signal = modified.completion_signal;
                modified.completion_signal = new_signal;

                writer(&modified, count);

                // Wait for kernel completion
                hsa_signal_wait_scacquire(new_signal, HSA_SIGNAL_CONDITION_EQ,
                                          0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

                // Now capture everything (inputs are unchanged, outputs are ready)
                capture_kernel(disp, kernel_name);

                // Complete the original signal
                if (old_signal.handle != 0) {
                    hsa_core_call(this, hsa_signal_subtract_relaxed, old_signal, 1);
                }
                hsa_core_call(this, hsa_signal_destroy, new_signal);
                return;
            }
        }

        // Not our target — pass through normally
        if (trace_dispatch && !is_target) {
            KERNCAP_INFO("Dispatch[%" PRIu64 "] pass-through", dispatch_seq);
        }
        writer(disp, count);

    } catch (const std::exception& e) {
        KERNCAP_ERROR("handle_packets exception: %s", e.what());
        writer(disp, count);
    }
}

}  // namespace kerncap

// ---------------------------------------------------------------------------
// rocprofiler-sdk entry points
// ---------------------------------------------------------------------------

static void rocp_hsa_table_callback(
        rocprofiler_intercept_table_t type,
        uint64_t lib_version, uint64_t lib_instance,
        void** tables, uint64_t num_tables, void* /*user_data*/) {
    if (type != ROCPROFILER_HSA_TABLE || num_tables < 1 || !tables || !tables[0])
        return;

    auto* table = static_cast<HsaApiTable*>(tables[0]);
    KERNCAP_INFO("rocprofiler-sdk: received HsaApiTable* (lib_version=%" PRIu64
                 ", lib_instance=%" PRIu64 ")", lib_version, lib_instance);
    kerncap::CaptureState::get_instance(table, lib_version, 0, nullptr);
}

static void rocp_tool_fini(void* /*user_data*/) {
    KERNCAP_INFO("rocprofiler-sdk: tool finalize");
}

extern "C" {

PUBLIC_API rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t version, const char* runtime_version,
                      uint32_t priority,
                      rocprofiler_client_id_t* client_id) {
    (void)version;
    (void)runtime_version;
    (void)priority;

    client_id->name = "kerncap";

    KERNCAP_INFO("rocprofiler_configure: registering HSA table interception");
    rocprofiler_at_intercept_table_registration(
        rocp_hsa_table_callback, ROCPROFILER_HSA_TABLE, nullptr);

    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t),
        nullptr,          // initialize
        rocp_tool_fini,   // finalize
        nullptr};         // tool_data
    return &cfg;
}

PUBLIC_API bool OnLoad(HsaApiTable* /*table*/,
                       uint64_t /*runtime_version*/,
                       uint64_t /*failed_tool_count*/,
                       const char* const* /*failed_tool_names*/) {
    KERNCAP_ERROR("kerncap no longer uses HSA_TOOLS_LIB. "
                  "Use LD_PRELOAD=libkerncap.so instead.");
    abort();
    return false;
}

PUBLIC_API void OnUnload() { }

}  // extern "C"
