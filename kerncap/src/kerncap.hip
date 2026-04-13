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
#include <cxxabi.h>
#include <fcntl.h>
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

namespace kerncap {

// ---------------------------------------------------------------------------
// Static members
// ---------------------------------------------------------------------------
CaptureState* CaptureState::singleton_{nullptr};
std::mutex CaptureState::mutex_{};

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
    const std::lock_guard<std::mutex> lock(mutex_);
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
    queue_agents_.clear();
    target_dispatch_count_ = 0;
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
    api_table_->core_->hsa_code_object_reader_create_from_memory_fn =
        CaptureState::hsa_code_object_reader_create_from_memory;
    api_table_->core_->hsa_executable_load_agent_code_object_fn =
        CaptureState::hsa_executable_load_agent_code_object;
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

        // Lazily associate kernel_object with its .hsaco blob by
        // following symbol -> executable -> blob.
        auto exec_it = inst->symbol_to_executable_.find(symbol);
        if (exec_it != inst->symbol_to_executable_.end()) {
            std::lock_guard co_lock(inst->code_object_mutex_);
            auto blob_it = inst->executable_blobs_.find(exec_it->second);
            if (blob_it != inst->executable_blobs_.end()) {
                inst->kernel_hsaco_[kernel_object] = blob_it->second;
            }
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
            inst->executable_blobs_[executable.handle] = std::move(it->second);
            inst->pending_reader_blobs_.erase(it);
            KERNCAP_INFO("Stored code object blob for executable 0x%lx",
                         executable.handle);
        }
    }
    return result;
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

    if (!inst->is_active_process()) {
        KERNCAP_INFO("Passive mode (pid=%d): passing through hsa_queue_create",
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
    auto hit = handle_to_symbol_.find(kernel_object);
    if (hit == handle_to_symbol_.end()) return "<unknown>";
    auto sit = symbol_names_.find(hit->second);
    if (sit == symbol_names_.end()) return "<unknown>";
    return demangle(sit->second.c_str());
}

std::string CaptureState::get_kernel_symbol(uint64_t kernel_object) {
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
    int ok_count = 0;
    for (const auto& a : allocs) {
        if (a.size == 0 || a.size > (1ULL << 40)) continue;

        uint64_t base = reinterpret_cast<uint64_t>(a.ptr);

        std::vector<char> buf(a.size);
        hsa_status_t st = hsa_memory_copy(buf.data(), a.ptr, a.size);

        if (st == HSA_STATUS_SUCCESS) {
            char filename[256];
            std::snprintf(filename, sizeof(filename),
                         "%s/region_%" PRIx64 ".bin", memdir.c_str(), base);
            std::ofstream out(filename, std::ios::binary);
            out.write(buf.data(), a.size);
            ok_count++;
        } else {
            KERNCAP_WARN("snapshot: hsa_memory_copy failed for region 0x%" PRIx64
                         " (%zu bytes, err=%d)", base, a.size, static_cast<int>(st));
        }
    }

    KERNCAP_INFO("Memory snapshot: %d/%zu regions captured",
                 ok_count, allocs.size());
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
        auto hit = handle_to_symbol_.find(disp->kernel_object);
        if (hit != handle_to_symbol_.end()) {
            hsa_core_call(this, hsa_executable_symbol_get_info,
                          hit->second,
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
        if (!queue_agents_.empty()) {
            agent = queue_agents_.begin()->second;
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
    // since snapshot's hsa_memory_copy may crash on freed regions)
    {
        std::lock_guard lock(code_object_mutex_);
        auto co_it = kernel_hsaco_.find(disp->kernel_object);
        if (co_it != kernel_hsaco_.end()) {
            std::string hsaco_path = output_dir_ + "/kernel.hsaco";
            std::ofstream out(hsaco_path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(co_it->second.data()),
                      co_it->second.size());
            KERNCAP_INFO("Saved kernel.hsaco (%zu bytes)", co_it->second.size());
        } else {
            KERNCAP_WARN("No .hsaco blob captured for this kernel "
                         "(code object interception may not have fired)");
        }
    }

    // Snapshot ALL tracked device memory for VA-faithful replay.
    // This creates memory_regions.json + memory/region_*.bin files
    // that kerncap-replay uses to restore device state at original VAs.
    // Done LAST because hsa_memory_copy may crash on freed regions.
    snapshot_all_tracked_memory(disp);

    KERNCAP_INFO("Capture complete -> %s", meta_path.c_str());
    captured_.store(true, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Packet interception callback
// ---------------------------------------------------------------------------
void CaptureState::on_submit_packet(
        const void* packets, uint64_t count, uint64_t /*user_que_idx*/,
        void* data, hsa_amd_queue_intercept_packet_writer writer) {
    auto* inst = get_instance();
    if (!inst) {
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
        hsa_queue_t* /*queue*/,
        const hsa_kernel_dispatch_packet_t* disp,
        uint64_t count,
        hsa_amd_queue_intercept_packet_writer writer) {
    try {
        // Resolve kernel name
        std::string kernel_name = get_kernel_name(disp->kernel_object);

        // Check if this is our target kernel
        bool is_target = !target_kernel_.empty() &&
                         kernel_name.find(target_kernel_) != std::string::npos;

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
