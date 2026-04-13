/**
 * kerncap.hpp — HSA interception data structures and API table wrappers
 *
 * Adapted from Accordo (AMD Research). Provides the capture state machine
 * for intercepting kernel dispatches and recording arguments, grid/block
 * dimensions, and buffer contents.
 */
#pragma once

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "kerncap_log.hpp"

#define PUBLIC_API __attribute__((visibility("default")))

// ---------------------------------------------------------------------------
// Macros to call through the saved (original) HSA API table
// ---------------------------------------------------------------------------
#define hsa_ext_call(inst, FUNC, ...) \
    (inst)->saved_api_.amd_ext_->FUNC##_fn(__VA_ARGS__)

#define hsa_core_call(inst, FUNC, ...) \
    (inst)->saved_api_.core_->FUNC##_fn(__VA_ARGS__)

namespace kerncap {

// ---------------------------------------------------------------------------
// HSA symbol helpers
// ---------------------------------------------------------------------------
struct SymbolHasher {
    std::size_t operator()(const hsa_executable_symbol_t& s) const {
        return std::hash<uint64_t>()(s.handle);
    }
};

struct SymbolEqual {
    bool operator()(const hsa_executable_symbol_t& a,
                    const hsa_executable_symbol_t& b) const {
        return a.handle == b.handle;
    }
};

struct AgentLess {
    bool operator()(const hsa_agent_t& a, const hsa_agent_t& b) const {
        return a.handle < b.handle;
    }
};

// ---------------------------------------------------------------------------
// CaptureState — the singleton HSA tool that intercepts kernel dispatches
// ---------------------------------------------------------------------------
class CaptureState {
public:
    // Singleton access — first call creates the instance from OnLoad args
    static CaptureState* get_instance(
        HsaApiTable* table = nullptr,
        uint64_t runtime_version = 0,
        uint64_t failed_tool_count = 0,
        const char* const* failed_tool_names = nullptr);

    // Returns true if this process should perform queue interception.
    // In multi-process apps (KERNCAP_CAPTURE_CHILD=1), the parent goes
    // passive after fork and only the child actively intercepts.
    bool is_active_process();

    // Saved (original) API table — public so macros can use it
    HsaApiTable saved_api_;

private:
    CaptureState(HsaApiTable* table,
                 uint64_t runtime_version,
                 uint64_t failed_tool_count,
                 const char* const* failed_tool_names);
    ~CaptureState();

    // Setup
    void save_hsa_api();
    void hook_api();

    // ---- Hooked HSA functions (static so they match the C signature) ----

    static hsa_status_t hsa_queue_create(
        hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
        void (*callback)(hsa_status_t, hsa_queue_t*, void*),
        void* data, uint32_t private_segment_size,
        uint32_t group_segment_size, hsa_queue_t** queue);

    static hsa_status_t hsa_queue_destroy(hsa_queue_t* queue);

    static hsa_status_t hsa_amd_memory_pool_allocate(
        hsa_amd_memory_pool_t pool, size_t size, uint32_t flags, void** ptr);

    static hsa_status_t hsa_amd_memory_pool_free(void* ptr);

    static hsa_status_t hsa_memory_allocate(
        hsa_region_t region, size_t size, void** ptr);

    static hsa_status_t hsa_executable_get_symbol_by_name(
        hsa_executable_t executable, const char* symbol_name,
        const hsa_agent_t* agent, hsa_executable_symbol_t* symbol);

    static hsa_status_t hsa_executable_symbol_get_info(
        hsa_executable_symbol_t symbol,
        hsa_executable_symbol_info_t attribute, void* value);

    // ---- VMEM tracking hooks ----

    static hsa_status_t hsa_amd_vmem_address_reserve(
        void** va, size_t size, uint64_t address, uint64_t flags);

    static hsa_status_t hsa_amd_vmem_address_free(
        void* va, size_t size);

    static hsa_status_t hsa_amd_vmem_map(
        void* va, size_t size, size_t in_offset,
        hsa_amd_vmem_alloc_handle_t memory_handle, uint64_t flags);

    static hsa_status_t hsa_amd_vmem_unmap(
        void* va, size_t size);

    // ---- Code object interception (for .hsaco capture) ----

    static hsa_status_t hsa_code_object_reader_create_from_memory(
        const void* code_object, size_t size,
        hsa_code_object_reader_t* code_object_reader);

    static hsa_status_t hsa_executable_load_agent_code_object(
        hsa_executable_t executable, hsa_agent_t agent,
        hsa_code_object_reader_t code_object_reader,
        const char* options, hsa_loaded_code_object_t* loaded_code_object);

    // ---- Packet interception ----

    static void on_submit_packet(
        const void* packets, uint64_t count, uint64_t user_que_idx,
        void* data, hsa_amd_queue_intercept_packet_writer writer);

    void handle_packets(
        hsa_queue_t* queue,
        const hsa_kernel_dispatch_packet_t* packets,
        uint64_t count,
        hsa_amd_queue_intercept_packet_writer writer);

    // ---- Helpers ----

    using PacketWord = uint32_t;

    static constexpr PacketWord kHeaderTypeMask =
        (1u << HSA_PACKET_HEADER_WIDTH_TYPE) - 1;

    static hsa_packet_type_t get_packet_type(const void* packet) {
        auto header = *reinterpret_cast<const PacketWord*>(packet);
        return static_cast<hsa_packet_type_t>(
            (header >> HSA_PACKET_HEADER_TYPE) & kHeaderTypeMask);
    }

    std::string get_kernel_name(uint64_t kernel_object);
    std::string get_kernel_symbol(uint64_t kernel_object);

    // Snapshot all tracked device memory for VA-faithful replay
    void snapshot_all_tracked_memory(const hsa_kernel_dispatch_packet_t* disp);

    // Capture kernel dispatch data to the output directory
    void capture_kernel(const hsa_kernel_dispatch_packet_t* disp,
                        const std::string& kernel_name);

    // Fork safety: clear inherited tracking state in the child process
    void reset_inherited_state();

    // pthread_atfork parent handler — sets fork_detected_ flag
    static void atfork_parent_handler();

private:
    // ---- State ----

    static CaptureState* singleton_;
    static std::shared_mutex mutex_;

    HsaApiTable* api_table_;   // live table (hooked)

    // Symbol tracking: kernel_object handle -> name
    std::unordered_map<hsa_executable_symbol_t, std::string,
                       SymbolHasher, SymbolEqual> symbol_names_;
    std::unordered_map<uint64_t, hsa_executable_symbol_t> handle_to_symbol_;
    // symbol -> executable handle (for lazy kernel_object -> hsaco association)
    std::unordered_map<hsa_executable_symbol_t, uint64_t,
                       SymbolHasher, SymbolEqual> symbol_to_executable_;

    // Memory tracking: device pointer -> allocation size
    std::unordered_map<void*, std::size_t> pointer_sizes_;
    std::unordered_set<void*> vmem_tracked_;  // subset of pointer_sizes_ from VMEM APIs
    std::mutex ptr_mutex_;

    // Code object tracking for .hsaco capture
    std::unordered_map<uint64_t, std::vector<uint8_t>> pending_reader_blobs_;  // reader handle -> blob
    std::unordered_map<uint64_t, std::vector<uint8_t>> executable_blobs_;      // executable handle -> blob
    std::unordered_map<uint64_t, std::vector<uint8_t>> kernel_hsaco_;          // kernel_object -> blob
    std::mutex code_object_mutex_;

    // Queue tracking
    std::map<hsa_queue_t*, hsa_agent_t> queue_agents_;

    // Dispatch counter for the target kernel (for --dispatch filtering)
    uint32_t target_dispatch_count_ = 0;

    // Configuration (read from env vars once)
    std::string target_kernel_;     // KERNCAP_KERNEL
    int target_dispatch_ = -1;      // KERNCAP_DISPATCH (-1 = first match)
    std::string output_dir_;        // KERNCAP_OUTPUT
    std::string gpu_arch_;          // e.g. "gfx90a" (queried at init)
    std::atomic<bool> captured_{false};  // true after a successful capture

    // Fork safety (multi-process support)
    pid_t initial_pid_ = 0;              // PID when CaptureState was created
    bool capture_child_mode_ = false;    // KERNCAP_CAPTURE_CHILD env var present
    std::atomic<bool> fork_detected_{false};    // set by atfork parent handler after fork()
    std::atomic<bool> child_state_reset_{false}; // true after inherited state cleared in child
};

}  // namespace kerncap
