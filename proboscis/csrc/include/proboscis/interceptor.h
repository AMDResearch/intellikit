/******************************************************************************
 * MIT License
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/
#pragma once

#include "proboscis.h"

#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

// Forward declarations from hsa_api_trace.h (can't include directly due to
// broken relative includes in ROCm 6.x/7.x installed headers).
typedef void (*hsa_amd_queue_intercept_packet_writer)(const void* pkts, uint64_t pkt_count);

#ifndef PUBLIC_API
#define PUBLIC_API __attribute__((visibility("default")))
#endif

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

    static void OnSubmitPackets(
        const void* in_packets, uint64_t count,
        uint64_t user_que_idx, void* data,
        hsa_amd_queue_intercept_packet_writer writer);

    // Probe buffer management
    void* allocateProbeBuffer(hsa_agent_t agent, size_t size);
    void freeProbeBuffer(void* ptr);
    void writeResults();

    bool shouldInstrument(const std::string& kernel_name) const;

private:
    ProboscisInterceptor() = default;
    ~ProboscisInterceptor();

    void addQueue(hsa_queue_t* queue, hsa_agent_t agent);
    void doPackets(hsa_queue_t* queue, const hsa_kernel_dispatch_packet_t* packets,
                   uint64_t count, hsa_amd_queue_intercept_packet_writer writer);

    static ProboscisInterceptor* singleton_;
    static std::mutex singleton_mutex_;

    ProbeConfig config_;
    std::map<hsa_queue_t*, hsa_agent_t> queues_;
    std::map<uint64_t, KernelInfo> kernel_objects_;
    std::vector<void*> probe_buffers_;
    std::mutex mutex_;
    std::atomic<bool> shutting_down_{false};
};
