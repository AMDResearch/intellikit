/****************************************************************************
MIT License

Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
****************************************************************************/

#include <hip/hip_runtime.h>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <hsa/hsa.h>
#include <vector>
#include <iostream>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <hsa/hsa_ven_amd_loader.h>
#include "log.hpp"

#define PUBLIC_API __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))

#define hsa_ext_call(instance, FUNC, ...) \
  instance->rocr_api_table_.amd_ext_->FUNC##_fn(__VA_ARGS__)

#define hsa_core_call(instance, FUNC, ...) \
  instance->rocr_api_table_.core_->FUNC##_fn(__VA_ARGS__)

namespace intelliperf {

struct hsa_executable_symbol_hasher {
  std::size_t operator()(const hsa_executable_symbol_t& symbol) const {
    return std::hash<uint64_t>()(symbol.handle);
  }
};
struct hsa_executable_symbol_compare {
  using result_type = bool;
  using first_argument_type = hsa_executable_symbol_t;
  using second_argument_type = hsa_executable_symbol_t;

  bool operator()(const hsa_executable_symbol_t& lhs,
                  const hsa_executable_symbol_t& rhs) const {
    return lhs.handle == rhs.handle;
  }
};

struct hsa_agent_compare {
  bool operator()(const hsa_agent_t& lhs, const hsa_agent_t& rhs) const {
    return lhs.handle < rhs.handle;
  }
};

struct HsaMemoryRegion {
    hsa_region_t region;
    size_t size;
    bool is_global;
    bool is_kernarg;
    bool is_local;

    HsaMemoryRegion(hsa_region_t r, size_t s, bool global, bool kernarg, bool local)
        : region(r), size(s), is_global(global), is_kernarg(kernarg), is_local(local) {}
};

struct HsaMemoryPool {
    hsa_amd_memory_pool_t pool;
    size_t size;
    bool is_fine_grained;
    bool is_coarse_grained;

    HsaMemoryPool(hsa_amd_memory_pool_t p, size_t s, bool fine, bool coarse)
        : pool(p), size(s), is_fine_grained(fine), is_coarse_grained(coarse) {}
};

struct HsaAgent {
    hsa_agent_t agent;
    std::string name;
    bool is_gpu;
    std::vector<HsaMemoryPool> memory_pools;
    std::vector<HsaMemoryRegion> memory_regions;

    explicit HsaAgent(hsa_agent_t a) : agent(a), is_gpu(false) {}

    void add_memory_region(const hsa_region_t& region, size_t size, bool global, bool kernarg, bool local) {
        memory_regions.emplace_back(region, size, global, kernarg, local);
    }

    void add_memory_pool(const hsa_amd_memory_pool_t& pool, size_t size, bool fine, bool coarse) {
        memory_pools.emplace_back(pool, size, fine, coarse);
    }

    void print_info() const {
        LOG_DETAIL("Agent: 0x{:x}, Name: {}, Type: {}", agent.handle, name, is_gpu ? "GPU" : "CPU");

        LOG_DETAIL("Memory Pools:");
        for (const auto& pool : memory_pools) {
            LOG_DETAIL("  - Pool Size: {}, Fine-Grained: {}, Coarse-Grained: {}",
                       pool.size, pool.is_fine_grained, pool.is_coarse_grained);
        }

        LOG_DETAIL("Memory Regions:");
        for (const auto& region : memory_regions) {
            LOG_DETAIL("  - Region Size: {}, Global: {}, Kernarg: {}, Local: {}",
                       region.size, region.is_global, region.is_kernarg, region.is_local);
        }
    }

    static void get_all_agents(std::vector<HsaAgent>& agents) {
        auto agent_callback = [](hsa_agent_t agent, void* data) -> hsa_status_t {
            auto* agents_vector = static_cast<std::vector<HsaAgent>*>(data);
            agents_vector->emplace_back(agent);
            HsaAgent& hsa_agent = agents_vector->back();

            // Get agent name
            char name[64] = {0};
            hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
            hsa_agent.name = name;

            // Get device type
            hsa_device_type_t device_type;
            hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
            hsa_agent.is_gpu = (device_type == HSA_DEVICE_TYPE_GPU);

            // Iterate over memory regions
            auto region_callback = [](hsa_region_t region, void* agent_ptr) -> hsa_status_t {
                HsaAgent& hsa_agent = *static_cast<HsaAgent*>(agent_ptr);

                hsa_region_segment_t segment;
                hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
                
                size_t size;
                hsa_region_get_info(region, HSA_REGION_INFO_SIZE, &size);
                
                bool is_global = (segment == HSA_REGION_SEGMENT_GLOBAL);
                bool is_kernarg = (segment == HSA_REGION_SEGMENT_KERNARG);
                bool is_local = (segment == HSA_REGION_SEGMENT_GROUP);
                
                hsa_agent.add_memory_region(region, size, is_global, is_kernarg, is_local);
                return HSA_STATUS_SUCCESS;
            };

            hsa_agent_iterate_regions(agent, region_callback, &hsa_agent);

            // Iterate over memory pools
            auto pool_callback = [](hsa_amd_memory_pool_t pool, void* agent_ptr) -> hsa_status_t {
                HsaAgent& hsa_agent = *static_cast<HsaAgent*>(agent_ptr);
                
                size_t size;
                hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SIZE, &size);

                hsa_amd_segment_t segment;
                hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
                
                bool is_fine = (segment == HSA_AMD_SEGMENT_GLOBAL);
                bool is_coarse = (segment == HSA_AMD_SEGMENT_GROUP);
                
                hsa_agent.add_memory_pool(pool, size, is_fine, is_coarse);
                return HSA_STATUS_SUCCESS;
            };

            hsa_amd_agent_iterate_memory_pools(agent, pool_callback, &hsa_agent);

            return HSA_STATUS_SUCCESS;
        };

        hsa_iterate_agents(agent_callback, &agents);
    }
};



class accordo {
 public:
  static accordo* get_instance(HsaApiTable* table = nullptr,
                              uint64_t runtime_version = 0,
                              uint64_t failed_tool_count = 0,
                              const char* const* failed_tool_names = nullptr);

 private:
  accordo(HsaApiTable* table,
         std::uint64_t runtime_version,
         std::uint64_t failed_tool_count,
         const char* const* failed_tool_names);
  ~accordo();
  void save_hsa_api();
  void restore_hsa_api();
  void hook_api();
  void discover_agents();
  static void on_submit_packet(const void* in_packets,
                               uint64_t count,
                               uint64_t user_que_idx,
                               void* data,
                               hsa_amd_queue_intercept_packet_writer writer);
  void write_packets(hsa_queue_t* queue,
                     const hsa_ext_amd_aql_pm4_packet_t* packet,
                     uint64_t count,
                     hsa_amd_queue_intercept_packet_writer writer);

  hsa_status_t add_queue(hsa_queue_t* queue, hsa_agent_t agent);
  std::string packet_to_text(const hsa_ext_amd_aql_pm4_packet_t* packet);
  std::optional<void*> is_traceable_packet(const hsa_ext_amd_aql_pm4_packet_t* packet);
  void send_message_and_wait(void* args);
  static hsa_status_t hsa_queue_create(hsa_agent_t agent,
                                       uint32_t size,
                                       hsa_queue_type32_t type,
                                       void (*callback)(hsa_status_t status,
                                                        hsa_queue_t* source,
                                                        void* data),
                                       void* data,
                                       uint32_t private_segment_size,
                                       uint32_t group_segment_size,
                                       hsa_queue_t** queue);
  static hsa_status_t hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t pool,
                                                   size_t size,
                                                   uint32_t flags,
                                                   void** ptr);
  static hsa_status_t hsa_memory_allocate(hsa_region_t region, size_t size, void** ptr);
  static hsa_status_t hsa_queue_destroy(hsa_queue_t* queue);
  static hsa_status_t hsa_executable_get_symbol_by_name(hsa_executable_t executable,
                                                        const char* symbol_name,
                                                        const hsa_agent_t* agent,
                                                        hsa_executable_symbol_t* symbol);
  static hsa_status_t hsa_executable_symbol_get_info(
      hsa_executable_symbol_t executable_symbol,
      hsa_executable_symbol_info_t attribute,
      void* value);

  using packet_word_t = uint32_t;

  static const packet_word_t header_type_mask = (1ul << HSA_PACKET_HEADER_WIDTH_TYPE) - 1;
  static const packet_word_t header_screlease_scope_mask = 0x3;
  static const packet_word_t header_scacquire_scope_mask = 0x3;
  static hsa_packet_type_t get_header_type(const hsa_ext_amd_aql_pm4_packet_t* packet) {
    const packet_word_t* header = reinterpret_cast<const packet_word_t*>(packet);
    return static_cast<hsa_packet_type_t>((*header >> HSA_PACKET_HEADER_TYPE) &
                                          header_type_mask);
  }
  static hsa_packet_type_t get_header_release_scope(
      const hsa_kernel_dispatch_packet_t* packet) {
    const packet_word_t* header = reinterpret_cast<const packet_word_t*>(packet);
    return static_cast<hsa_packet_type_t>(
        (*header >> HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE) &
        header_screlease_scope_mask);
  }
  static hsa_packet_type_t get_header_acquire_scope(
      const hsa_kernel_dispatch_packet_t* packet) {
    const packet_word_t* header = reinterpret_cast<const packet_word_t*>(packet);
    return static_cast<hsa_packet_type_t>(
        (*header >> HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) &
        header_scacquire_scope_mask);
  }

  std::string get_kernel_name(const std::uint64_t kernel_object);

 private:
  static std::mutex mutex_;
  static std::shared_mutex stop_mutex_;
  static accordo* singleton_;

  std::vector<HsaAgent> agents_;

  HsaApiTable* api_table_;
  HsaApiTable rocr_api_table_;

  std::map<hsa_queue_t*, std::pair<unsigned int, std::uint64_t>> queue_ids_;
  std::map<hsa_agent_t, std::string, hsa_agent_compare> agents_names_;
  std::unordered_map<hsa_executable_symbol_t,
                     std::string,
                     hsa_executable_symbol_hasher,
                     hsa_executable_symbol_compare>
      symbols_names_;

  std::unordered_map<std::string, hsa_executable_t> kernels_executables_;

  std::unordered_map<std::uint64_t, hsa_executable_symbol_t> handles_symbols_;
  std::unordered_map<void*, std::size_t> pointer_sizes_;
  std::mutex mm_mutex_;
};

}  // namespace intelliperf
