// kerncap-replay — VA-faithful HSA kernel replay binary
//
// Loads a captured kernel dispatch (dispatch.json, kernarg.bin, kernel.hsaco,
// memory_regions.json, memory/*.bin) and replays it using HSA direct dispatch
// with virtual-address-faithful memory mapping (hsa_amd_vmem).
//
// Vendored from rocm-perf-lab/rocm_perf_lab/replay/replay_full_vm.cpp
// with additions for --dump-output post-execution memory dump.

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hip/hip_runtime.h>

#include <nlohmann/json.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto fsize = f.tellg();
    f.seekg(0);
    std::vector<char> buf(static_cast<size_t>(fsize));
    f.read(buf.data(), fsize);
    return buf;
}

static hsa_agent_t g_gpu_agent{};

static hsa_status_t find_gpu(hsa_agent_t agent, void*) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_GPU) {
        g_gpu_agent = agent;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

struct ReplayMetadata {
    std::string gpu_arch;
    std::string rocm_version;
    std::string hsa_version;
    pid_t pid;
};

struct ReplayResult {
    ReplayMetadata metadata;
    size_t iterations;
    bool recopy;

    std::string kernel_name;
    std::string hsaco_path;

    std::vector<uint64_t> raw_ns;

    double avg_us;
    double min_us;
    double max_us;
};

static void print_json_output(const ReplayResult& r) {
    nlohmann::json j = {
        {"kernel", {
            {"name",       r.kernel_name},
            {"hsaco_path", r.hsaco_path}
        }},
        {"execution", {
            {"iterations", r.iterations},
            {"mode",       r.recopy ? "stateless" : "stateful"}
        }},
        {"environment", {
            {"gpu_arch",      r.metadata.gpu_arch},
            {"rocm_version",  r.metadata.rocm_version},
            {"hsa_version",   r.metadata.hsa_version},
            {"pid",           r.metadata.pid}
        }}
    };
    if (!r.raw_ns.empty()) {
        j["timing"] = {
            {"unit",    "microseconds"},
            {"average", r.avg_us},
            {"min",     r.min_us},
            {"max",     r.max_us}
        };
    }
    std::cout << j.dump(2) << "\n";
}

static ReplayMetadata collect_metadata(hsa_agent_t agent) {
    ReplayMetadata meta{};

    char name[64] = {};
    if (hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name) == HSA_STATUS_SUCCESS) {
        meta.gpu_arch = name;
    } else {
        meta.gpu_arch = "unknown";
    }

    uint16_t hsa_major = 0, hsa_minor = 0;
    if (hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, &hsa_major) == HSA_STATUS_SUCCESS &&
        hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MINOR, &hsa_minor) == HSA_STATUS_SUCCESS) {
        meta.hsa_version = std::to_string(hsa_major) + "." + std::to_string(hsa_minor);
    } else {
        meta.hsa_version = "unknown";
    }

    auto read_rocm_version = []() -> std::string {
        const char* rocm_path = std::getenv("ROCM_PATH");
        std::vector<std::string> candidates;
        if (rocm_path)
            candidates.push_back(std::string(rocm_path) + "/.info/version");
        candidates.push_back("/opt/rocm/.info/version");
        for (const auto& path : candidates) {
            std::ifstream f(path);
            std::string ver;
            if (f.is_open() && std::getline(f, ver) && !ver.empty())
                return ver;
        }
        return "unknown";
    };
    meta.rocm_version = read_rocm_version();

    meta.pid = getpid();

    return meta;
}

struct RegionMeta {
    uint64_t base;
    size_t size;
    uint64_t aligned_base;
    size_t aligned_size;
    size_t offset;
};

struct RegionRuntime {
    void* reserved;
    size_t size;
    size_t offset;
    uint64_t original_base;
    std::vector<char> blob;
};

static void dump_output_regions(
    const std::string& output_dir,
    const std::vector<RegionRuntime>& runtime_regions)
{
    mkdir(output_dir.c_str(), 0755);

    for (const auto& rr : runtime_regions) {
        std::vector<char> host_buf(rr.size);
        void* src = static_cast<void*>(
            static_cast<uint8_t*>(rr.reserved) + rr.offset);
        hsa_status_t st = hsa_memory_copy(host_buf.data(), src, rr.size);
        if (st != HSA_STATUS_SUCCESS) {
            std::cerr << "Warning: hsa_memory_copy failed for region 0x"
                      << std::hex << rr.original_base << "\n";
            continue;
        }

        std::stringstream fname;
        fname << output_dir << "/region_"
              << std::hex << rr.original_base << ".bin";
        std::ofstream out(fname.str(), std::ios::binary);
        out.write(host_buf.data(), host_buf.size());
    }
}

int main(int argc, char** argv) {

    if (argc < 2) {
        std::cerr << "Usage: kerncap-replay <capture_dir> "
                     "[--iterations N] [--no-recopy] [--json] "
                     "[--hsaco FILE] [--dump-output]\n";
        return 1;
    }

    std::string capture_dir = argv[1];

    size_t iterations = 1;
    bool recopy = true;
    bool json_output = false;
    bool dump_output = false;
    bool use_hip = false;
    std::string override_hsaco_path;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoull(argv[++i]);
        } else if (arg == "--no-recopy") {
            recopy = false;
        } else if (arg == "--json") {
            json_output = true;
        } else if (arg == "--hsaco" && i + 1 < argc) {
            override_hsaco_path = argv[++i];
        } else if (arg == "--dump-output") {
            dump_output = true;
        } else if (arg == "--hip-launch") {
            use_hip = true;
        }
    }

    // ==========================================================
    // STAGE 0: PARSE REGION METADATA (BEFORE hsa_init)
    // ==========================================================

    auto meta_blob = read_file(capture_dir + "/memory_regions.json");
    if (meta_blob.empty()) {
        std::cerr << "memory_regions.json not found in " << capture_dir << "\n";
        return 1;
    }

    std::string contents(meta_blob.begin(), meta_blob.end());

    std::vector<RegionMeta> regions;

    nlohmann::json regions_json = nlohmann::json::parse(contents, nullptr, /*exceptions=*/false);
    if (regions_json.is_discarded()) {
        std::cerr << "Failed to parse memory_regions.json\n";
        return 1;
    }

    auto regions_array = regions_json.find("regions");
    if (regions_array == regions_json.end() || !regions_array->is_array()) {
        std::cerr << "memory_regions.json missing \"regions\" array\n";
        return 1;
    }

    for (const auto& entry : *regions_array) {
        uint64_t region_base = entry.at("base").get<uint64_t>();
        size_t   size        = entry.at("size").get<size_t>();

        const size_t page = 4096;
        uint64_t aligned_base = region_base & ~(page - 1);
        uint64_t end_addr     = region_base + size;
        uint64_t aligned_end  = (end_addr + page - 1) & ~(page - 1);
        size_t   aligned_size = aligned_end - aligned_base;
        size_t   offset       = region_base - aligned_base;

        regions.push_back({region_base, size, aligned_base, aligned_size, offset});
    }

    if (regions.empty()) {
        std::cerr << "No memory regions found in memory_regions.json\n";
        return 1;
    }

    std::cerr << "Stage 0: parsed " << regions.size() << " regions\n";
    std::cerr.flush();

    // ==========================================================
    // STAGE 0.5: PRE-MMAP TO STEER ROCr SVM APERTURE
    // ==========================================================

    struct PreMap { void* addr; size_t size; };
    std::vector<PreMap> premaps;

    for (const auto& r : regions) {
        void* addr = mmap(reinterpret_cast<void*>(r.aligned_base),
                          r.aligned_size,
                          PROT_NONE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE,
                          -1,
                          0);
        if (addr != MAP_FAILED) {
            premaps.push_back({addr, r.aligned_size});
        }
    }

    // ==========================================================
    // STAGE 1: INIT HSA
    // ==========================================================

    std::cerr << "Stage 0.5: pre-mmap done (" << premaps.size() << " premaps)\n";
    std::cerr.flush();

    if (hsa_init() != HSA_STATUS_SUCCESS) {
        std::cerr << "hsa_init failed\n";
        return 1;
    }

    std::cerr << "Stage 1: hsa_init done\n";
    std::cerr.flush();

    for (auto& pm : premaps) {
        munmap(pm.addr, pm.size);
    }

    hsa_iterate_agents(find_gpu, nullptr);

    std::cerr << "Stage 1: GPU agent found\n";
    std::cerr.flush();

    ReplayResult result{};
    result.metadata = collect_metadata(g_gpu_agent);
    result.iterations = iterations;
    result.recopy = recopy;

    // ==========================================================
    // STAGE 2: SELECT BACKING POOL
    // ==========================================================

    hsa_amd_memory_pool_t backing_pool{};
    bool found_pool = false;

    auto pool_cb = [](hsa_amd_memory_pool_t pool, void* data) {
        auto* ctx = reinterpret_cast<std::pair<hsa_amd_memory_pool_t*, bool*>*>(data);
        hsa_amd_segment_t segment;
        hsa_amd_memory_pool_get_info(pool,
            HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
            &segment);
        bool alloc_allowed = false;
        hsa_amd_memory_pool_get_info(pool,
            HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
            &alloc_allowed);

        if (segment == HSA_AMD_SEGMENT_GLOBAL && alloc_allowed) {
            *ctx->first = pool;
            *ctx->second = true;
            return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
    };

    std::pair<hsa_amd_memory_pool_t*, bool*> pool_ctx{&backing_pool, &found_pool};
    hsa_amd_agent_iterate_memory_pools(g_gpu_agent, pool_cb, &pool_ctx);

    if (!found_pool) {
        std::cerr << "No suitable memory pool found\n";
        return 1;
    }

    std::cerr << "Stage 2: backing pool found\n";
    std::cerr.flush();

    // ==========================================================
    // STAGE 3: STRICT RESERVE + MAP (NO COPY YET)
    // ==========================================================

    std::vector<RegionRuntime> runtime_regions;

    for (size_t ri = 0; ri < regions.size(); ++ri) {
        const auto& r = regions[ri];

        std::cerr << "Stage 3: region " << ri + 1 << "/" << regions.size()
                  << " base=0x" << std::hex << r.base << std::dec
                  << " size=" << r.size << "\n";
        std::cerr.flush();

        void* reserved = nullptr;

        if (hsa_amd_vmem_address_reserve(&reserved,
                                         r.aligned_size,
                                         r.aligned_base,
                                         0) != HSA_STATUS_SUCCESS ||
            reinterpret_cast<uint64_t>(reserved) != r.aligned_base) {

            std::cerr << "Relocation detected or reserve failed at 0x"
                      << std::hex << r.base << std::dec << "\n";
            return 1;
        }

        hsa_amd_vmem_alloc_handle_t handle{};
        if (hsa_amd_vmem_handle_create(backing_pool,
                                       r.aligned_size,
                                       (hsa_amd_memory_type_t)0,
                                       0,
                                       &handle) != HSA_STATUS_SUCCESS) {
            std::cerr << "vmem_handle_create failed for region 0x"
                      << std::hex << r.base << std::dec << "\n";
            return 1;
        }

        if (hsa_amd_vmem_map(reserved,
                             r.aligned_size,
                             0,
                             handle,
                             0) != HSA_STATUS_SUCCESS) {
            std::cerr << "vmem_map failed for region 0x"
                      << std::hex << r.base << std::dec << "\n";
            return 1;
        }

        hsa_amd_memory_access_desc_t access{};
        access.agent_handle = g_gpu_agent;
        access.permissions = HSA_ACCESS_PERMISSION_RW;
        hsa_amd_vmem_set_access(reserved, r.aligned_size, &access, 1);

        std::stringstream fname;
        fname << capture_dir << "/memory/region_"
              << std::hex << r.base << ".bin";

        std::vector<char> blob = read_file(fname.str());

        if (blob.empty()) {
            std::cerr << "Error: missing or empty memory blob: " << fname.str() << "\n";
            return 1;
        }

        runtime_regions.push_back({
            reserved,
            r.size,
            r.offset,
            r.base,
            std::move(blob)
        });
    }

    // ==========================================================
    // STAGE 4: LOAD EXECUTABLE
    // ==========================================================

    std::string hsaco_path = override_hsaco_path.empty()
        ? (capture_dir + "/kernel.hsaco")
        : override_hsaco_path;

    result.hsaco_path = hsaco_path;

    // Parse dispatch metadata FIRST so we have the mangled name for HIP
    auto dispatch_blob = read_file(capture_dir + "/dispatch.json");
    std::string dcontents(dispatch_blob.begin(), dispatch_blob.end());

    nlohmann::json dispatch = nlohmann::json::parse(dcontents, nullptr, /*exceptions=*/false);
    if (dispatch.is_discarded()) {
        std::cerr << "Failed to parse dispatch.json\n";
        return 1;
    }

    auto get_str = [&](const char* key) -> std::string {
        auto it = dispatch.find(key);
        if (it == dispatch.end() || !it->is_string()) return "unknown";
        return it->get<std::string>();
    };

    auto get_uint = [&](const char* key) -> uint32_t {
        auto it = dispatch.find(key);
        if (it == dispatch.end() || !it->is_number()) return 0;
        return it->get<uint32_t>();
    };

    std::string demangled = get_str("demangled_name");
    std::string mangled   = get_str("mangled_name");

    hipModule_t hip_module = nullptr;
    hipFunction_t hip_function = nullptr;
    hsa_executable_t executable;
    hsa_executable_symbol_t kernel_symbol{};

    if (use_hip) {
        if (hipInit(0) != hipSuccess) {
             std::cerr << "hipInit failed\n";
             return 1;
        }
        hipDevice_t device;
        if (hipDeviceGet(&device, 0) != hipSuccess) {
             std::cerr << "hipDeviceGet failed\n";
             return 1;
        }
        if (hipSetDevice(0) != hipSuccess) {
             std::cerr << "hipSetDevice failed\n";
             return 1;
        }
        std::vector<char> hsaco = read_file(hsaco_path);
        if (hsaco.empty()) {
            std::cerr << "Failed to open HSACO: " << hsaco_path << "\n";
            return 1;
        }
        if (hipModuleLoadData(&hip_module, hsaco.data()) != hipSuccess) {
            std::cerr << "Failed to load module via HIP: " << hsaco_path << "\n";
            std::cerr << "Error: " << hipGetErrorString(hipGetLastError()) << "\n";
            return 1;
        }
        if (hipModuleGetFunction(&hip_function, hip_module, mangled.c_str()) != hipSuccess) {
            std::cerr << "Failed to get function via HIP: " << mangled << "\n";
            std::cerr << "Error: " << hipGetErrorString(hipGetLastError()) << "\n";
            return 1;
        }
        std::cerr << "Loaded kernel via HIP: " << mangled << "\n";
    } else {
        std::vector<char> hsaco = read_file(hsaco_path);
        if (hsaco.empty()) {
            std::cerr << "Failed to open HSACO: " << hsaco_path << "\n";
            return 1;
        }

        hsa_code_object_reader_t reader;
        hsa_code_object_reader_create_from_memory(
            hsaco.data(), hsaco.size(), &reader);

        hsa_executable_create(HSA_PROFILE_FULL,
                              HSA_EXECUTABLE_STATE_UNFROZEN,
                              nullptr,
                              &executable);

        hsa_executable_load_agent_code_object(
            executable, g_gpu_agent, reader, nullptr, nullptr);

        hsa_executable_freeze(executable, nullptr);

        // Resolve the kernel symbol by mangled name from dispatch.json.
        // The HSACO may contain many kernel variants (e.g., template
        // instantiations); we must select the exact one that was dispatched.
        bool symbol_found = false;

        if (mangled != "unknown" && !mangled.empty()) {
            std::string kd_name = mangled + ".kd";
            hsa_status_t sym_st = hsa_executable_get_symbol_by_name(
                executable, kd_name.c_str(), &g_gpu_agent, &kernel_symbol);
            if (sym_st == HSA_STATUS_SUCCESS) {
                symbol_found = true;
                std::cerr << "Resolved kernel by name: " << mangled << "\n";
            } else {
                std::cerr << "Warning: could not resolve '" << kd_name
                          << "', falling back to symbol iteration\n";
            }
        }

        if (!symbol_found) {
            hsa_executable_iterate_symbols(executable,
                [](hsa_executable_t,
                   hsa_executable_symbol_t sym,
                   void* data) -> hsa_status_t {
                    uint32_t type;
                    hsa_executable_symbol_get_info(sym,
                        HSA_EXECUTABLE_SYMBOL_INFO_TYPE,
                        &type);
                    if (type == HSA_SYMBOL_KIND_KERNEL) {
                        *reinterpret_cast<hsa_executable_symbol_t*>(data) = sym;
                        return HSA_STATUS_INFO_BREAK;
                    }
                    return HSA_STATUS_SUCCESS;
                },
                &kernel_symbol);
            std::cerr << "Warning: using first kernel symbol (HSACO has multiple kernels)\n";
        }
    }

    uint64_t kernel_object = 0;
    uint32_t kernarg_size = 0;
    uint32_t group_segment = 0;
    uint32_t private_segment = 0;

    if (use_hip) {
        // We still need these sizes. For HIP, we can get them from the dispatch.json
        // or query attributes if needed, but dispatch.json is the source of truth for replay.
        kernarg_size = get_uint("kernarg_size");
        group_segment = get_uint("group_segment_size");
        private_segment = get_uint("private_segment_size");
    } else {
        hsa_executable_symbol_get_info(kernel_symbol,
            HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
            &kernel_object);
        hsa_executable_symbol_get_info(kernel_symbol,
            HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
            &kernarg_size);
        hsa_executable_symbol_get_info(kernel_symbol,
            HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
            &group_segment);
        hsa_executable_symbol_get_info(kernel_symbol,
            HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
            &private_segment);
    }

    uint32_t grid[3] = {1, 1, 1};
    uint32_t block[3] = {1, 1, 1};
    if (auto it = dispatch.find("grid"); it != dispatch.end() && it->is_array()) {
        for (int i = 0; i < 3 && i < static_cast<int>(it->size()); ++i)
            grid[i] = (*it)[i].get<uint32_t>();
    }
    if (auto it = dispatch.find("block"); it != dispatch.end() && it->is_array()) {
        for (int i = 0; i < 3 && i < static_cast<int>(it->size()); ++i)
            block[i] = (*it)[i].get<uint32_t>();
    }

    uint32_t dispatch_group_seg   = get_uint("group_segment_size");
    uint32_t dispatch_private_seg = get_uint("private_segment_size");
    if (dispatch_group_seg > group_segment)
        group_segment = dispatch_group_seg;
    if (dispatch_private_seg > private_segment)
        private_segment = dispatch_private_seg;

    if (demangled != "unknown") {
        auto pos = demangled.find(" [");
        if (pos != std::string::npos) {
            result.kernel_name = demangled.substr(0, pos);
        } else {
            result.kernel_name = demangled;
        }
    } else if (mangled != "unknown") {
        result.kernel_name = mangled;
    } else {
        result.kernel_name = "unknown";
    }

    // ==========================================================
    // STAGE 5: MULTI-ITERATION DISPATCH WITH PROFILING
    // ==========================================================

    auto restore_memory = [&]() {
        for (auto& rr : runtime_regions) {
            void* dst = static_cast<void*>(
                static_cast<uint8_t*>(rr.reserved) + rr.offset);
            hsa_memory_copy(dst, rr.blob.data(), rr.size);
        }
    };

    void* kernarg = nullptr;
    hsa_amd_memory_pool_allocate(backing_pool,
                                 kernarg_size,
                                 0,
                                 &kernarg);

    std::vector<char> kblob = read_file(capture_dir + "/kernarg.bin");
    // kernarg_size comes from the loaded HSACO symbol; kblob is from the captured
    // kernarg.bin. These should always match, but can diverge when --hsaco points
    // to a recompiled variant. Guard against overflow (kblob too large) and leave
    // any trailing padding bytes zeroed (kblob too small).
    if (kblob.size() != kernarg_size) {
        std::cerr << "Warning: kernarg.bin size (" << kblob.size()
                  << ") != HSACO kernarg_size (" << kernarg_size
                  << "); kernel ABI may not match captured data\n";
    }
    memcpy(kernarg, kblob.data(), std::min(kblob.size(), (size_t)kernarg_size));

    result.raw_ns.clear();
    result.raw_ns.reserve(iterations);

    if (use_hip) {
        int static_shared = 0;
        hipFuncGetAttribute(&static_shared, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, hip_function);
        int dynamic_shared = (int)group_segment - static_shared;
        if (dynamic_shared < 0) dynamic_shared = 0;

        size_t kernarg_size_st = kernarg_size;
        void* extra[] = {
            HIP_LAUNCH_PARAM_BUFFER_POINTER,
            kernarg,
            HIP_LAUNCH_PARAM_BUFFER_SIZE,
            &kernarg_size_st,
            HIP_LAUNCH_PARAM_END
        };

        // HSA grid sizes are total work-items; HIP expects block counts
        uint32_t blocks[3] = {
            grid[0] / block[0],
            grid[1] / block[1],
            grid[2] / block[2],
        };

        std::cerr << "Dispatch config (HIP):\n"
                  << "  grid: " << grid[0] << " x " << grid[1] << " x " << grid[2]
                  << " (" << blocks[0] << " x " << blocks[1] << " x " << blocks[2] << " blocks)\n"
                  << "  block: " << block[0] << " x " << block[1] << " x " << block[2] << "\n"
                  << "  group_segment: " << group_segment << "\n"
                  << "  dynamic_shared: " << dynamic_shared << "\n"
                  << "  kernarg_size: " << kernarg_size << "\n"
                  << "  regions: " << runtime_regions.size() << "\n";

        restore_memory();

        for (size_t iter = 0; iter < iterations; ++iter) {
            if (iter > 0 && recopy) {
                restore_memory();
            }

            hipModuleLaunchKernel(
                hip_function,
                blocks[0], blocks[1], blocks[2],
                block[0], block[1], block[2],
                dynamic_shared,
                0, // stream
                nullptr, // kernelParams
                extra
            );

            hipDeviceSynchronize();
        }

    } else {
        hsa_queue_t* queue = nullptr;
        hsa_queue_create(g_gpu_agent,
                         128,
                         HSA_QUEUE_TYPE_MULTI,
                         nullptr,
                         nullptr,
                         private_segment,
                         group_segment,
                         &queue);

        hsa_status_t pst = hsa_amd_profiling_set_profiler_enabled(queue, 1);
        if (pst != HSA_STATUS_SUCCESS) {
            std::cerr << "Failed to enable profiling\n";
            return 1;
        }

        hsa_signal_t completion_signal;
        hsa_status_t ss = hsa_signal_create(1, 0, nullptr, &completion_signal);

        if (ss != HSA_STATUS_SUCCESS) {
            std::cerr << "Failed to create completion signal\n";
            return 1;
        }

        std::cerr << "Dispatch config:\n"
                  << "  grid: " << grid[0] << " x " << grid[1] << " x " << grid[2] << "\n"
                  << "  block: " << block[0] << " x " << block[1] << " x " << block[2] << "\n"
                  << "  dimensions: " << (grid[2] > 1 ? 3 : (grid[1] > 1 ? 2 : 1)) << "\n"
                  << "  group_segment: " << group_segment << "\n"
                  << "  private_segment: " << private_segment << "\n"
                  << "  kernarg_size: " << kernarg_size << "\n"
                  << "  kernel_object: 0x" << std::hex << kernel_object << std::dec << "\n"
                  << "  kernarg blob: " << kblob.size() << " bytes\n"
                  << "  regions: " << runtime_regions.size() << "\n";

        restore_memory();

        for (size_t iter = 0; iter < iterations; ++iter) {

            if (iter > 0 && recopy) {
                restore_memory();
            }

            uint64_t index = hsa_queue_load_write_index_relaxed(queue);
            auto* packet = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(
                queue->base_address) + (index % queue->size);

            memset(packet, 0, sizeof(*packet));

            uint32_t dims = 1;
            if (grid[2] > 1 || block[2] > 1) dims = 3;
            else if (grid[1] > 1 || block[1] > 1) dims = 2;

            packet->setup = dims << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
            packet->workgroup_size_x = block[0];
            packet->workgroup_size_y = block[1];
            packet->workgroup_size_z = block[2];
            packet->grid_size_x = grid[0];
            packet->grid_size_y = grid[1];
            packet->grid_size_z = grid[2];
            packet->kernel_object = kernel_object;
            packet->kernarg_address = kernarg;
            packet->private_segment_size = private_segment;
            packet->group_segment_size = group_segment;
            packet->completion_signal = completion_signal;

            uint16_t header =
                (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                (1 << HSA_PACKET_HEADER_BARRIER);
            packet->header = header;

            hsa_queue_store_write_index_relaxed(queue, index + 1);
            hsa_signal_store_relaxed(queue->doorbell_signal, index);

            while (hsa_signal_wait_relaxed(
                       completion_signal,
                       HSA_SIGNAL_CONDITION_EQ,
                       0,
                       UINT64_MAX,
                       HSA_WAIT_STATE_ACTIVE) != 0) {}

            hsa_amd_profiling_dispatch_time_t time{};
            hsa_status_t dt = hsa_amd_profiling_get_dispatch_time(
                g_gpu_agent,
                completion_signal,
                &time);

            if (dt != HSA_STATUS_SUCCESS) {
                std::cerr << "Failed to get dispatch time\n";
                return 1;
            }

            result.raw_ns.push_back(time.end - time.start);

            hsa_signal_store_relaxed(completion_signal, 1);
        }
    }

    // ==========================================================
    // STAGE 6: DUMP OUTPUT AND REPORT
    // ==========================================================

    if (dump_output) {
        std::string output_dir = capture_dir + "/output";
        dump_output_regions(output_dir, runtime_regions);
        if (!json_output) {
            std::cerr << "Dumped " << runtime_regions.size()
                      << " output regions to " << output_dir << "\n";
        }
    }

    if (!result.raw_ns.empty()) {
        uint64_t min = result.raw_ns[0];
        uint64_t max = result.raw_ns[0];
        uint64_t sum = 0;

        for (auto d : result.raw_ns) {
            if (d < min) min = d;
            if (d > max) max = d;
            sum += d;
        }

        double avg_ns = double(sum) / result.raw_ns.size();

        result.avg_us = avg_ns / 1000.0;
        result.min_us = min / 1000.0;
        result.max_us = max / 1000.0;
    }

    if (json_output) {
        print_json_output(result);
    } else {
        std::cout << "Kernel:     " << result.kernel_name << "\n";
        std::cout << "Grid:       " << grid[0] << " x " << grid[1] << " x " << grid[2] << "\n";
        std::cout << "Block:      " << block[0] << " x " << block[1] << " x " << block[2] << "\n";
        std::cout << "Iterations: " << iterations << "\n";
        std::cout << "Mode:       " << (recopy ? "stateless" : "stateful") << "\n";
        if (!result.raw_ns.empty()) {
            std::cout << "Average GPU time: " << result.avg_us << " us\n";
            std::cout << "Min: " << result.min_us << " us\n";
            std::cout << "Max: " << result.max_us << " us\n";
        }
    }

    hsa_shut_down();
    return 0;
}
