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
 *   - hsa_code_object_reader_create_from_memory -> patches code objects
 *   - OnSubmitPackets                 -> repacks kernargs with probe buffer
 ******************************************************************************/

#include "proboscis/interceptor.h"
#include "proboscis/proboscis.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <cxxabi.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

// ---- Forward declarations for probe buffer / kernarg repack ----------------

extern void initProbeBuffer(void* buffer, uint64_t max_records, uint64_t record_size);
extern size_t computeProbeBufferSize(uint64_t max_records, uint64_t record_size);
extern void repackInstrumentedKernArgs(void* dst, const void* src,
                                       void* probe_ctx, const ArgDescriptor& desc);

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

// We only define fields we access by name. Fields beyond 94 are accessed
// via core_field() helper to avoid a massive struct.
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
    // Fields 95+ accessed via core_field() below.
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
static constexpr size_t FIELD_CODE_OBJ_READER_CREATE_FROM_MEMORY = 117;

// ---- Saved original HSA function pointers ---------------------------------

static decltype(hsa_queue_create)* orig_hsa_queue_create = nullptr;
static decltype(hsa_queue_destroy)* orig_hsa_queue_destroy = nullptr;
static decltype(hsa_amd_queue_intercept_create)* orig_intercept_create = nullptr;
static decltype(hsa_amd_queue_intercept_register)* orig_intercept_register = nullptr;
static decltype(hsa_executable_symbol_get_info)* orig_symbol_get_info = nullptr;
static decltype(hsa_executable_get_symbol_by_name)* orig_get_symbol_by_name = nullptr;

using code_obj_reader_create_from_memory_t =
    hsa_status_t (*)(const void*, size_t, hsa_code_object_reader_t*);
static code_obj_reader_create_from_memory_t orig_code_obj_reader_create_from_memory = nullptr;

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
    config_.max_records = 10000;
    config_.sample_rate = 1;

    if (config_path) {
        std::ifstream f(config_path);
        if (f.good()) {
            std::ostringstream ss;
            ss << f.rdbuf();
            std::string content = ss.str();

            auto extract_str = [&content](const std::string& key) -> std::string {
                auto pos = content.find("\"" + key + "\"");
                if (pos == std::string::npos) return "";
                pos = content.find("\"", pos + key.size() + 2);
                if (pos == std::string::npos) return "";
                auto end = content.find("\"", pos + 1);
                if (end == std::string::npos) return "";
                return content.substr(pos + 1, end - pos - 1);
            };
            auto extract_int = [&content](const std::string& key, int def) -> int {
                auto pos = content.find("\"" + key + "\"");
                if (pos == std::string::npos) return def;
                pos = content.find(":", pos);
                if (pos == std::string::npos) return def;
                pos++;
                while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t')) pos++;
                return std::atoi(content.c_str() + pos);
            };

            config_.probe_type = extract_str("probe_type");
            config_.target_kernel = extract_str("target_kernel");
            config_.max_records = extract_int("max_records", 10000);
            config_.sample_rate = extract_int("sample_rate", 1);

            if (config_.log_level >= 2) {
                std::cerr << "[proboscis] Config: probe_type=" << config_.probe_type
                          << " target=" << config_.target_kernel
                          << " max_records=" << config_.max_records
                          << " sample_rate=" << config_.sample_rate << std::endl;
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

// ---- Code object patching -------------------------------------------------
// Minimal in-memory ELF surgery: parse .note section, find AMDGPU metadata,
// add hidden_proboscis_ctx to each kernel's args, update kernarg_segment_size.
// Uses the Python msgpack codec via an embedded minimal implementation.

// Align value up to boundary.
static size_t align_up(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

// Minimal msgpack reading helpers (read-only, enough for AMDGPU metadata).
// The metadata is a map with "amdhsa.kernels" containing an array of maps.
// We don't need a full msgpack parser — we use Python for the actual patching
// and call it via subprocess.

bool ProboscisInterceptor::patchCodeObject(
    const void* code_object, size_t size, std::vector<uint8_t>& patched)
{
    // Write code object to temp file
    char tmpco[] = "/tmp/proboscis_co_XXXXXX";
    int fd = mkstemp(tmpco);
    if (fd < 0) return false;
    write(fd, code_object, size);
    close(fd);

    // Write plan output path
    char tmpplan[] = "/tmp/proboscis_plan_XXXXXX";
    int fd2 = mkstemp(tmpplan);
    if (fd2 < 0) { unlink(tmpco); return false; }
    close(fd2);

    // Call Python patcher
    std::ostringstream cmd;
    cmd << "python3 -m proboscis.patcher " << tmpco << " " << tmpplan;
    if (!config_.target_kernel.empty()) {
        cmd << " --target " << config_.target_kernel;
    }

    if (config_.log_level >= 2) {
        std::cerr << "[proboscis] Patching code object: " << cmd.str() << std::endl;
    }

    int ret = system(cmd.str().c_str());
    if (ret != 0) {
        if (config_.log_level >= 1) {
            std::cerr << "[proboscis] Patcher returned " << ret << std::endl;
        }
        unlink(tmpco);
        unlink(tmpplan);
        return false;
    }

    // Read patched code object
    std::ifstream pf(tmpco, std::ios::binary | std::ios::ate);
    if (!pf.good()) { unlink(tmpco); unlink(tmpplan); return false; }
    size_t patched_size = pf.tellg();
    pf.seekg(0);
    patched.resize(patched_size);
    pf.read(reinterpret_cast<char*>(patched.data()), patched_size);
    pf.close();

    // Read plan JSON (arg descriptors per kernel)
    std::ifstream planf(tmpplan);
    if (planf.good()) {
        std::ostringstream ss;
        ss << planf.rdbuf();
        std::string plan_json = ss.str();
        planf.close();

        // Simple JSON parsing for plan output:
        // [{"kernel":"name", "orig_size":N, "new_size":M, "probe_ctx_offset":O, "explicit_len":E}, ...]
        // Parse each entry
        size_t pos = 0;
        while ((pos = plan_json.find("\"kernel\"", pos)) != std::string::npos) {
            auto extract_str = [&](const std::string& key) -> std::string {
                auto p = plan_json.find("\"" + key + "\"", pos);
                if (p == std::string::npos) return "";
                p = plan_json.find("\"", p + key.size() + 2);
                if (p == std::string::npos) return "";
                auto e = plan_json.find("\"", p + 1);
                if (e == std::string::npos) return "";
                return plan_json.substr(p + 1, e - p - 1);
            };
            auto extract_int = [&](const std::string& key) -> uint32_t {
                auto p = plan_json.find("\"" + key + "\"", pos);
                if (p == std::string::npos) return 0;
                p = plan_json.find(":", p);
                if (p == std::string::npos) return 0;
                p++;
                while (p < plan_json.size() && plan_json[p] == ' ') p++;
                return std::atoi(plan_json.c_str() + p);
            };

            std::string kernel_name = extract_str("kernel");
            if (!kernel_name.empty()) {
                ArgDescriptor desc{};
                desc.explicit_args_length = extract_int("explicit_len");
                desc.kernarg_length = extract_int("orig_size");
                desc.instrumented_kernarg_length = extract_int("new_size");
                desc.probe_ctx_offset = extract_int("probe_ctx_offset");
                desc.hidden_args_length = desc.kernarg_length - desc.explicit_args_length;

                std::lock_guard<std::mutex> lock(mutex_);
                kernel_arg_descs_[kernel_name] = desc;

                if (config_.log_level >= 1) {
                    std::cerr << "[proboscis] Patched kernel: " << kernel_name
                              << " kernarg " << desc.kernarg_length
                              << " -> " << desc.instrumented_kernarg_length
                              << " probe_ctx@" << desc.probe_ctx_offset << std::endl;
                }
            }
            pos++;
        }
    }

    unlink(tmpco);
    unlink(tmpplan);
    return true;
}

// ---- Static instruction analysis ------------------------------------------
// Runs llvm-objdump on the code object and counts memory instructions.

void ProboscisInterceptor::analyzeInstructions(
    const std::string& kernel_name, const void* code_object, size_t size)
{
    // Write code object to temp file
    char tmpco[] = "/tmp/proboscis_disasm_XXXXXX";
    int fd = mkstemp(tmpco);
    if (fd < 0) return;
    write(fd, code_object, size);
    close(fd);

    // Run llvm-objdump (try several paths)
    std::string objdump;
    const char* rocm = std::getenv("ROCM_PATH");
    if (!rocm) rocm = "/opt/rocm";
    std::vector<std::string> candidates = {
        std::string(rocm) + "/llvm/bin/llvm-objdump",
        "/opt/rocm/llvm/bin/llvm-objdump",
        "llvm-objdump",
    };
    for (const auto& c : candidates) {
        std::string test_cmd = c + " --version > /dev/null 2>&1";
        if (system(test_cmd.c_str()) == 0) {
            objdump = c;
            break;
        }
    }
    if (objdump.empty()) {
        if (config_.log_level >= 1)
            std::cerr << "[proboscis] llvm-objdump not found, skipping analysis" << std::endl;
        unlink(tmpco);
        return;
    }

    char tmpout[] = "/tmp/proboscis_asm_XXXXXX";
    int fd2 = mkstemp(tmpout);
    if (fd2 < 0) { unlink(tmpco); return; }
    close(fd2);

    std::ostringstream cmd;
    cmd << objdump << " -d " << tmpco << " > " << tmpout << " 2>/dev/null";
    if (config_.log_level >= 2) {
        std::cerr << "[proboscis] Running: " << cmd.str() << std::endl;
    }
    int ret = system(cmd.str().c_str());
    if (ret != 0) {
        if (config_.log_level >= 1) {
            std::cerr << "[proboscis] llvm-objdump failed (ret=" << ret << ")" << std::endl;
        }
        unlink(tmpco);
        unlink(tmpout);
        return;
    }

    // Parse disassembly output
    std::ifstream disasm(tmpout);
    if (!disasm.good()) { unlink(tmpco); unlink(tmpout); return; }

    InstructionAnalysis analysis{};
    std::string line;
    // We parse the entire code object — if there are multiple kernels, we aggregate.
    // A more precise approach would track kernel boundaries via labels.
    while (std::getline(disasm, line)) {
        // Trim leading whitespace
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        std::string trimmed = line.substr(start);

        // Skip non-instruction lines
        if (trimmed.empty() || trimmed[0] == ';' || trimmed[0] == '.' || trimmed[0] == '<') continue;

        // llvm-objdump AMDGPU format with raw bytes:
        //   "  100:\t01 02 03 04\t\tglobal_load_dword v0, v[0:1], off"
        // The mnemonic starts after the last \t sequence.
        auto colon = trimmed.find(':');
        if (colon == std::string::npos) continue;
        std::string after_colon = trimmed.substr(colon + 1);

        // Find the last group of text — that's the mnemonic + operands.
        // llvm-objdump uses \t to separate fields.
        // Strategy: find the last non-hex-looking word as the mnemonic.
        // Simpler: scan for known instruction prefixes in the whole line.
        std::string insn;
        for (const char* prefix : {"global_load", "global_store", "global_atomic",
                                    "flat_load", "flat_store", "flat_atomic",
                                    "buffer_load", "buffer_store", "buffer_atomic",
                                    "ds_read", "ds_write", "ds_add", "ds_inc", "ds_cmpst"}) {
            auto pos = after_colon.find(prefix);
            if (pos != std::string::npos) {
                insn = after_colon.substr(pos);
                break;
            }
        }
        if (insn.empty()) continue;

        // Determine access size from suffix
        auto classify_size = [](const std::string& inst, bool is_load,
                               InstructionAnalysis& a) {
            if (inst.find("dwordx4") != std::string::npos ||
                inst.find("b128") != std::string::npos) {
                if (is_load) a.loads_16b++; else a.stores_16b++;
            } else if (inst.find("dwordx3") != std::string::npos ||
                       inst.find("b96") != std::string::npos) {
                if (is_load) a.loads_12b++; else a.stores_12b++;
            } else if (inst.find("dwordx2") != std::string::npos ||
                       inst.find("b64") != std::string::npos ||
                       inst.find("d16_hi") != std::string::npos) {
                if (is_load) a.loads_8b++; else a.stores_8b++;
            } else if (inst.find("short") != std::string::npos ||
                       inst.find("b16") != std::string::npos ||
                       inst.find("u16") != std::string::npos) {
                if (is_load) a.loads_2b++; else a.stores_2b++;
            } else if (inst.find("ubyte") != std::string::npos ||
                       inst.find("sbyte") != std::string::npos ||
                       inst.find("b8") != std::string::npos) {
                if (is_load) a.loads_1b++; else a.stores_1b++;
            } else {
                // Default: dword (4 bytes)
                if (is_load) a.loads_4b++; else a.stores_4b++;
            }
        };

        // Classify instruction
        if (insn.find("global_load") == 0) {
            analysis.global_loads++;
            classify_size(insn, true, analysis);
        } else if (insn.find("global_store") == 0) {
            analysis.global_stores++;
            classify_size(insn, false, analysis);
        } else if (insn.find("flat_load") == 0) {
            analysis.flat_loads++;
            classify_size(insn, true, analysis);
        } else if (insn.find("flat_store") == 0) {
            analysis.flat_stores++;
            classify_size(insn, false, analysis);
        } else if (insn.find("buffer_load") == 0) {
            analysis.buffer_loads++;
            classify_size(insn, true, analysis);
        } else if (insn.find("buffer_store") == 0) {
            analysis.buffer_stores++;
            classify_size(insn, false, analysis);
        } else if (insn.find("ds_read") == 0) {
            analysis.ds_reads++;
            classify_size(insn, true, analysis);
        } else if (insn.find("ds_write") == 0) {
            analysis.ds_writes++;
            classify_size(insn, false, analysis);
        } else if (insn.find("global_atomic") == 0 ||
                   insn.find("flat_atomic") == 0 ||
                   insn.find("buffer_atomic") == 0 ||
                   insn.find("ds_add") == 0 ||
                   insn.find("ds_inc") == 0 ||
                   insn.find("ds_cmpst") == 0) {
            analysis.atomics++;
        }
    }

    disasm.close();
    unlink(tmpco);
    unlink(tmpout);

    if (analysis.total_loads() + analysis.total_stores() + analysis.atomics > 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        // Store under a generic key since we analyze the whole code object.
        // We'll match kernel names at results time.
        instruction_analysis_["__code_object__"] = analysis;

        if (config_.log_level >= 1) {
            std::cerr << "[proboscis] Instruction analysis: "
                      << analysis.total_loads() << " loads, "
                      << analysis.total_stores() << " stores, "
                      << analysis.atomics << " atomics"
                      << " (loads >4B: " << analysis.loads_gt_4b() << ")" << std::endl;
        }
    }
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
    hsa_queue_t* queue,
    const hsa_kernel_dispatch_packet_t* packets,
    uint64_t count,
    hsa_amd_queue_intercept_packet_writer writer)
{
    // We need a mutable copy of the packets if we're going to repack
    std::vector<hsa_kernel_dispatch_packet_t> modified_packets(packets, packets + count);
    bool any_modified = false;

    for (uint64_t i = 0; i < count; i++) {
        auto& pkt = modified_packets[i];
        uint8_t pkt_type = pkt.header & 0xFF;
        if (pkt_type != HSA_PACKET_TYPE_KERNEL_DISPATCH) continue;

        std::string name = getKernelName(pkt.kernel_object);
        if (!shouldInstrument(name)) continue;

        // Record dispatch metadata
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

        // Check if we have an arg descriptor for this kernel.
        // kernel_arg_descs_ keys are mangled names from the code object metadata
        // (e.g. "_Z7vec_addPKfS0_Pfi"), but dispatch names are demangled
        // (e.g. "vec_add(float const*, float const*, float*, int)").
        // Also try matching with ".kd" suffix stripped.
        ArgDescriptor desc{};
        bool has_desc = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto& kv : kernel_arg_descs_) {
                // Try: mangled name contains demangled, or vice versa
                if (name.find(kv.first) != std::string::npos ||
                    kv.first.find(name) != std::string::npos) {
                    desc = kv.second;
                    has_desc = true;
                    break;
                }
                // Try demangling the stored key and comparing
                std::string demangled_key = demangle(kv.first.c_str());
                if (name.find(demangled_key) != std::string::npos ||
                    demangled_key.find(name) != std::string::npos) {
                    desc = kv.second;
                    has_desc = true;
                    break;
                }
                // Try with .kd suffix stripped from dispatch name
                std::string name_stripped = name;
                if (name_stripped.size() > 3 &&
                    name_stripped.substr(name_stripped.size() - 3) == ".kd") {
                    name_stripped = name_stripped.substr(0, name_stripped.size() - 3);
                }
                if (name_stripped.find(kv.first) != std::string::npos ||
                    kv.first.find(name_stripped) != std::string::npos) {
                    desc = kv.second;
                    has_desc = true;
                    break;
                }
            }
        }

        if (has_desc) {
            // Allocate probe buffer
            hsa_agent_t agent{};
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto q = queues_.find(queue);
                if (q != queues_.end()) agent = q->second;
            }

            uint64_t record_size = 24; // default for memory_trace
            if (config_.probe_type == "block_count") record_size = 16;
            else if (config_.probe_type == "register_snapshot") record_size = 16;

            size_t buf_size = computeProbeBufferSize(config_.max_records, record_size);
            void* probe_buf = allocateProbeBuffer(agent, buf_size);

            if (probe_buf) {
                initProbeBuffer(probe_buf, config_.max_records, record_size);

                // Allocate new kernarg buffer from the GPU agent's global pool.
                void* new_kernarg = nullptr;
                hsa_amd_memory_pool_t pool{};
                hsa_amd_agent_iterate_memory_pools(agent,
                    [](hsa_amd_memory_pool_t p, void* data) -> hsa_status_t {
                        hsa_amd_segment_t seg;
                        hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
                        if (seg == HSA_AMD_SEGMENT_GLOBAL) {
                            bool accessible = false;
                            hsa_amd_memory_pool_get_info(p,
                                HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &accessible);
                            if (accessible) {
                                *static_cast<hsa_amd_memory_pool_t*>(data) = p;
                                return HSA_STATUS_INFO_BREAK;
                            }
                        }
                        return HSA_STATUS_SUCCESS;
                    }, &pool);

                if (hsa_amd_memory_pool_allocate(pool, desc.instrumented_kernarg_length,
                                                  0, &new_kernarg) == HSA_STATUS_SUCCESS) {
                    repackInstrumentedKernArgs(new_kernarg, pkt.kernarg_address,
                                               probe_buf, desc);

                    // Update the dispatch packet to use the new kernarg
                    pkt.kernarg_address = new_kernarg;
                    any_modified = true;

                    // Track the probe buffer for readback
                    ProbeBufferInfo pbi{};
                    pbi.buffer = probe_buf;
                    pbi.size = buf_size;
                    pbi.max_records = config_.max_records;
                    pbi.record_size = record_size;
                    pbi.agent = agent;

                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        probe_buffer_records_[name].push_back(pbi);
                    }

                    if (config_.log_level >= 2) {
                        std::cerr << "[proboscis] Repacked kernarg for " << name
                                  << " probe_buf=" << probe_buf
                                  << " new_kernarg=" << new_kernarg << std::endl;
                    }
                } else {
                    if (config_.log_level >= 1) {
                        std::cerr << "[proboscis] Failed to allocate kernarg buffer for "
                                  << name << std::endl;
                    }
                }
            } else {
                if (config_.log_level >= 1) {
                    std::cerr << "[proboscis] Failed to allocate probe buffer for "
                              << name << std::endl;
                }
            }
        }

        if (config_.log_level >= 2) {
            std::cerr << "[proboscis] Dispatch: " << name
                      << " grid=[" << pkt.grid_size_x << "," << pkt.grid_size_y
                      << "," << pkt.grid_size_z << "]"
                      << " wg=[" << pkt.workgroup_size_x << "," << pkt.workgroup_size_y
                      << "," << pkt.workgroup_size_z << "]"
                      << (has_desc ? " [instrumented]" : " [passthrough]") << std::endl;
        }
    }

    writer(modified_packets.data(), count);
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
          << ", \"total_threads\": " << total_threads;

        // Add instrumentation info if kernel was patched
        auto desc_it = kernel_arg_descs_.find(kernel_name);
        if (desc_it == kernel_arg_descs_.end()) {
            // Try substring match
            for (const auto& d : kernel_arg_descs_) {
                if (kernel_name.find(d.first) != std::string::npos ||
                    d.first.find(kernel_name) != std::string::npos) {
                    desc_it = kernel_arg_descs_.find(d.first);
                    break;
                }
            }
        }

        if (desc_it != kernel_arg_descs_.end()) {
            f << ", \"instrumented\": true"
              << ", \"kernarg_original\": " << desc_it->second.kernarg_length
              << ", \"kernarg_instrumented\": " << desc_it->second.instrumented_kernarg_length
              << ", \"probe_ctx_offset\": " << desc_it->second.probe_ctx_offset;
        }

        // Add probe buffer readback data
        auto pbr_it = probe_buffer_records_.find(kernel_name);
        if (pbr_it != probe_buffer_records_.end() && !pbr_it->second.empty()) {
            f << ", \"probe_buffers\": [";
            for (size_t pi = 0; pi < pbr_it->second.size(); pi++) {
                const auto& pbi = pbr_it->second[pi];
                if (pi > 0) f << ", ";

                // Read back probe buffer header
                auto* hdr = static_cast<ProbeBufferHeader*>(pbi.buffer);
                f << "{\"write_offset\": " << hdr->write_offset
                  << ", \"max_records\": " << hdr->max_records
                  << ", \"record_size\": " << hdr->record_size
                  << ", \"records_written\": "
                  << std::min(hdr->write_offset, hdr->max_records) << "}";
            }
            f << "]";
        }

        f << "}";

        // Add static instruction analysis
        // Look for analysis matching this kernel or the global code object analysis
        auto ia_it = instruction_analysis_.find(kernel_name);
        if (ia_it == instruction_analysis_.end()) {
            ia_it = instruction_analysis_.find("__code_object__");
        }
        if (ia_it != instruction_analysis_.end()) {
            const auto& ia = ia_it->second;
            f << ", \"instruction_analysis\": {"
              << "\"total_loads\": " << ia.total_loads()
              << ", \"total_stores\": " << ia.total_stores()
              << ", \"atomics\": " << ia.atomics
              << ", \"global_loads\": " << ia.global_loads
              << ", \"global_stores\": " << ia.global_stores
              << ", \"flat_loads\": " << ia.flat_loads
              << ", \"flat_stores\": " << ia.flat_stores
              << ", \"buffer_loads\": " << ia.buffer_loads
              << ", \"buffer_stores\": " << ia.buffer_stores
              << ", \"ds_reads\": " << ia.ds_reads
              << ", \"ds_writes\": " << ia.ds_writes
              << ", \"loads_by_size\": {"
              << "\"1B\": " << ia.loads_1b
              << ", \"2B\": " << ia.loads_2b
              << ", \"4B\": " << ia.loads_4b
              << ", \"8B\": " << ia.loads_8b
              << ", \"12B\": " << ia.loads_12b
              << ", \"16B\": " << ia.loads_16b
              << "}, \"stores_by_size\": {"
              << "\"1B\": " << ia.stores_1b
              << ", \"2B\": " << ia.stores_2b
              << ", \"4B\": " << ia.stores_4b
              << ", \"8B\": " << ia.stores_8b
              << ", \"12B\": " << ia.stores_12b
              << ", \"16B\": " << ia.stores_16b
              << "}, \"loads_gt_4B\": " << ia.loads_gt_4b()
              << ", \"stores_gt_4B\": " << ia.stores_gt_4b()
              << "}";
        }

        f << "}";
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

hsa_status_t ProboscisInterceptor::hsa_code_object_reader_create_from_memory(
    const void* code_object, size_t size,
    hsa_code_object_reader_t* code_object_reader)
{
    auto* self = getInstance();

    // Check if this is an AMDGPU code object (ELF magic + ELFOSABI_AMDGPU_HSA)
    bool is_amdgpu = false;
    if (size >= 8) {
        const uint8_t* bytes = static_cast<const uint8_t*>(code_object);
        is_amdgpu = (bytes[0] == 0x7f && bytes[1] == 'E' &&
                     bytes[2] == 'L' && bytes[3] == 'F');
    }

    if (is_amdgpu && !self->shutting_down_.load()) {
        if (self->config_.log_level >= 2) {
            std::cerr << "[proboscis] Intercepted code object (" << size << " bytes)"
                      << std::endl;
        }

        // Run static instruction analysis on ALL code objects
        self->analyzeInstructions("__code_object__", code_object, size);

        // Only patch code objects that are NOT from the HIP runtime.
        // HIP runtime code objects (containing __amd_rocclr_*) are loaded
        // before user code and should not be modified. We detect them by
        // checking for the __amd_rocclr_ prefix in the .note metadata.
        // A simpler heuristic: skip patching entirely and only do instruction
        // analysis + dispatch interception with kernarg repacking.
        //
        // For now, we do NOT patch code objects. The kernarg repacking in
        // doPackets relies on knowing the ArgDescriptor for each kernel.
        // Instead, we extract kernel metadata (kernarg sizes, hidden arg
        // layout) from the code object without modifying it, and compute
        // the ArgDescriptor as if we had patched it.
        //
        // The probe buffer pointer goes at the end of the existing kernarg
        // buffer — we over-allocate the kernarg and place it there. The
        // kernel won't read it (no assembly injection yet), but the
        // plumbing is verified.

        // Extract kernel metadata without patching
        // Write code object to temp file for Python analysis
        char tmpco[] = "/tmp/proboscis_analyze_XXXXXX";
        int fd = mkstemp(tmpco);
        if (fd >= 0) {
            write(fd, code_object, size);
            close(fd);

            // Call patcher in --plan-only mode to get arg descriptors
            char tmpplan[] = "/tmp/proboscis_plan_XXXXXX";
            int fd2 = mkstemp(tmpplan);
            if (fd2 >= 0) {
                close(fd2);
                std::ostringstream cmd;
                cmd << "python3 -m proboscis.patcher --plan-only "
                    << tmpco << " " << tmpplan;
                if (!self->config_.target_kernel.empty()) {
                    cmd << " --target " << self->config_.target_kernel;
                }

                int ret = system(cmd.str().c_str());
                if (ret == 0) {
                    // Read plan JSON
                    std::ifstream planf(tmpplan);
                    if (planf.good()) {
                        std::ostringstream ss;
                        ss << planf.rdbuf();
                        std::string plan_json = ss.str();
                        planf.close();

                        // Parse plans
                        size_t pos = 0;
                        while ((pos = plan_json.find("\"kernel\"", pos)) != std::string::npos) {
                            auto extract_str = [&](const std::string& key) -> std::string {
                                auto p = plan_json.find("\"" + key + "\"", pos);
                                if (p == std::string::npos) return "";
                                p = plan_json.find("\"", p + key.size() + 2);
                                if (p == std::string::npos) return "";
                                auto e = plan_json.find("\"", p + 1);
                                if (e == std::string::npos) return "";
                                return plan_json.substr(p + 1, e - p - 1);
                            };
                            auto extract_int = [&](const std::string& key) -> uint32_t {
                                auto p = plan_json.find("\"" + key + "\"", pos);
                                if (p == std::string::npos) return 0;
                                p = plan_json.find(":", p);
                                if (p == std::string::npos) return 0;
                                p++;
                                while (p < plan_json.size() && plan_json[p] == ' ') p++;
                                return std::atoi(plan_json.c_str() + p);
                            };

                            std::string kernel_name = extract_str("kernel");
                            if (!kernel_name.empty()) {
                                ArgDescriptor desc{};
                                desc.explicit_args_length = extract_int("explicit_len");
                                desc.kernarg_length = extract_int("orig_size");
                                desc.instrumented_kernarg_length = extract_int("new_size");
                                desc.probe_ctx_offset = extract_int("probe_ctx_offset");
                                desc.hidden_args_length = desc.kernarg_length - desc.explicit_args_length;

                                std::lock_guard<std::mutex> lock(self->mutex_);
                                self->kernel_arg_descs_[kernel_name] = desc;

                                if (self->config_.log_level >= 1) {
                                    std::cerr << "[proboscis] Kernel: " << kernel_name
                                              << " kernarg=" << desc.kernarg_length
                                              << " probe_ctx_offset=" << desc.probe_ctx_offset
                                              << " new_size=" << desc.instrumented_kernarg_length
                                              << std::endl;
                                }
                            }
                            pos++;
                        }
                    }
                }
                unlink(tmpplan);
            }
            unlink(tmpco);
        }
    }

    // Fall through: use original code object
    return orig_code_obj_reader_create_from_memory(code_object, size, code_object_reader);
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
    orig_code_obj_reader_create_from_memory =
        reinterpret_cast<code_obj_reader_create_from_memory_t>(
            *core_field(table->core_, FIELD_CODE_OBJ_READER_CREATE_FROM_MEMORY));

    // Install hooks
    table->core_->hsa_queue_create_fn = ProboscisInterceptor::hsa_queue_create;
    table->core_->hsa_executable_symbol_get_info_fn =
        ProboscisInterceptor::hsa_executable_symbol_get_info;
    *core_field(table->core_, FIELD_GET_SYMBOL_BY_NAME) =
        reinterpret_cast<void*>(ProboscisInterceptor::hsa_executable_get_symbol_by_name);
    *core_field(table->core_, FIELD_CODE_OBJ_READER_CREATE_FROM_MEMORY) =
        reinterpret_cast<void*>(ProboscisInterceptor::hsa_code_object_reader_create_from_memory);

    ProboscisInterceptor::getInstance();
    std::cerr << "[proboscis] Tool loaded" << std::endl;
    return true;
}

PUBLIC_API void OnUnload() {
    ProboscisInterceptor::cleanup();
    std::cerr << "[proboscis] Tool unloaded" << std::endl;
}

} // extern "C"
