/**
 * kernarg_metadata.cpp — AMDGPU kernarg metadata parser implementation
 */
#include "kernarg_metadata.hpp"

#include "kerncap_log.hpp"

#include <amd_comgr/amd_comgr.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <cstring>
#include <unordered_map>

namespace kerncap {

const char* to_string(ValueKind k) {
    switch (k) {
        case ValueKind::Unknown:                return "unknown";
        case ValueKind::GlobalBuffer:           return "global_buffer";
        case ValueKind::DynamicSharedPointer:   return "dynamic_shared_pointer";
        case ValueKind::ByValue:                return "by_value";
        case ValueKind::Image:                  return "image";
        case ValueKind::Sampler:                return "sampler";
        case ValueKind::Pipe:                   return "pipe";
        case ValueKind::Queue:                  return "queue";
        case ValueKind::HiddenGlobalOffsetX:    return "hidden_global_offset_x";
        case ValueKind::HiddenGlobalOffsetY:    return "hidden_global_offset_y";
        case ValueKind::HiddenGlobalOffsetZ:    return "hidden_global_offset_z";
        case ValueKind::HiddenNone:             return "hidden_none";
        case ValueKind::HiddenPrintfBuffer:     return "hidden_printf_buffer";
        case ValueKind::HiddenHostcallBuffer:   return "hidden_hostcall_buffer";
        case ValueKind::HiddenDefaultQueue:     return "hidden_default_queue";
        case ValueKind::HiddenCompletionAction: return "hidden_completion_action";
        case ValueKind::HiddenMultigridSyncArg: return "hidden_multigrid_sync_arg";
        case ValueKind::HiddenBlockCountX:      return "hidden_block_count_x";
        case ValueKind::HiddenBlockCountY:      return "hidden_block_count_y";
        case ValueKind::HiddenBlockCountZ:      return "hidden_block_count_z";
        case ValueKind::HiddenGroupSizeX:       return "hidden_group_size_x";
        case ValueKind::HiddenGroupSizeY:       return "hidden_group_size_y";
        case ValueKind::HiddenGroupSizeZ:       return "hidden_group_size_z";
        case ValueKind::HiddenRemainderX:       return "hidden_remainder_x";
        case ValueKind::HiddenRemainderY:       return "hidden_remainder_y";
        case ValueKind::HiddenRemainderZ:       return "hidden_remainder_z";
        case ValueKind::HiddenGridDims:         return "hidden_grid_dims";
        case ValueKind::HiddenHeapV1:           return "hidden_heap_v1";
        case ValueKind::HiddenDynamicLdsSize:   return "hidden_dynamic_lds_size";
        case ValueKind::HiddenPrivateBase:      return "hidden_private_base";
        case ValueKind::HiddenSharedBase:       return "hidden_shared_base";
        case ValueKind::HiddenQueuePtr:         return "hidden_queue_ptr";
        case ValueKind::HiddenOther:            return "hidden_other";
    }
    return "unknown";
}

ValueKind classify_value_kind(const std::string& s) {
    static const std::unordered_map<std::string, ValueKind> kTable = {
        {"global_buffer",            ValueKind::GlobalBuffer},
        {"dynamic_shared_pointer",   ValueKind::DynamicSharedPointer},
        {"by_value",                 ValueKind::ByValue},
        {"image",                    ValueKind::Image},
        {"sampler",                  ValueKind::Sampler},
        {"pipe",                     ValueKind::Pipe},
        {"queue",                    ValueKind::Queue},
        {"hidden_global_offset_x",   ValueKind::HiddenGlobalOffsetX},
        {"hidden_global_offset_y",   ValueKind::HiddenGlobalOffsetY},
        {"hidden_global_offset_z",   ValueKind::HiddenGlobalOffsetZ},
        {"hidden_none",              ValueKind::HiddenNone},
        {"hidden_printf_buffer",     ValueKind::HiddenPrintfBuffer},
        {"hidden_hostcall_buffer",   ValueKind::HiddenHostcallBuffer},
        {"hidden_default_queue",     ValueKind::HiddenDefaultQueue},
        {"hidden_completion_action", ValueKind::HiddenCompletionAction},
        {"hidden_multigrid_sync_arg",ValueKind::HiddenMultigridSyncArg},
        {"hidden_block_count_x",     ValueKind::HiddenBlockCountX},
        {"hidden_block_count_y",     ValueKind::HiddenBlockCountY},
        {"hidden_block_count_z",     ValueKind::HiddenBlockCountZ},
        {"hidden_group_size_x",      ValueKind::HiddenGroupSizeX},
        {"hidden_group_size_y",      ValueKind::HiddenGroupSizeY},
        {"hidden_group_size_z",      ValueKind::HiddenGroupSizeZ},
        {"hidden_remainder_x",       ValueKind::HiddenRemainderX},
        {"hidden_remainder_y",       ValueKind::HiddenRemainderY},
        {"hidden_remainder_z",       ValueKind::HiddenRemainderZ},
        {"hidden_grid_dims",         ValueKind::HiddenGridDims},
        {"hidden_heap_v1",           ValueKind::HiddenHeapV1},
        {"hidden_dynamic_lds_size",  ValueKind::HiddenDynamicLdsSize},
        {"hidden_private_base",      ValueKind::HiddenPrivateBase},
        {"hidden_shared_base",       ValueKind::HiddenSharedBase},
        {"hidden_queue_ptr",         ValueKind::HiddenQueuePtr},
    };
    auto it = kTable.find(s);
    if (it != kTable.end()) return it->second;
    if (s.rfind("hidden_", 0) == 0) return ValueKind::HiddenOther;
    return ValueKind::Unknown;
}

namespace {

// RAII wrapper for amd_comgr_metadata_node_t.
struct ComgrNode {
    amd_comgr_metadata_node_t node{};
    bool valid = false;
    ComgrNode() = default;
    ComgrNode(const ComgrNode&) = delete;
    ComgrNode& operator=(const ComgrNode&) = delete;
    ComgrNode(ComgrNode&& o) noexcept : node(o.node), valid(o.valid) {
        o.valid = false;
    }
    ~ComgrNode() {
        if (valid) amd_comgr_destroy_metadata(node);
    }
};

struct ComgrData {
    amd_comgr_data_t data{};
    bool valid = false;
    ComgrData() = default;
    ComgrData(const ComgrData&) = delete;
    ComgrData& operator=(const ComgrData&) = delete;
    ~ComgrData() {
        if (valid) amd_comgr_release_data(data);
    }
};

bool get_string(amd_comgr_metadata_node_t node, std::string& out) {
    size_t sz = 0;
    if (amd_comgr_get_metadata_string(node, &sz, nullptr) != AMD_COMGR_STATUS_SUCCESS) {
        return false;
    }
    if (sz == 0) {
        out.clear();
        return true;
    }
    out.resize(sz);
    if (amd_comgr_get_metadata_string(node, &sz, out.data()) != AMD_COMGR_STATUS_SUCCESS) {
        return false;
    }
    // The returned size includes the trailing NUL.
    if (!out.empty() && out.back() == '\0') out.pop_back();
    return true;
}

bool lookup_string(amd_comgr_metadata_node_t map, const char* key, std::string& out) {
    amd_comgr_metadata_node_t v{};
    if (amd_comgr_metadata_lookup(map, key, &v) != AMD_COMGR_STATUS_SUCCESS) {
        return false;
    }
    ComgrNode owner;
    owner.node = v;
    owner.valid = true;
    return get_string(owner.node, out);
}

bool lookup_uint(amd_comgr_metadata_node_t map, const char* key, uint64_t& out) {
    std::string s;
    if (!lookup_string(map, key, s)) return false;
    try {
        out = std::stoull(s);
        return true;
    } catch (...) {
        return false;
    }
}

amd_comgr_status_t parse_arg(amd_comgr_metadata_node_t arg_node,
                             KernargSlot& slot) {
    uint64_t u = 0;
    if (lookup_uint(arg_node, ".offset", u)) slot.offset = static_cast<uint32_t>(u);
    if (lookup_uint(arg_node, ".size", u)) slot.size = static_cast<uint32_t>(u);
    lookup_string(arg_node, ".value_kind", slot.kind_str);
    lookup_string(arg_node, ".value_type", slot.value_type);
    lookup_string(arg_node, ".address_space", slot.address_space);
    lookup_string(arg_node, ".name", slot.name);
    slot.kind = classify_value_kind(slot.kind_str);
    return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t parse_kernel(amd_comgr_metadata_node_t kernel_node,
                                KernelKernargInfo& info) {
    lookup_string(kernel_node, ".symbol", info.symbol);
    lookup_string(kernel_node, ".name", info.name);
    uint64_t u = 0;
    if (lookup_uint(kernel_node, ".kernarg_segment_size", u))
        info.kernarg_segment_size = static_cast<uint32_t>(u);
    if (lookup_uint(kernel_node, ".group_segment_fixed_size", u))
        info.group_segment_fixed_size = static_cast<uint32_t>(u);
    if (lookup_uint(kernel_node, ".private_segment_fixed_size", u))
        info.private_segment_fixed_size = static_cast<uint32_t>(u);
    if (lookup_uint(kernel_node, ".sgpr_count", u))
        info.sgpr_count = static_cast<uint32_t>(u);
    if (lookup_uint(kernel_node, ".vgpr_count", u))
        info.vgpr_count = static_cast<uint32_t>(u);
    if (lookup_uint(kernel_node, ".kernarg_segment_align", u))
        info.kernarg_segment_align = static_cast<uint32_t>(u);

    amd_comgr_metadata_node_t args_node{};
    if (amd_comgr_metadata_lookup(kernel_node, ".args", &args_node)
            != AMD_COMGR_STATUS_SUCCESS) {
        return AMD_COMGR_STATUS_SUCCESS;  // no .args is legal (e.g. pure scalar kernel)
    }
    ComgrNode args_owner;
    args_owner.node = args_node;
    args_owner.valid = true;

    size_t n = 0;
    if (amd_comgr_get_metadata_list_size(args_node, &n) != AMD_COMGR_STATUS_SUCCESS)
        return AMD_COMGR_STATUS_SUCCESS;

    info.args.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        amd_comgr_metadata_node_t arg{};
        if (amd_comgr_index_list_metadata(args_node, i, &arg)
                != AMD_COMGR_STATUS_SUCCESS) continue;
        ComgrNode arg_owner;
        arg_owner.node = arg;
        arg_owner.valid = true;
        KernargSlot slot;
        parse_arg(arg, slot);
        info.args.push_back(std::move(slot));
    }
    return AMD_COMGR_STATUS_SUCCESS;
}

}  // namespace

std::vector<KernelKernargInfo>
parse_kernarg_metadata(const void* hsaco, size_t bytes) {
    std::vector<KernelKernargInfo> out;
    if (!hsaco || bytes == 0) return out;

    ComgrData data;
    if (amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &data.data)
            != AMD_COMGR_STATUS_SUCCESS) {
        KERNCAP_WARN("amd_comgr_create_data failed");
        return out;
    }
    data.valid = true;

    if (amd_comgr_set_data(data.data, bytes,
                           static_cast<const char*>(hsaco))
            != AMD_COMGR_STATUS_SUCCESS) {
        KERNCAP_WARN("amd_comgr_set_data failed");
        return out;
    }

    amd_comgr_metadata_node_t root{};
    if (amd_comgr_get_data_metadata(data.data, &root)
            != AMD_COMGR_STATUS_SUCCESS) {
        KERNCAP_WARN("amd_comgr_get_data_metadata failed");
        return out;
    }
    ComgrNode root_owner;
    root_owner.node = root;
    root_owner.valid = true;

    amd_comgr_metadata_node_t kernels{};
    if (amd_comgr_metadata_lookup(root, "amdhsa.kernels", &kernels)
            != AMD_COMGR_STATUS_SUCCESS) {
        KERNCAP_WARN("amdhsa.kernels not found in HSACO metadata");
        return out;
    }
    ComgrNode kernels_owner;
    kernels_owner.node = kernels;
    kernels_owner.valid = true;

    size_t n = 0;
    if (amd_comgr_get_metadata_list_size(kernels, &n)
            != AMD_COMGR_STATUS_SUCCESS) {
        return out;
    }

    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        amd_comgr_metadata_node_t kernel{};
        if (amd_comgr_index_list_metadata(kernels, i, &kernel)
                != AMD_COMGR_STATUS_SUCCESS) continue;
        ComgrNode kernel_owner;
        kernel_owner.node = kernel;
        kernel_owner.valid = true;
        KernelKernargInfo info;
        parse_kernel(kernel, info);
        out.push_back(std::move(info));
    }
    return out;
}

// Format the parsed table as a JSON string (used by the Python ctypes
// shim for unit tests).
static std::string to_json(const std::vector<KernelKernargInfo>& all) {
    nlohmann::json kernels = nlohmann::json::array();
    for (const auto& k : all) {
        nlohmann::json args = nlohmann::json::array();
        for (const auto& a : k.args) {
            args.push_back({
                {"offset", a.offset},
                {"size", a.size},
                {"value_kind", a.kind_str},
                {"value_type", a.value_type},
                {"address_space", a.address_space},
                {"name", a.name},
            });
        }
        kernels.push_back({
            {"symbol", k.symbol},
            {"name", k.name},
            {"kernarg_segment_size", k.kernarg_segment_size},
            {"group_segment_fixed_size", k.group_segment_fixed_size},
            {"private_segment_fixed_size", k.private_segment_fixed_size},
            {"sgpr_count", k.sgpr_count},
            {"vgpr_count", k.vgpr_count},
            {"args", std::move(args)},
        });
    }
    return kernels.dump();
}

extern "C" {

// C entry point for the Python ctypes shim used by unit tests.  Caller
// must free the returned buffer with ``free`` (i.e. ``libc.free``).  On
// failure returns NULL.
__attribute__((visibility("default")))
char* kerncap_parse_kernarg_metadata_json(const void* hsaco, size_t bytes) {
    auto kernels = parse_kernarg_metadata(hsaco, bytes);
    auto s = to_json(kernels);
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (!out) return nullptr;
    std::memcpy(out, s.data(), s.size());
    out[s.size()] = '\0';
    return out;
}

}  // extern "C"

std::optional<KernelKernargInfo>
find_kernarg_metadata(const std::vector<KernelKernargInfo>& all,
                      const std::string& symbol) {
    auto bare = symbol;
    if (bare.size() > 3 && bare.compare(bare.size() - 3, 3, ".kd") == 0)
        bare.resize(bare.size() - 3);
    auto kd = bare + ".kd";
    for (const auto& k : all) {
        if (k.symbol == symbol || k.symbol == bare || k.symbol == kd ||
            k.name == bare) {
            return k;
        }
    }
    return std::nullopt;
}

}  // namespace kerncap
