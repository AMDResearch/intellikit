/**
 * kernarg_metadata.hpp — AMDGPU kernarg metadata parser
 *
 * Extracts per-kernel kernarg slot tables (offset, size, value_kind,
 * value_type, name, address_space) from the ``NT_AMDGPU_METADATA`` note
 * in an AMDGPU code object (HSACO).
 *
 * Implementation uses ``amd_comgr`` (the official ROCm comgr API) to
 * walk the msgpack-encoded metadata.  No HSA runtime is required, which
 * makes the parser unit-testable from a static HSACO fixture.
 *
 * See Phase 1 spike under ``kerncap_paper/spikes/triton_hsa_capture/``
 * for the empirical validation of the layout this parser recovers.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace kerncap {

enum class ValueKind {
    Unknown,
    GlobalBuffer,
    DynamicSharedPointer,
    ByValue,
    Image,
    Sampler,
    Pipe,
    Queue,
    HiddenGlobalOffsetX,
    HiddenGlobalOffsetY,
    HiddenGlobalOffsetZ,
    HiddenNone,
    HiddenPrintfBuffer,
    HiddenHostcallBuffer,
    HiddenDefaultQueue,
    HiddenCompletionAction,
    HiddenMultigridSyncArg,
    HiddenBlockCountX,
    HiddenBlockCountY,
    HiddenBlockCountZ,
    HiddenGroupSizeX,
    HiddenGroupSizeY,
    HiddenGroupSizeZ,
    HiddenRemainderX,
    HiddenRemainderY,
    HiddenRemainderZ,
    HiddenGridDims,
    HiddenHeapV1,
    HiddenDynamicLdsSize,
    HiddenPrivateBase,
    HiddenSharedBase,
    HiddenQueuePtr,
    HiddenOther,
};

const char* to_string(ValueKind k);

// Map the raw ``.value_kind`` string from AMDGPU msgpack metadata onto
// the enum.  Returns ``HiddenOther`` for any ``hidden_*`` we don't
// recognise (so they're still tagged as zero-fill candidates) and
// ``Unknown`` for anything else.
ValueKind classify_value_kind(const std::string& s);

struct KernargSlot {
    uint32_t offset = 0;
    uint32_t size = 0;
    ValueKind kind = ValueKind::Unknown;
    std::string kind_str;        // raw .value_kind string (debug aid)
    std::string value_type;      // optional .value_type
    std::string address_space;   // optional .address_space
    std::string name;            // optional .name
};

struct KernelKernargInfo {
    std::string symbol;          // e.g. "triton_poi_fused_relu_0.kd"
    std::string name;            // e.g. "triton_poi_fused_relu_0"
    uint32_t kernarg_segment_size = 0;
    uint32_t group_segment_fixed_size = 0;
    uint32_t private_segment_fixed_size = 0;
    uint32_t sgpr_count = 0;
    uint32_t vgpr_count = 0;
    uint32_t kernarg_segment_align = 0;
    std::vector<KernargSlot> args;
};

// Parse every ``amdhsa.kernels`` entry in the HSACO. Returns an empty
// vector on failure (no exceptions, so the call site can decide what
// to do — typically log a warning and fall back to dumping the raw
// kernarg blob).
std::vector<KernelKernargInfo>
parse_kernarg_metadata(const void* hsaco, size_t bytes);

// Convenience helper: find one kernel by its HSA symbol name. The
// symbol may include the trailing ``.kd`` (kernel descriptor) suffix
// or not — both forms are tried.
std::optional<KernelKernargInfo>
find_kernarg_metadata(const std::vector<KernelKernargInfo>& all,
                      const std::string& symbol);

}  // namespace kerncap
