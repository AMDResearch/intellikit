/******************************************************************************
 * MIT License
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Kernarg buffer repacking for instrumented dispatches.
 *
 * When dispatching an instrumented kernel, the original kernarg buffer must
 * be repacked into a larger buffer that includes the hidden_proboscis_ctx
 * pointer at the offset specified by the metadata.
 *
 * Ported from omniprobe src/utils.cc: repackInstrumentedKernArgs()
 ******************************************************************************/

#include "proboscis/proboscis.h"

#include <cstring>
#include <iostream>

/// Repack kernarg buffer to include the probe context pointer.
///
/// @param dst         Destination buffer (instrumented_kernarg_length bytes)
/// @param src         Source kernarg buffer from the original dispatch
/// @param probe_ctx   Pointer to the probe buffer (will be written at probe_ctx_offset)
/// @param desc        Argument descriptor with layout information
void repackInstrumentedKernArgs(
    void* dst,
    const void* src,
    void* probe_ctx,
    const ArgDescriptor& desc)
{
    // Zero the destination buffer
    std::memset(dst, 0, desc.instrumented_kernarg_length);

    // Copy the entire original kernarg buffer (explicit + hidden args)
    std::memcpy(dst, src, desc.kernarg_length);

    // Write the probe context pointer at the computed offset
    // This is the hidden_proboscis_ctx argument that the metadata declares
    auto* dst_bytes = static_cast<uint8_t*>(dst);
    std::memcpy(dst_bytes + desc.probe_ctx_offset, &probe_ctx, sizeof(void*));
}
