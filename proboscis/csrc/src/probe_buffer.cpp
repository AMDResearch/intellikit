/******************************************************************************
 * MIT License
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Probe buffer allocation and management.
 *
 * The probe buffer is a GPU-visible memory region passed to instrumented
 * kernels via the hidden_proboscis_ctx pointer. Layout:
 *
 *   [0:8]   write_offset  — atomic counter, incremented per record
 *   [8:16]  max_records   — capacity
 *   [16:24] record_size   — bytes per record
 *   [24:..] records[]     — array of probe-type-specific records
 ******************************************************************************/

#include "proboscis/proboscis.h"

#include <cstring>
#include <iostream>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

/// Initialize a probe buffer header with the given capacity and record size.
void initProbeBuffer(void* buffer, uint64_t max_records, uint64_t record_size) {
    auto* header = static_cast<ProbeBufferHeader*>(buffer);
    header->write_offset = 0;
    header->max_records = max_records;
    header->record_size = record_size;
}

/// Compute the total byte size needed for a probe buffer.
size_t computeProbeBufferSize(uint64_t max_records, uint64_t record_size) {
    return sizeof(ProbeBufferHeader) + max_records * record_size;
}
