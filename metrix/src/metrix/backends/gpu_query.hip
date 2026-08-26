/*
 * gpu_query — Portable HIP device property query tool.
 *
 * Queries hipDeviceProp_t for all visible GPUs and emits a JSON array
 * to stdout.
 *
 * Usage:  ./gpu_query              -> JSON for all GPUs
 *         ./gpu_query 0            -> JSON for device 0 only
 *         ./gpu_query --device 2   -> JSON for device 2 only
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(call)                                                    \
    do {                                                                   \
        hipError_t err = (call);                                           \
        if (err != hipSuccess) {                                           \
            fprintf(stderr, "HIP error: %s (%d) at %s:%d\n",              \
                    hipGetErrorString(err), err, __FILE__, __LINE__);      \
            return 1;                                                      \
        }                                                                  \
    } while (0)

static void escape_json_string(const char* s, FILE* out) {
    fputc('"', out);
    for (; *s; ++s) {
        switch (*s) {
            case '"':  fputs("\\\"", out); break;
            case '\\': fputs("\\\\", out); break;
            case '\n': fputs("\\n", out);  break;
            case '\r': fputs("\\r", out);  break;
            case '\t': fputs("\\t", out);  break;
            default:   fputc(*s, out);     break;
        }
    }
    fputc('"', out);
}

static void print_device_json(int device_id, FILE* out) {
    hipDeviceProp_t prop;
    (void)hipGetDeviceProperties(&prop, device_id);

    fprintf(out, "  {\n");
    fprintf(out, "    \"device_id\": %d,\n", device_id);
    fprintf(out, "    \"name\": ");       escape_json_string(prop.name, out);       fprintf(out, ",\n");
    fprintf(out, "    \"gcn_arch_name\": "); escape_json_string(prop.gcnArchName, out); fprintf(out, ",\n");
    fprintf(out, "    \"num_cu\": %d,\n", prop.multiProcessorCount);
    fprintf(out, "    \"wavefront_size\": %d,\n", prop.warpSize);
    fprintf(out, "    \"max_threads_per_multiprocessor\": %d,\n", prop.maxThreadsPerMultiProcessor);
    fprintf(out, "    \"clock_rate_khz\": %d,\n", prop.clockRate);
    fprintf(out, "    \"memory_clock_rate_khz\": %d,\n", prop.memoryClockRate);
    fprintf(out, "    \"memory_bus_width_bits\": %d,\n", prop.memoryBusWidth);
    fprintf(out, "    \"l2_cache_size_bytes\": %d,\n", prop.l2CacheSize);
    fprintf(out, "    \"max_shared_memory_per_multiprocessor\": %zu,\n", prop.maxSharedMemoryPerMultiProcessor);
    fprintf(out, "    \"total_global_mem_bytes\": %zu,\n", prop.totalGlobalMem);
    fprintf(out, "    \"max_threads_per_block\": %d,\n", prop.maxThreadsPerBlock);
    fprintf(out, "    \"shared_mem_per_block_bytes\": %zu,\n", prop.sharedMemPerBlock);
    fprintf(out, "    \"regs_per_block\": %d,\n", prop.regsPerBlock);
    fprintf(out, "    \"compute_major\": %d,\n", prop.major);
    fprintf(out, "    \"compute_minor\": %d,\n", prop.minor);
    fprintf(out, "    \"ecc_enabled\": %d,\n", prop.ECCEnabled);
    fprintf(out, "    \"pci_bus_id\": %d,\n", prop.pciBusID);
    fprintf(out, "    \"pci_device_id\": %d,\n", prop.pciDeviceID);
    fprintf(out, "    \"pci_domain_id\": %d\n", prop.pciDomainID);
    fprintf(out, "  }");
}

int main(int argc, char** argv) {
    int target_device = -1;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            target_device = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--device N | N]\n", argv[0]);
            return 0;
        } else {
            char* end;
            long v = strtol(argv[i], &end, 10);
            if (*end == '\0')
                target_device = static_cast<int>(v);
        }
    }

    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));

    if (device_count == 0) {
        fprintf(stderr, "No HIP devices found\n");
        return 1;
    }

    if (target_device >= device_count) {
        fprintf(stderr, "Device %d out of range (0..%d)\n", target_device, device_count - 1);
        return 1;
    }

    printf("[\n");
    if (target_device >= 0) {
        print_device_json(target_device, stdout);
        printf("\n");
    } else {
        for (int i = 0; i < device_count; ++i) {
            print_device_json(i, stdout);
            printf(i + 1 < device_count ? ",\n" : "\n");
        }
    }
    printf("]\n");
    return 0;
}
