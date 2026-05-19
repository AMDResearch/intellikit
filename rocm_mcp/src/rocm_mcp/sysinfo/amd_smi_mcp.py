# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

import argparse
from typing import Annotated

from fastmcp import Context, FastMCP
from fastmcp.utilities.logging import get_logger
from pydantic import Field

from rocm_mcp.sysinfo import AmdSmi

# initialize server
mcp = FastMCP(
    name="amd-smi",
    instructions=("MCP server for querying AMD GPU system management information."),
)
logger = get_logger(mcp.name)
amd_smi = AmdSmi(logger=logger)


@mcp.tool()
async def get_driver_information(ctx: Annotated[Context, Field(description="MCP context.")]) -> str:
    """Get the AMD GPU driver version via the amdsmi Python API.

    Returns:
        str: The driver version, name, and date.
    """
    try:
        result = amd_smi.get_driver_information()
    except Exception as e:
        msg = f"Failed to get driver version: {e!s}"
        await ctx.error(msg)
        return msg
    else:
        return (
            f"Driver Name: {result.name}\n"
            f"Driver Version: {result.version}\n"
            f"Driver Date: {result.date}"
        )


@mcp.tool()
async def list_gpus(ctx: Annotated[Context, Field(description="MCP context.")]) -> str:
    """List all AMD GPUs detected in the system.

    Returns:
        str: GPU index, BDF address, and UUID for each GPU.
    """
    try:
        gpus = amd_smi.list_gpus()
    except Exception as e:
        msg = f"Failed to list GPUs: {e!s}"
        await ctx.error(msg)
        return msg
    else:
        lines = [f"GPU {gpu.gpu_index}:\n  BDF: {gpu.bdf}\n  UUID: {gpu.uuid}" for gpu in gpus]
        return "\n".join(lines)


@mcp.tool()
async def get_gpu_static_info(
    ctx: Annotated[Context, Field(description="MCP context.")],
    gpu_index: Annotated[
        int | None,
        Field(description="Zero-based GPU index. If None, returns info for all GPUs."),
    ] = None,
) -> str:
    """Get static hardware properties for AMD GPUs.

    Returns:
        str: Market name, VRAM, compute units, power caps, and other hardware details.
    """
    try:
        infos = amd_smi.get_gpu_static_info(gpu_index)
    except IndexError as e:
        return f"Error: {e!s}"
    except Exception as e:
        msg = f"Failed to get GPU static info: {e!s}"
        await ctx.error(msg)
        return msg
    else:
        lines = [
            f"GPU {info.gpu_index} ({info.bdf}):\n"
            f"  Market Name: {info.market_name}\n"
            f"  Vendor/Device/Rev ID: {info.vendor_id}/{info.device_id}/{info.rev_id}\n"
            f"  ASIC Serial: {info.asic_serial}\n"
            f"  Compute Units: {info.num_compute_units}\n"
            f"  VRAM: {info.vram_size_mb} MB {info.vram_type} ({info.vram_vendor})\n"
            f"  Model: {info.model_name}\n"
            f"  Power Cap: {info.power_cap_w}W "
            f"(default {info.default_power_cap_w}W, "
            f"range {info.min_power_cap_w}-{info.max_power_cap_w}W)"
            for info in infos
        ]
        return "\n".join(lines)


@mcp.tool()
async def get_gpu_metrics(
    ctx: Annotated[Context, Field(description="MCP context.")],
    gpu_index: Annotated[
        int | None,
        Field(description="Zero-based GPU index. If None, returns metrics for all GPUs."),
    ] = None,
) -> str:
    """Get real-time performance metrics for AMD GPUs.

    Returns:
        str: Activity, temperature, power, clock, and VRAM usage for each GPU.
    """
    try:
        metrics_list = amd_smi.get_gpu_metrics(gpu_index)
    except IndexError as e:
        return f"Error: {e!s}"
    except Exception as e:
        msg = f"Failed to get GPU metrics: {e!s}"
        await ctx.error(msg)
        return msg
    else:
        lines = [
            f"GPU {m.gpu_index} ({m.bdf}):\n"
            f"  Activity: GFX {m.gfx_activity_pct}%, UMC {m.umc_activity_pct}%, "
            f"MM {m.mm_activity_pct}%\n"
            f"  Temperature: Edge {m.temp_edge_c}C, Hotspot {m.temp_hotspot_c}C, "
            f"VRAM {m.temp_vram_c}C\n"
            f"  Power: {m.current_power_w}W\n"
            f"  Voltage: GFX {m.gfx_voltage_mv}mV, SoC {m.soc_voltage_mv}mV, "
            f"Mem {m.mem_voltage_mv}mV\n"
            f"  Clocks: GFX {m.gfx_clock_mhz}/{m.gfx_max_clock_mhz} MHz, "
            f"MEM {m.mem_clock_mhz}/{m.mem_max_clock_mhz} MHz\n"
            f"  VRAM: {m.vram_used_mb}/{m.vram_total_mb} MB"
            for m in metrics_list
        ]
        return "\n".join(lines)


@mcp.tool()
async def get_gpu_firmware_info(
    ctx: Annotated[Context, Field(description="MCP context.")],
    gpu_index: Annotated[
        int | None,
        Field(description="Zero-based GPU index. If None, returns info for all GPUs."),
    ] = None,
) -> str:
    """Get firmware version information for AMD GPUs.

    Returns:
        str: VBIOS info and firmware block versions for each GPU.
    """
    try:
        fw_list = amd_smi.get_gpu_firmware_info(gpu_index)
    except IndexError as e:
        return f"Error: {e!s}"
    except Exception as e:
        msg = f"Failed to get GPU firmware info: {e!s}"
        await ctx.error(msg)
        return msg
    else:
        lines = []
        for fw in fw_list:
            fw_lines = [f"    {entry.name}: {entry.version}" for entry in fw.firmware_list]
            fw_block = "\n".join(fw_lines) if fw_lines else "    (none)"
            lines.append(
                f"GPU {fw.gpu_index} ({fw.bdf}):\n"
                f"  VBIOS: {fw.vbios_name} v{fw.vbios_version} "
                f"({fw.vbios_build_date}, {fw.vbios_part_number})\n"
                f"  Firmware:\n{fw_block}"
            )
        return "\n".join(lines)


@mcp.tool()
async def get_gpu_process_info(
    ctx: Annotated[Context, Field(description="MCP context.")],
    gpu_index: Annotated[
        int | None,
        Field(description="Zero-based GPU index. If None, returns info for all GPUs."),
    ] = None,
) -> str:
    """Get information about processes using AMD GPUs.

    Returns:
        str: PID, name, and VRAM usage for processes on each GPU.
    """
    try:
        proc_list = amd_smi.get_gpu_process_info(gpu_index)
    except IndexError as e:
        return f"Error: {e!s}"
    except Exception as e:
        msg = f"Failed to get GPU process info: {e!s}"
        await ctx.error(msg)
        return msg
    else:
        lines = []
        for gpu in proc_list:
            if gpu.processes:
                proc_lines = [
                    f"    PID {p.pid} ({p.name}): {p.vram_usage_mb} MB VRAM" for p in gpu.processes
                ]
                proc_block = "\n".join(proc_lines)
            else:
                proc_block = "    (no processes)"
            lines.append(f"GPU {gpu.gpu_index} ({gpu.bdf}):\n{proc_block}")
        return "\n".join(lines)


@mcp.tool()
async def get_gpu_bad_pages(
    ctx: Annotated[Context, Field(description="MCP context.")],
    gpu_index: Annotated[
        int | None,
        Field(description="Zero-based GPU index. If None, returns info for all GPUs."),
    ] = None,
) -> str:
    """Get bad page and ECC error information for AMD GPUs.

    Returns:
        str: ECC error counts and retired page count for each GPU.
    """
    try:
        bp_list = amd_smi.get_gpu_bad_page_info(gpu_index)
    except IndexError as e:
        return f"Error: {e!s}"
    except Exception as e:
        msg = f"Failed to get GPU bad page info: {e!s}"
        await ctx.error(msg)
        return msg
    else:
        lines = [
            f"GPU {bp.gpu_index} ({bp.bdf}):\n"
            f"  ECC Errors: {bp.correctable_ecc_count} correctable, "
            f"{bp.uncorrectable_ecc_count} uncorrectable, "
            f"{bp.deferred_ecc_count} deferred\n"
            f"  Bad Pages: {bp.bad_page_count}"
            for bp in bp_list
        ]
        return "\n".join(lines)


def main() -> None:
    """Main function to run the amd-smi MCP server."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport to use",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the HTTP server to (only used if transport is http)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind the HTTP server to (only used if transport is http)",
    )
    parser.add_argument(
        "--path",
        default="/rocm_mcp/amd_smi",
        help="Path to serve the HTTP server on (only used if transport is http)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port, path=args.path)


if __name__ == "__main__":
    main()
