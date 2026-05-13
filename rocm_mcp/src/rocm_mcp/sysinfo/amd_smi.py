# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import amdsmi


@dataclass(frozen=True)
class DriverInformationResult:
    """Result of driver version query.

    Attributes:
        version (str): The driver version string.
        name (str): The driver name (e.g., "amdgpu").
        date (str): The driver date string.
    """

    version: str
    name: str
    date: str


@dataclass(frozen=True)
class GpuInfo:
    """Basic GPU identification info.

    Attributes:
        gpu_index (int): Zero-based GPU index.
        bdf (str): PCIe Bus/Device/Function address.
        uuid (str): GPU UUID.
    """

    gpu_index: int
    bdf: str
    uuid: str


@dataclass(frozen=True)
class GpuStaticInfo:
    """Static hardware properties of a GPU.

    Attributes:
        gpu_index (int): Zero-based GPU index.
        bdf (str): PCIe Bus/Device/Function address.
        market_name (str): Marketing name of the GPU.
        vendor_id (str): Vendor ID (hex string).
        device_id (str): Device ID (hex string).
        rev_id (str): Revision ID.
        asic_serial (str): ASIC serial number.
        num_compute_units (int): Number of compute units.
        vram_type (str): VRAM type (e.g., "HBM3", "GDDR6").
        vram_vendor (str): VRAM vendor name.
        vram_size_mb (int): VRAM size in megabytes.
        model_name (str): Board model name.
        power_cap_w (int): Current power cap in watts.
        default_power_cap_w (int): Default power cap in watts.
        min_power_cap_w (int): Minimum power cap in watts.
        max_power_cap_w (int): Maximum power cap in watts.
    """

    gpu_index: int
    bdf: str
    market_name: str
    vendor_id: str
    device_id: str
    rev_id: str
    asic_serial: str
    num_compute_units: int
    vram_type: str
    vram_vendor: str
    vram_size_mb: int
    model_name: str
    power_cap_w: int
    default_power_cap_w: int
    min_power_cap_w: int
    max_power_cap_w: int


@dataclass(frozen=True)
class GpuMetrics:
    """Real-time GPU performance metrics.

    Attributes:
        gpu_index (int): Zero-based GPU index.
        bdf (str): PCIe Bus/Device/Function address.
        gfx_activity_pct (int): Graphics engine activity (%).
        umc_activity_pct (int): Memory controller activity (%).
        mm_activity_pct (int): Multimedia engine activity (%).
        temp_edge_c (int): Edge temperature (Celsius).
        temp_hotspot_c (int): Hotspot temperature (Celsius).
        temp_vram_c (int): VRAM temperature (Celsius).
        current_power_w (int): Current socket power draw (watts).
        gfx_voltage_mv (int): Graphics voltage (millivolts).
        soc_voltage_mv (int): SoC voltage (millivolts).
        mem_voltage_mv (int): Memory voltage (millivolts).
        gfx_clock_mhz (int): Current graphics clock (MHz).
        gfx_max_clock_mhz (int): Maximum graphics clock (MHz).
        mem_clock_mhz (int): Current memory clock (MHz).
        mem_max_clock_mhz (int): Maximum memory clock (MHz).
        vram_used_mb (int): VRAM used (megabytes).
        vram_total_mb (int): VRAM total (megabytes).
    """

    gpu_index: int
    bdf: str
    gfx_activity_pct: int
    umc_activity_pct: int
    mm_activity_pct: int
    temp_edge_c: int
    temp_hotspot_c: int
    temp_vram_c: int
    current_power_w: int
    gfx_voltage_mv: int
    soc_voltage_mv: int
    mem_voltage_mv: int
    gfx_clock_mhz: int
    gfx_max_clock_mhz: int
    mem_clock_mhz: int
    mem_max_clock_mhz: int
    vram_used_mb: int
    vram_total_mb: int


@dataclass(frozen=True)
class FirmwareEntry:
    """A single firmware component.

    Attributes:
        name (str): Firmware block name.
        version (str): Firmware version string.
    """

    name: str
    version: str


@dataclass(frozen=True)
class GpuFirmwareInfo:
    """Firmware information for a GPU.

    Attributes:
        gpu_index (int): Zero-based GPU index.
        bdf (str): PCIe Bus/Device/Function address.
        vbios_name (str): VBIOS name.
        vbios_version (str): VBIOS version.
        vbios_build_date (str): VBIOS build date.
        vbios_part_number (str): VBIOS part number.
        firmware_list (list[FirmwareEntry]): List of firmware entries.
    """

    gpu_index: int
    bdf: str
    vbios_name: str
    vbios_version: str
    vbios_build_date: str
    vbios_part_number: str
    firmware_list: list[FirmwareEntry] = field(default_factory=list)


@dataclass(frozen=True)
class ProcessEntry:
    """A process using a GPU.

    Attributes:
        pid (int): Process ID.
        name (str): Process name.
        vram_usage_mb (int): VRAM usage in megabytes.
    """

    pid: int
    name: str
    vram_usage_mb: int


@dataclass(frozen=True)
class GpuProcessInfo:
    """Process information for a GPU.

    Attributes:
        gpu_index (int): Zero-based GPU index.
        bdf (str): PCIe Bus/Device/Function address.
        processes (list[ProcessEntry]): List of processes using this GPU.
    """

    gpu_index: int
    bdf: str
    processes: list[ProcessEntry] = field(default_factory=list)


@dataclass(frozen=True)
class GpuBadPageInfo:
    """Bad page and ECC error information for a GPU.

    Attributes:
        gpu_index (int): Zero-based GPU index.
        bdf (str): PCIe Bus/Device/Function address.
        correctable_ecc_count (int): Number of correctable ECC errors.
        uncorrectable_ecc_count (int): Number of uncorrectable ECC errors.
        deferred_ecc_count (int): Number of deferred ECC errors.
        bad_page_count (int): Number of retired memory pages.
    """

    gpu_index: int
    bdf: str
    correctable_ecc_count: int
    uncorrectable_ecc_count: int
    deferred_ecc_count: int
    bad_page_count: int


class AmdSmi:
    """Class to query AMD GPU system management information via the amdsmi Python API.

    Example:
        Basic usage to get the GPU driver version::

            from rocm_mcp.sysinfo import AmdSmi

            smi = AmdSmi()
            result = smi.get_driver_information()
            print(f"Driver: {result.name} {result.version}")

    Attributes:
        logger (logging.Logger): Logger for logging messages.
    """

    logger: logging.Logger

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize the AmdSmi wrapper.

        Args:
            logger (logging.Logger): Logger instance. If None, a default logger is created.
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        amdsmi.amdsmi_init()

        self.logger.info("Initialized amd-smi wrapper.")

    def __del__(self) -> None:
        """Clean up resources on deletion."""
        try:
            amdsmi.amdsmi_shut_down()
            self.logger.info("Shut down amd-smi wrapper.")
        except Exception as e:
            msg = f"Exception during amd-smi shutdown: {e}"
            self.logger.exception(msg)

    def _get_handles(self) -> list[tuple[int, Any]]:
        """Get all GPU processor handles with their zero-based indices.

        Returns:
            list[tuple[int, Any]]: List of (gpu_index, handle) tuples.

        Raises:
            RuntimeError: If no GPU processors are found.
        """
        handles = amdsmi.amdsmi_get_processor_handles()
        if not handles:
            msg = "No GPU processors found"
            self.logger.error(msg)
            raise RuntimeError(msg)
        return list(enumerate(handles))

    def _safe_query(self, func: Any, *args: Any, default: Any = None) -> Any:  # noqa: ANN401
        """Call an amdsmi function, returning a default on failure."""
        try:
            return func(*args)
        except amdsmi.AmdSmiException as e:
            self.logger.warning("%s failed: %s", getattr(func, "__name__", func), e)
            return default

    def get_driver_information(self) -> DriverInformationResult:
        """Get AMD GPU driver version via the amdsmi Python API.

        Returns:
            DriverInformationResult: The driver version, name, and date.

        Raises:
            RuntimeError: If no GPU processors are found or the query fails.
        """
        handles = self._get_handles()
        _, handle = handles[0]

        try:
            info = amdsmi.amdsmi_get_gpu_driver_info(handle)
        except amdsmi.AmdSmiException as e:
            msg = f"amdsmi query failed: {e}"
            self.logger.exception(msg)
            raise RuntimeError(msg) from e

        return DriverInformationResult(
            version=info["driver_version"],
            name=info["driver_name"],
            date=info["driver_date"],
        )

    def list_gpus(self) -> list[GpuInfo]:
        """List all AMD GPUs in the system.

        Returns:
            list[GpuInfo]: List of GPU identification info.

        Raises:
            RuntimeError: If no GPU processors are found or the query fails.
        """
        handles = self._get_handles()
        results = []
        for idx, handle in handles:
            try:
                bdf = amdsmi.amdsmi_get_gpu_device_bdf(handle)
                uuid = amdsmi.amdsmi_get_gpu_device_uuid(handle)
            except amdsmi.AmdSmiException as e:
                msg = f"amdsmi query failed for GPU {idx}: {e}"
                self.logger.exception(msg)
                raise RuntimeError(msg) from e
            results.append(GpuInfo(gpu_index=idx, bdf=bdf, uuid=uuid))
        return results

    def get_gpu_static_info(self) -> list[GpuStaticInfo]:
        """Get static hardware properties for all GPUs.

        Returns:
            list[GpuStaticInfo]: List of static GPU info.

        Raises:
            RuntimeError: If no GPU processors are found.
        """
        handles = self._get_handles()
        results = []
        for idx, handle in handles:
            bdf = self._safe_query(amdsmi.amdsmi_get_gpu_device_bdf, handle, default="N/A")
            asic = self._safe_query(amdsmi.amdsmi_get_gpu_asic_info, handle, default={})
            vram = self._safe_query(amdsmi.amdsmi_get_gpu_vram_info, handle, default={})
            board = self._safe_query(amdsmi.amdsmi_get_gpu_board_info, handle, default={})
            pcap = self._safe_query(amdsmi.amdsmi_get_power_cap_info, handle, default={})

            vram_type_val = vram.get("vram_type", "N/A")
            if hasattr(vram_type_val, "name"):
                vram_type_val = vram_type_val.name

            results.append(GpuStaticInfo(
                gpu_index=idx,
                bdf=bdf,
                market_name=asic.get("market_name", "N/A"),
                vendor_id=str(asic.get("vendor_id", "N/A")),
                device_id=str(asic.get("device_id", "N/A")),
                rev_id=str(asic.get("rev_id", "N/A")),
                asic_serial=str(asic.get("asic_serial", "N/A")),
                num_compute_units=asic.get("num_compute_units", 0),
                vram_type=str(vram_type_val),
                vram_vendor=vram.get("vram_vendor", "N/A"),
                vram_size_mb=vram.get("vram_size", 0),
                model_name=board.get("product_name", "N/A"),
                power_cap_w=pcap.get("power_cap", 0),
                default_power_cap_w=pcap.get("default_power_cap", 0),
                min_power_cap_w=pcap.get("min_power_cap", 0),
                max_power_cap_w=pcap.get("max_power_cap", 0),
            ))
        return results

    def get_gpu_metrics(self) -> list[GpuMetrics]:
        """Get real-time performance metrics for all GPUs.

        Returns:
            list[GpuMetrics]: List of GPU metrics.

        Raises:
            RuntimeError: If no GPU processors are found.
        """
        handles = self._get_handles()
        results = []
        for idx, handle in handles:
            bdf = self._safe_query(amdsmi.amdsmi_get_gpu_device_bdf, handle, default="N/A")
            activity = self._safe_query(amdsmi.amdsmi_get_gpu_activity, handle, default={})
            power = self._safe_query(amdsmi.amdsmi_get_power_info, handle, default={})
            vram_usage = self._safe_query(amdsmi.amdsmi_get_gpu_vram_usage, handle, default={})

            gfx_clk = self._safe_query(
                amdsmi.amdsmi_get_clock_info, handle, amdsmi.AmdSmiClkType.GFX, default={},
            )
            mem_clk = self._safe_query(
                amdsmi.amdsmi_get_clock_info, handle, amdsmi.AmdSmiClkType.MEM, default={},
            )

            temp_edge = self._safe_query(
                amdsmi.amdsmi_get_temp_metric, handle,
                amdsmi.AmdSmiTemperatureType.EDGE,
                amdsmi.AmdSmiTemperatureMetric.CURRENT,
                default=0,
            )
            temp_hotspot = self._safe_query(
                amdsmi.amdsmi_get_temp_metric, handle,
                amdsmi.AmdSmiTemperatureType.HOTSPOT,
                amdsmi.AmdSmiTemperatureMetric.CURRENT,
                default=0,
            )
            temp_vram = self._safe_query(
                amdsmi.amdsmi_get_temp_metric, handle,
                amdsmi.AmdSmiTemperatureType.VRAM,
                amdsmi.AmdSmiTemperatureMetric.CURRENT,
                default=0,
            )

            def _int(val: int | str) -> int:
                if isinstance(val, str):
                    return 0
                return int(val)

            results.append(GpuMetrics(
                gpu_index=idx,
                bdf=bdf,
                gfx_activity_pct=_int(activity.get("gfx_activity", 0)),
                umc_activity_pct=_int(activity.get("umc_activity", 0)),
                mm_activity_pct=_int(activity.get("mm_activity", 0)),
                temp_edge_c=_int(temp_edge),
                temp_hotspot_c=_int(temp_hotspot),
                temp_vram_c=_int(temp_vram),
                current_power_w=_int(power.get("current_socket_power", 0)),
                gfx_voltage_mv=_int(power.get("gfx_voltage", 0)),
                soc_voltage_mv=_int(power.get("soc_voltage", 0)),
                mem_voltage_mv=_int(power.get("mem_voltage", 0)),
                gfx_clock_mhz=_int(gfx_clk.get("clk", 0)),
                gfx_max_clock_mhz=_int(gfx_clk.get("max_clk", 0)),
                mem_clock_mhz=_int(mem_clk.get("clk", 0)),
                mem_max_clock_mhz=_int(mem_clk.get("max_clk", 0)),
                vram_used_mb=_int(vram_usage.get("vram_used", 0)),
                vram_total_mb=_int(vram_usage.get("vram_total", 0)),
            ))
        return results

    def get_gpu_firmware_info(self) -> list[GpuFirmwareInfo]:
        """Get firmware version information for all GPUs.

        Returns:
            list[GpuFirmwareInfo]: List of GPU firmware info.

        Raises:
            RuntimeError: If no GPU processors are found.
        """
        handles = self._get_handles()
        results = []
        for idx, handle in handles:
            bdf = self._safe_query(amdsmi.amdsmi_get_gpu_device_bdf, handle, default="N/A")
            vbios = self._safe_query(amdsmi.amdsmi_get_gpu_vbios_info, handle, default={})
            fw_info = self._safe_query(amdsmi.amdsmi_get_fw_info, handle, default={})

            fw_entries = [
                FirmwareEntry(
                    name=fw.get("fw_name", "N/A"),
                    version=fw.get("fw_version", "N/A"),
                )
                for fw in fw_info.get("fw_list", [])
            ]

            results.append(GpuFirmwareInfo(
                gpu_index=idx,
                bdf=bdf,
                vbios_name=vbios.get("name", "N/A"),
                vbios_version=vbios.get("version", "N/A"),
                vbios_build_date=vbios.get("build_date", "N/A"),
                vbios_part_number=vbios.get("part_number", "N/A"),
                firmware_list=fw_entries,
            ))
        return results

    def get_gpu_process_info(self) -> list[GpuProcessInfo]:
        """Get process information for all GPUs.

        Returns:
            list[GpuProcessInfo]: List of per-GPU process info.

        Raises:
            RuntimeError: If no GPU processors are found.
        """
        handles = self._get_handles()
        results = []
        for idx, handle in handles:
            bdf = self._safe_query(amdsmi.amdsmi_get_gpu_device_bdf, handle, default="N/A")
            proc_list = self._safe_query(amdsmi.amdsmi_get_gpu_process_list, handle, default=[])

            entries = []
            for proc in proc_list:
                mem_usage = proc.get("memory_usage", {})
                vram_mem = mem_usage.get("vram_mem", 0) if isinstance(mem_usage, dict) else 0
                entries.append(ProcessEntry(
                    pid=proc.get("pid", 0),
                    name=proc.get("name", "N/A"),
                    vram_usage_mb=vram_mem // (1024 * 1024) if isinstance(vram_mem, int) else 0,
                ))

            results.append(GpuProcessInfo(gpu_index=idx, bdf=bdf, processes=entries))
        return results

    def get_gpu_bad_page_info(self) -> list[GpuBadPageInfo]:
        """Get bad page and ECC error information for all GPUs.

        Returns:
            list[GpuBadPageInfo]: List of per-GPU bad page and ECC info.

        Raises:
            RuntimeError: If no GPU processors are found.
        """
        handles = self._get_handles()
        results = []
        for idx, handle in handles:
            bdf = self._safe_query(amdsmi.amdsmi_get_gpu_device_bdf, handle, default="N/A")
            ecc = self._safe_query(amdsmi.amdsmi_get_gpu_total_ecc_count, handle, default={})
            bad_pages = self._safe_query(amdsmi.amdsmi_get_gpu_bad_page_info, handle, default=[])

            results.append(GpuBadPageInfo(
                gpu_index=idx,
                bdf=bdf,
                correctable_ecc_count=ecc.get("correctable_count", 0),
                uncorrectable_ecc_count=ecc.get("uncorrectable_count", 0),
                deferred_ecc_count=ecc.get("deferred_count", 0),
                bad_page_count=len(bad_pages),
            ))
        return results
