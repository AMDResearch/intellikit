# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

from unittest.mock import MagicMock, patch

import pytest

from rocm_mcp import (
    AmdSmi,
    DriverInformationResult,
)
from rocm_mcp.sysinfo.amd_smi import (
    FirmwareEntry,
    GpuBadPageInfo,
    GpuFirmwareInfo,
    GpuInfo,
    GpuMetrics,
    GpuProcessInfo,
    GpuStaticInfo,
    ProcessEntry,
)

SAMPLE_DRIVER_INFO = {
    "driver_name": "amdgpu",
    "driver_version": "6.16.13",
    "driver_date": "2015/01/01 00:00",
}

SAMPLE_BDF = "0000:03:00.0"
SAMPLE_UUID = "GPU-abcdef12-3456"

SAMPLE_ASIC_INFO = {
    "market_name": "Instinct MI300X",
    "vendor_id": "0x1002",
    "device_id": "0x74a0",
    "rev_id": "0x00",
    "asic_serial": "0x1234567890ABCDEF",
    "num_compute_units": 304,
    "vendor_name": "AMD",
}

SAMPLE_VRAM_INFO = {
    "vram_type": MagicMock(name="HBM3"),
    "vram_vendor": "SK Hynix",
    "vram_size": 196608,
}

SAMPLE_BOARD_INFO = {
    "product_name": "MI300X OAM",
    "product_serial": "SN123",
}

SAMPLE_POWER_CAP_INFO = {
    "power_cap": 750,
    "default_power_cap": 750,
    "min_power_cap": 0,
    "max_power_cap": 750,
}

SAMPLE_ACTIVITY = {
    "gfx_activity": 85,
    "umc_activity": 42,
    "mm_activity": 10,
}

SAMPLE_POWER_INFO = {
    "current_socket_power": 320,
    "gfx_voltage": 900,
    "soc_voltage": 850,
    "mem_voltage": 1200,
}

SAMPLE_GFX_CLK = {"clk": 2100, "max_clk": 2400}
SAMPLE_MEM_CLK = {"clk": 1600, "max_clk": 2000}

SAMPLE_TEMPS = [55, 72, 48]

SAMPLE_VRAM_USAGE = {"vram_used": 65536, "vram_total": 196608}

SAMPLE_VBIOS_INFO = {
    "name": "VBIOS-MI300X",
    "version": "1.0.0",
    "build_date": "2024-01-15",
    "part_number": "PN-12345",
}

SAMPLE_FW_INFO = {
    "fw_list": [
        {"fw_name": "SMU", "fw_version": "13.0.6"},
        {"fw_name": "CP_ME", "fw_version": "46"},
    ],
}

SAMPLE_PROCESS_LIST = [
    {
        "pid": 1234,
        "name": "python3",
        "memory_usage": {"vram_mem": 1048576, "gtt_mem": 0, "cpu_mem": 0},
    },
    {
        "pid": 5678,
        "name": "rocm_app",
        "memory_usage": {"vram_mem": 2097152, "gtt_mem": 0, "cpu_mem": 0},
    },
]

SAMPLE_ECC_COUNT = {
    "correctable_count": 5,
    "uncorrectable_count": 1,
    "deferred_count": 2,
}

SAMPLE_BAD_PAGES = [
    {"address": "0x1000", "status": "retired"},
    {"address": "0x2000", "status": "retired"},
]


# ---------------------------------------------------------------------------
# get_driver_information tests
# ---------------------------------------------------------------------------


@patch("rocm_mcp.sysinfo.amd_smi.amdsmi")
def test_get_driver_information_success(mock_amdsmi: MagicMock) -> None:
    """Test getting driver version via amdsmi."""
    mock_handle = MagicMock()
    mock_amdsmi.amdsmi_get_processor_handles.return_value = [mock_handle]
    mock_amdsmi.amdsmi_get_gpu_driver_info.return_value = SAMPLE_DRIVER_INFO

    smi = AmdSmi()
    result = smi.get_driver_information()
    del smi

    assert isinstance(result, DriverInformationResult)
    assert result.version == SAMPLE_DRIVER_INFO["driver_version"]
    assert result.name == SAMPLE_DRIVER_INFO["driver_name"]
    assert result.date == SAMPLE_DRIVER_INFO["driver_date"]
    mock_amdsmi.amdsmi_init.assert_called_once()
    mock_amdsmi.amdsmi_shut_down.assert_called_once()


@patch("rocm_mcp.sysinfo.amd_smi.amdsmi")
def test_get_driver_information_no_gpus(mock_amdsmi: MagicMock) -> None:
    """Test that RuntimeError is raised when no GPU processors are found."""
    mock_amdsmi.amdsmi_get_processor_handles.return_value = []

    smi = AmdSmi()
    with pytest.raises(RuntimeError, match="No GPU processors found"):
        smi.get_driver_information()
    del smi

    mock_amdsmi.amdsmi_shut_down.assert_called_once()


@patch("rocm_mcp.sysinfo.amd_smi.amdsmi")
def test_get_driver_information_amdsmi_error(mock_amdsmi: MagicMock) -> None:
    """Test that RuntimeError is raised when amdsmi query fails."""
    mock_handle = MagicMock()
    mock_amdsmi.amdsmi_get_processor_handles.return_value = [mock_handle]
    mock_amdsmi.AmdSmiException = Exception
    mock_amdsmi.amdsmi_get_gpu_driver_info.side_effect = Exception("query failed")

    smi = AmdSmi()
    with pytest.raises(RuntimeError, match="amdsmi query failed"):
        smi.get_driver_information()


@patch("rocm_mcp.sysinfo.amd_smi.amdsmi")
def test_shutdown_called_on_success(mock_amdsmi: MagicMock) -> None:
    """Test that amdsmi_shut_down is always called, even on success."""
    mock_handle = MagicMock()
    mock_amdsmi.amdsmi_get_processor_handles.return_value = [mock_handle]
    mock_amdsmi.amdsmi_get_gpu_driver_info.return_value = SAMPLE_DRIVER_INFO

    smi = AmdSmi()
    smi.get_driver_information()
    del smi

    mock_amdsmi.amdsmi_shut_down.assert_called_once()


@patch("rocm_mcp.sysinfo.amd_smi.amdsmi")
def test_uses_first_gpu_handle(mock_amdsmi: MagicMock) -> None:
    """Test that driver info is queried from the first GPU handle."""
    handle_0 = MagicMock(name="gpu0")
    handle_1 = MagicMock(name="gpu1")
    mock_amdsmi.amdsmi_get_processor_handles.return_value = [handle_0, handle_1]
    mock_amdsmi.amdsmi_get_gpu_driver_info.return_value = SAMPLE_DRIVER_INFO

    smi = AmdSmi()
    smi.get_driver_information()
    del smi

    mock_amdsmi.amdsmi_get_gpu_driver_info.assert_called_once_with(handle_0)


# ---------------------------------------------------------------------------
# list_gpus tests
# ---------------------------------------------------------------------------


@patch("rocm_mcp.sysinfo.amd_smi.amdsmi")
def test_list_gpus_single(mock_amdsmi: MagicMock) -> None:
    """Test listing a single GPU."""
    mock_handle = MagicMock()
    mock_amdsmi.amdsmi_get_processor_handles.return_value = [mock_handle]
    mock_amdsmi.amdsmi_get_gpu_device_bdf.return_value = SAMPLE_BDF
    mock_amdsmi.amdsmi_get_gpu_device_uuid.return_value = SAMPLE_UUID

    smi = AmdSmi()
    result = smi.list_gpus()
    del smi

    assert len(result) == 1
    assert isinstance(result[0], GpuInfo)
    assert result[0].gpu_index == 0
    assert result[0].bdf == SAMPLE_BDF
    assert result[0].uuid == SAMPLE_UUID


@patch("rocm_mcp.sysinfo.amd_smi.amdsmi")
def test_list_gpus_multiple(mock_amdsmi: MagicMock) -> None:
    """Test listing multiple GPUs."""
    bdf_1 = "0000:43:00.0"
    h0, h1 = MagicMock(), MagicMock()
    mock_amdsmi.amdsmi_get_processor_handles.return_value = [h0, h1]
    mock_amdsmi.amdsmi_get_gpu_device_bdf.side_effect = [SAMPLE_BDF, bdf_1]
    mock_amdsmi.amdsmi_get_gpu_device_uuid.side_effect = ["UUID-0", "UUID-1"]

    smi = AmdSmi()
    result = smi.list_gpus()
    del smi

    assert len(result) == len([h0, h1])
    assert result[0].gpu_index == 0
    assert result[1].gpu_index == 1
    assert result[1].bdf == bdf_1


@patch("rocm_mcp.sysinfo.amd_smi.amdsmi")
def test_list_gpus_no_gpus(mock_amdsmi: MagicMock) -> None:
    """Test that RuntimeError is raised when no GPUs are found."""
    mock_amdsmi.amdsmi_get_processor_handles.return_value = []

    smi = AmdSmi()
    with pytest.raises(RuntimeError, match="No GPU processors found"):
        smi.list_gpus()
    del smi


@patch("rocm_mcp.sysinfo.amd_smi.amdsmi")
def test_list_gpus_query_error(mock_amdsmi: MagicMock) -> None:
    """Test that RuntimeError propagates when BDF/UUID query fails."""
    mock_handle = MagicMock()
    mock_amdsmi.amdsmi_get_processor_handles.return_value = [mock_handle]
    mock_amdsmi.AmdSmiException = Exception
    mock_amdsmi.amdsmi_get_gpu_device_bdf.side_effect = Exception("bdf failed")

    smi = AmdSmi()
    with pytest.raises(RuntimeError, match="amdsmi query failed"):
        smi.list_gpus()


# ---------------------------------------------------------------------------
# get_gpu_static_info tests
# ---------------------------------------------------------------------------


@patch("rocm_mcp.sysinfo.amd_smi.amdsmi")
def test_get_gpu_static_info_success(mock_amdsmi: MagicMock) -> None:
    """Test getting static GPU info."""
    mock_handle = MagicMock()
    mock_amdsmi.amdsmi_get_processor_handles.return_value = [mock_handle]
    mock_amdsmi.AmdSmiException = Exception
    mock_amdsmi.amdsmi_get_gpu_device_bdf.return_value = SAMPLE_BDF
    mock_amdsmi.amdsmi_get_gpu_asic_info.return_value = SAMPLE_ASIC_INFO
    mock_amdsmi.amdsmi_get_gpu_vram_info.return_value = SAMPLE_VRAM_INFO
    mock_amdsmi.amdsmi_get_gpu_board_info.return_value = SAMPLE_BOARD_INFO
    mock_amdsmi.amdsmi_get_power_cap_info.return_value = SAMPLE_POWER_CAP_INFO

    smi = AmdSmi()
    result = smi.get_gpu_static_info()
    del smi

    assert len(result) == 1
    info = result[0]
    assert isinstance(info, GpuStaticInfo)
    assert info.market_name == SAMPLE_ASIC_INFO["market_name"]
    assert info.num_compute_units == SAMPLE_ASIC_INFO["num_compute_units"]
    assert info.vram_size_mb == SAMPLE_VRAM_INFO["vram_size"]
    assert info.model_name == SAMPLE_BOARD_INFO["product_name"]
    assert info.power_cap_w == SAMPLE_POWER_CAP_INFO["power_cap"]
    assert info.max_power_cap_w == SAMPLE_POWER_CAP_INFO["max_power_cap"]


@patch("rocm_mcp.sysinfo.amd_smi.amdsmi")
def test_get_gpu_static_info_partial_failure(mock_amdsmi: MagicMock) -> None:
    """Test that partial sub-query failures use defaults."""
    mock_handle = MagicMock()
    mock_amdsmi.amdsmi_get_processor_handles.return_value = [mock_handle]
    mock_amdsmi.AmdSmiException = Exception
    mock_amdsmi.amdsmi_get_gpu_device_bdf.return_value = SAMPLE_BDF
    mock_amdsmi.amdsmi_get_gpu_asic_info.return_value = SAMPLE_ASIC_INFO
    mock_amdsmi.amdsmi_get_gpu_vram_info.side_effect = Exception("unsupported")
    mock_amdsmi.amdsmi_get_gpu_board_info.return_value = SAMPLE_BOARD_INFO
    mock_amdsmi.amdsmi_get_power_cap_info.return_value = SAMPLE_POWER_CAP_INFO

    smi = AmdSmi()
    result = smi.get_gpu_static_info()
    del smi

    info = result[0]
    assert info.vram_type == "N/A"
    assert info.vram_size_mb == 0
    assert info.market_name == SAMPLE_ASIC_INFO["market_name"]


@patch("rocm_mcp.sysinfo.amd_smi.amdsmi")
def test_get_gpu_static_info_vram_type_enum(mock_amdsmi: MagicMock) -> None:
    """Test that vram_type enum is converted to its name."""
    mock_handle = MagicMock()
    mock_amdsmi.amdsmi_get_processor_handles.return_value = [mock_handle]
    mock_amdsmi.AmdSmiException = Exception
    mock_amdsmi.amdsmi_get_gpu_device_bdf.return_value = SAMPLE_BDF
    mock_amdsmi.amdsmi_get_gpu_asic_info.return_value = SAMPLE_ASIC_INFO

    vram_type_enum = MagicMock()
    vram_type_enum.name = "HBM3"
    mock_amdsmi.amdsmi_get_gpu_vram_info.return_value = {
        "vram_type": vram_type_enum,
        "vram_vendor": "SK Hynix",
        "vram_size": 196608,
    }
    mock_amdsmi.amdsmi_get_gpu_board_info.return_value = SAMPLE_BOARD_INFO
    mock_amdsmi.amdsmi_get_power_cap_info.return_value = SAMPLE_POWER_CAP_INFO

    smi = AmdSmi()
    result = smi.get_gpu_static_info()
    del smi

    assert result[0].vram_type == vram_type_enum.name


# ---------------------------------------------------------------------------
# get_gpu_metrics tests
# ---------------------------------------------------------------------------


def _setup_metrics_mocks(mock_amdsmi: MagicMock) -> None:
    """Set up common mocks for metrics tests."""
    mock_handle = MagicMock()
    mock_amdsmi.amdsmi_get_processor_handles.return_value = [mock_handle]
    mock_amdsmi.AmdSmiException = Exception
    mock_amdsmi.amdsmi_get_gpu_device_bdf.return_value = SAMPLE_BDF


@patch("rocm_mcp.sysinfo.amd_smi.amdsmi")
def test_get_gpu_metrics_success(mock_amdsmi: MagicMock) -> None:
    """Test getting GPU metrics."""
    _setup_metrics_mocks(mock_amdsmi)
    mock_amdsmi.amdsmi_get_gpu_activity.return_value = SAMPLE_ACTIVITY
    mock_amdsmi.amdsmi_get_power_info.return_value = SAMPLE_POWER_INFO
    mock_amdsmi.amdsmi_get_gpu_vram_usage.return_value = SAMPLE_VRAM_USAGE
    mock_amdsmi.amdsmi_get_clock_info.side_effect = [SAMPLE_GFX_CLK, SAMPLE_MEM_CLK]
    mock_amdsmi.amdsmi_get_temp_metric.side_effect = list(SAMPLE_TEMPS)

    smi = AmdSmi()
    result = smi.get_gpu_metrics()
    del smi

    assert len(result) == 1
    m = result[0]
    assert isinstance(m, GpuMetrics)
    assert m.gfx_activity_pct == SAMPLE_ACTIVITY["gfx_activity"]
    assert m.umc_activity_pct == SAMPLE_ACTIVITY["umc_activity"]
    assert m.temp_edge_c == SAMPLE_TEMPS[0]
    assert m.temp_hotspot_c == SAMPLE_TEMPS[1]
    assert m.temp_vram_c == SAMPLE_TEMPS[2]
    assert m.current_power_w == SAMPLE_POWER_INFO["current_socket_power"]
    assert m.gfx_clock_mhz == SAMPLE_GFX_CLK["clk"]
    assert m.mem_clock_mhz == SAMPLE_MEM_CLK["clk"]
    assert m.vram_used_mb == SAMPLE_VRAM_USAGE["vram_used"]
    assert m.vram_total_mb == SAMPLE_VRAM_USAGE["vram_total"]


@patch("rocm_mcp.sysinfo.amd_smi.amdsmi")
def test_get_gpu_metrics_string_activity_value(mock_amdsmi: MagicMock) -> None:
    """Test that string 'N/A' activity values become 0."""
    _setup_metrics_mocks(mock_amdsmi)
    mock_amdsmi.amdsmi_get_gpu_activity.return_value = {
        "gfx_activity": "N/A",
        "umc_activity": SAMPLE_ACTIVITY["umc_activity"],
        "mm_activity": "N/A",
    }
    mock_amdsmi.amdsmi_get_power_info.return_value = SAMPLE_POWER_INFO
    mock_amdsmi.amdsmi_get_gpu_vram_usage.return_value = SAMPLE_VRAM_USAGE
    mock_amdsmi.amdsmi_get_clock_info.side_effect = [SAMPLE_GFX_CLK, SAMPLE_MEM_CLK]
    mock_amdsmi.amdsmi_get_temp_metric.side_effect = list(SAMPLE_TEMPS)

    smi = AmdSmi()
    result = smi.get_gpu_metrics()
    del smi

    assert result[0].gfx_activity_pct == 0
    assert result[0].umc_activity_pct == SAMPLE_ACTIVITY["umc_activity"]
    assert result[0].mm_activity_pct == 0


@patch("rocm_mcp.sysinfo.amd_smi.amdsmi")
def test_get_gpu_metrics_partial_failure(mock_amdsmi: MagicMock) -> None:
    """Test that partial failures use defaults for metrics."""
    _setup_metrics_mocks(mock_amdsmi)
    mock_amdsmi.amdsmi_get_gpu_activity.side_effect = Exception("unsupported")
    mock_amdsmi.amdsmi_get_power_info.return_value = SAMPLE_POWER_INFO
    mock_amdsmi.amdsmi_get_gpu_vram_usage.return_value = SAMPLE_VRAM_USAGE
    mock_amdsmi.amdsmi_get_clock_info.side_effect = [SAMPLE_GFX_CLK, SAMPLE_MEM_CLK]
    mock_amdsmi.amdsmi_get_temp_metric.side_effect = list(SAMPLE_TEMPS)

    smi = AmdSmi()
    result = smi.get_gpu_metrics()
    del smi

    assert result[0].gfx_activity_pct == 0
    assert result[0].current_power_w == SAMPLE_POWER_INFO["current_socket_power"]


# ---------------------------------------------------------------------------
# get_gpu_firmware_info tests
# ---------------------------------------------------------------------------


@patch("rocm_mcp.sysinfo.amd_smi.amdsmi")
def test_get_gpu_firmware_info_success(mock_amdsmi: MagicMock) -> None:
    """Test getting firmware info."""
    mock_handle = MagicMock()
    mock_amdsmi.amdsmi_get_processor_handles.return_value = [mock_handle]
    mock_amdsmi.AmdSmiException = Exception
    mock_amdsmi.amdsmi_get_gpu_device_bdf.return_value = SAMPLE_BDF
    mock_amdsmi.amdsmi_get_gpu_vbios_info.return_value = SAMPLE_VBIOS_INFO
    mock_amdsmi.amdsmi_get_fw_info.return_value = SAMPLE_FW_INFO

    smi = AmdSmi()
    result = smi.get_gpu_firmware_info()
    del smi

    assert len(result) == 1
    fw = result[0]
    assert isinstance(fw, GpuFirmwareInfo)
    assert fw.vbios_name == SAMPLE_VBIOS_INFO["name"]
    assert fw.vbios_version == SAMPLE_VBIOS_INFO["version"]
    assert fw.vbios_build_date == SAMPLE_VBIOS_INFO["build_date"]
    assert len(fw.firmware_list) == len(SAMPLE_FW_INFO["fw_list"])
    assert isinstance(fw.firmware_list[0], FirmwareEntry)
    assert fw.firmware_list[0].name == SAMPLE_FW_INFO["fw_list"][0]["fw_name"]
    assert fw.firmware_list[0].version == SAMPLE_FW_INFO["fw_list"][0]["fw_version"]
    assert fw.firmware_list[1].name == SAMPLE_FW_INFO["fw_list"][1]["fw_name"]


@patch("rocm_mcp.sysinfo.amd_smi.amdsmi")
def test_get_gpu_firmware_info_no_fw_list(mock_amdsmi: MagicMock) -> None:
    """Test firmware info when fw_info query fails."""
    mock_handle = MagicMock()
    mock_amdsmi.amdsmi_get_processor_handles.return_value = [mock_handle]
    mock_amdsmi.AmdSmiException = Exception
    mock_amdsmi.amdsmi_get_gpu_device_bdf.return_value = SAMPLE_BDF
    mock_amdsmi.amdsmi_get_gpu_vbios_info.return_value = SAMPLE_VBIOS_INFO
    mock_amdsmi.amdsmi_get_fw_info.side_effect = Exception("unsupported")

    smi = AmdSmi()
    result = smi.get_gpu_firmware_info()
    del smi

    assert result[0].firmware_list == []


# ---------------------------------------------------------------------------
# get_gpu_process_info tests
# ---------------------------------------------------------------------------


@patch("rocm_mcp.sysinfo.amd_smi.amdsmi")
def test_get_gpu_process_info_success(mock_amdsmi: MagicMock) -> None:
    """Test getting process info."""
    mock_handle = MagicMock()
    mock_amdsmi.amdsmi_get_processor_handles.return_value = [mock_handle]
    mock_amdsmi.AmdSmiException = Exception
    mock_amdsmi.amdsmi_get_gpu_device_bdf.return_value = SAMPLE_BDF
    mock_amdsmi.amdsmi_get_gpu_process_list.return_value = SAMPLE_PROCESS_LIST

    smi = AmdSmi()
    result = smi.get_gpu_process_info()
    del smi

    assert len(result) == 1
    gpu = result[0]
    assert isinstance(gpu, GpuProcessInfo)
    assert len(gpu.processes) == len(SAMPLE_PROCESS_LIST)
    assert isinstance(gpu.processes[0], ProcessEntry)
    assert gpu.processes[0].pid == SAMPLE_PROCESS_LIST[0]["pid"]
    assert gpu.processes[0].name == SAMPLE_PROCESS_LIST[0]["name"]
    vram_bytes_0 = SAMPLE_PROCESS_LIST[0]["memory_usage"]["vram_mem"]
    assert gpu.processes[0].vram_usage_mb == vram_bytes_0 // (1024 * 1024)
    assert gpu.processes[1].pid == SAMPLE_PROCESS_LIST[1]["pid"]
    vram_bytes_1 = SAMPLE_PROCESS_LIST[1]["memory_usage"]["vram_mem"]
    assert gpu.processes[1].vram_usage_mb == vram_bytes_1 // (1024 * 1024)


@patch("rocm_mcp.sysinfo.amd_smi.amdsmi")
def test_get_gpu_process_info_no_processes(mock_amdsmi: MagicMock) -> None:
    """Test process info when no processes are running on GPU."""
    mock_handle = MagicMock()
    mock_amdsmi.amdsmi_get_processor_handles.return_value = [mock_handle]
    mock_amdsmi.AmdSmiException = Exception
    mock_amdsmi.amdsmi_get_gpu_device_bdf.return_value = SAMPLE_BDF
    mock_amdsmi.amdsmi_get_gpu_process_list.return_value = []

    smi = AmdSmi()
    result = smi.get_gpu_process_info()
    del smi

    assert result[0].processes == []


@patch("rocm_mcp.sysinfo.amd_smi.amdsmi")
def test_get_gpu_process_info_query_failure(mock_amdsmi: MagicMock) -> None:
    """Test process info when process list query fails."""
    mock_handle = MagicMock()
    mock_amdsmi.amdsmi_get_processor_handles.return_value = [mock_handle]
    mock_amdsmi.AmdSmiException = Exception
    mock_amdsmi.amdsmi_get_gpu_device_bdf.return_value = SAMPLE_BDF
    mock_amdsmi.amdsmi_get_gpu_process_list.side_effect = Exception("no permission")

    smi = AmdSmi()
    result = smi.get_gpu_process_info()
    del smi

    assert result[0].processes == []


# ---------------------------------------------------------------------------
# get_gpu_bad_page_info tests
# ---------------------------------------------------------------------------


@patch("rocm_mcp.sysinfo.amd_smi.amdsmi")
def test_get_gpu_bad_page_info_success(mock_amdsmi: MagicMock) -> None:
    """Test getting bad page info."""
    mock_handle = MagicMock()
    mock_amdsmi.amdsmi_get_processor_handles.return_value = [mock_handle]
    mock_amdsmi.AmdSmiException = Exception
    mock_amdsmi.amdsmi_get_gpu_device_bdf.return_value = SAMPLE_BDF
    mock_amdsmi.amdsmi_get_gpu_total_ecc_count.return_value = SAMPLE_ECC_COUNT
    mock_amdsmi.amdsmi_get_gpu_bad_page_info.return_value = SAMPLE_BAD_PAGES

    smi = AmdSmi()
    result = smi.get_gpu_bad_page_info()
    del smi

    assert len(result) == 1
    bp = result[0]
    assert isinstance(bp, GpuBadPageInfo)
    assert bp.correctable_ecc_count == SAMPLE_ECC_COUNT["correctable_count"]
    assert bp.uncorrectable_ecc_count == SAMPLE_ECC_COUNT["uncorrectable_count"]
    assert bp.deferred_ecc_count == SAMPLE_ECC_COUNT["deferred_count"]
    assert bp.bad_page_count == len(SAMPLE_BAD_PAGES)


@patch("rocm_mcp.sysinfo.amd_smi.amdsmi")
def test_get_gpu_bad_page_info_no_errors(mock_amdsmi: MagicMock) -> None:
    """Test bad page info when there are no ECC errors or bad pages."""
    mock_handle = MagicMock()
    mock_amdsmi.amdsmi_get_processor_handles.return_value = [mock_handle]
    mock_amdsmi.AmdSmiException = Exception
    mock_amdsmi.amdsmi_get_gpu_device_bdf.return_value = SAMPLE_BDF
    mock_amdsmi.amdsmi_get_gpu_total_ecc_count.return_value = {
        "correctable_count": 0,
        "uncorrectable_count": 0,
        "deferred_count": 0,
    }
    mock_amdsmi.amdsmi_get_gpu_bad_page_info.return_value = []

    smi = AmdSmi()
    result = smi.get_gpu_bad_page_info()
    del smi

    assert result[0].correctable_ecc_count == 0
    assert result[0].bad_page_count == 0


@patch("rocm_mcp.sysinfo.amd_smi.amdsmi")
def test_get_gpu_bad_page_info_ecc_failure(mock_amdsmi: MagicMock) -> None:
    """Test bad page info when ECC query fails uses defaults."""
    mock_handle = MagicMock()
    mock_amdsmi.amdsmi_get_processor_handles.return_value = [mock_handle]
    mock_amdsmi.AmdSmiException = Exception
    mock_amdsmi.amdsmi_get_gpu_device_bdf.return_value = SAMPLE_BDF
    mock_amdsmi.amdsmi_get_gpu_total_ecc_count.side_effect = Exception("unsupported")
    mock_amdsmi.amdsmi_get_gpu_bad_page_info.return_value = SAMPLE_BAD_PAGES

    smi = AmdSmi()
    result = smi.get_gpu_bad_page_info()
    del smi

    assert result[0].correctable_ecc_count == 0
    assert result[0].uncorrectable_ecc_count == 0
    assert result[0].bad_page_count == len(SAMPLE_BAD_PAGES)
