# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

from unittest.mock import MagicMock, patch

import pytest

from rocm_mcp import AmdSmi, DriverInformationResult

SAMPLE_DRIVER_INFO = {
    "driver_name": "amdgpu",
    "driver_version": "6.16.13",
    "driver_date": "2015/01/01 00:00",
}


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
    assert result.version == "6.16.13"
    assert result.name == "amdgpu"
    assert result.date == "2015/01/01 00:00"
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
