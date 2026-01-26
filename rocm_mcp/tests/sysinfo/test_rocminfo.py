# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.


from subprocess import CalledProcessError
from unittest.mock import MagicMock, patch

import pytest

from rocm_mcp import DeviceType, Rocminfo, RocminfoResult

# Sample rocminfo output for testing
SAMPLE_ROCMINFO_OUTPUT = """ROCk module is loaded
=====================
HSA System Attributes
=====================
Runtime Version:         1.1
System Timestamp Freq.:  1000.000000MHz

==========
HSA Agents
==========
*******
Agent 1
*******
  Name:                    AMD Ryzen 9 7950X 16-Core Processor
  Uuid:                    CPU-XX
  Marketing Name:          AMD Ryzen 9 7950X 16-Core Processor
  Vendor Name:             CPU
  Feature:                 None specified
  Profile:                 FULL_PROFILE
  Float Round Mode:        NEAR
  Max Queue Number:        0(0x0)
  Queue Min Size:          0(0x0)
  Queue Max Size:          0(0x0)
  Queue Type:              MULTI
  Node:                    0
  Device Type:             CPU
  Cache Info:
    L1:                      32768(0x8000) KB
  Chip ID:                 0(0x0)
  Cacheline Size:          64(0x40)
  Max Clock Freq. (MHz):   5879
  BDFID:                   0
  Internal Node ID:        0
  Compute Unit:            32
  SIMDs per CU:            0
  Shader Engines:          0
  Shader Arrs. per Eng.:   0
  WatchPts on Addr. Ranges:1
  Features:                None
*******
Agent 2
*******
  Name:                    gfx1100
  Uuid:                    GPU-XX
  Marketing Name:          AMD Radeon RX 7900 XTX
  Vendor Name:             AMD
  Feature:                 KERNEL_DISPATCH
  Profile:                 BASE_PROFILE
  Float Round Mode:        NEAR
  Max Queue Number:        128(0x80)
  Queue Min Size:          64(0x40)
  Queue Max Size:          131072(0x20000)
  Queue Type:              MULTI
  Node:                    1
  Device Type:             GPU
  Cache Info:
    L1:                      32(0x20) KB
    L2:                      6144(0x1800) KB
  Chip ID:                 29772(0x744c)
  Cacheline Size:          64(0x40)
  Max Clock Freq. (MHz):   2680
  BDFID:                   2816
  Internal Node ID:        1
  Compute Unit:            96
  SIMDs per CU:            2
  Shader Engines:          6
  Shader Arrs. per Eng.:   2
  WatchPts on Addr. Ranges:4
  Features:                KERNEL_DISPATCH
  Fast F16 Operation:      TRUE
  Wavefront Size:          32(0x20)
  Workgroup Max Size:      1024(0x400)
*** Done ***
"""


def test_rocminfo_init_default() -> None:
    """Test Rocminfo initialization with default executable."""
    rocminfo = Rocminfo()
    assert rocminfo.rocminfo_exe == "rocminfo"


def test_rocminfo_init_custom_path() -> None:
    """Test Rocminfo initialization with custom path."""
    custom_path = "/opt/rocm/bin/rocminfo"
    rocminfo = Rocminfo(rocminfo=custom_path)
    assert rocminfo.rocminfo_exe == custom_path


def test_rocminfo_init_env_variable() -> None:
    """Test Rocminfo initialization with environment variable."""
    env_path = "/custom/path/to/rocminfo"
    with patch.dict("os.environ", {"OMNIKIT_ROCMINFO": env_path}):
        rocminfo = Rocminfo()
        assert rocminfo.rocminfo_exe == env_path


@patch("subprocess.run")
def test_get_agents_success(mock_run: MagicMock) -> None:
    """Test successful execution and parsing of rocminfo."""
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=SAMPLE_ROCMINFO_OUTPUT,
        stderr="",
    )

    rocminfo = Rocminfo()
    result = rocminfo.get_agents()

    assert isinstance(result, RocminfoResult)
    assert len(result.agents) == 2  # noqa: PLR2004
    assert result.raw_output == SAMPLE_ROCMINFO_OUTPUT

    # Check CPU agent (Agent 1)
    cpu_agent = result.agents[0]
    assert cpu_agent.agent_number == 1
    assert cpu_agent.name == "AMD Ryzen 9 7950X 16-Core Processor"
    assert cpu_agent.uuid == "CPU-XX"
    assert cpu_agent.marketing_name == "AMD Ryzen 9 7950X 16-Core Processor"
    assert cpu_agent.vendor_name == "CPU"
    assert cpu_agent.device_type == DeviceType.CPU
    assert cpu_agent.compute_units == 32  # noqa: PLR2004
    assert cpu_agent.max_clock_freq == 5879  # noqa: PLR2004
    assert cpu_agent.profile == "FULL_PROFILE"

    # Check GPU agent (Agent 2)
    gpu_agent = result.agents[1]
    assert gpu_agent.agent_number == 2  # noqa: PLR2004
    assert gpu_agent.name == "gfx1100"
    assert gpu_agent.uuid == "GPU-XX"
    assert gpu_agent.marketing_name == "AMD Radeon RX 7900 XTX"
    assert gpu_agent.vendor_name == "AMD"
    assert gpu_agent.device_type == DeviceType.GPU
    assert gpu_agent.compute_units == 96  # noqa: PLR2004
    assert gpu_agent.max_clock_freq == 2680  # noqa: PLR2004
    assert gpu_agent.profile == "BASE_PROFILE"


@patch("subprocess.run")
def test_get_agents_file_not_found(mock_run: MagicMock) -> None:
    """Test when rocminfo executable is not found."""
    mock_run.side_effect = FileNotFoundError("rocminfo not found")

    rocminfo = Rocminfo()
    with pytest.raises(FileNotFoundError, match="rocminfo executable not found"):
        rocminfo.get_agents()


@patch("subprocess.run")
def test_get_agents_execution_failure(mock_run: MagicMock) -> None:
    """Test when rocminfo execution fails."""
    mock_run.side_effect = CalledProcessError(1, ["rocminfo"], stderr="Error")

    rocminfo = Rocminfo()
    with pytest.raises(RuntimeError, match="rocminfo execution failed"):
        rocminfo.get_agents()


@patch("subprocess.run")
def test_get_agents_empty_output(mock_run: MagicMock) -> None:
    """Test parsing of empty rocminfo output."""
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="",
        stderr="",
    )

    rocminfo = Rocminfo()
    result = rocminfo.get_agents()

    assert isinstance(result, RocminfoResult)
    assert len(result.agents) == 0
    assert result.raw_output == ""


@patch("subprocess.run")
def test_get_agents_partial_data(mock_run: MagicMock) -> None:
    """Test parsing of rocminfo output with partial agent data."""
    partial_output = """*******
Agent 1
*******
  Name:                    TestAgent
  Device Type:             GPU
*** Done ***
"""
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=partial_output,
        stderr="",
    )

    rocminfo = Rocminfo()
    result = rocminfo.get_agents()

    assert isinstance(result, RocminfoResult)
    assert len(result.agents) == 1
    agent = result.agents[0]
    assert agent.agent_number == 1
    assert agent.name == "TestAgent"
    assert agent.device_type == DeviceType.GPU
    assert agent.compute_units is None
    assert agent.max_clock_freq is None
