"""Capture orchestrator — runs an application under libkerncap.so.

Sets up the environment, launches the application under LD_PRELOAD
(rocprofiler-sdk registration), and returns the capture directory
containing the VA-faithful snapshot.
"""

import logging
import os
import subprocess
from typing import List, Optional

logger = logging.getLogger(__name__)


def run_capture(
    kernel_name: str,
    cmd: List[str],
    output_dir: str,
    dispatch: int = -1,
    timeout: int = 300,
    language: Optional[str] = None,
) -> str:
    """Run the application under libkerncap.so and capture kernel data.

    The capture produces the VA-faithful format:
      dispatch.json, kernarg.bin, kernel.hsaco,
      memory_regions.json, memory/region_*.bin

    Parameters
    ----------
    kernel_name : str
        Kernel name (or substring) to capture.
    cmd : list[str]
        Application command to execute.
    output_dir : str
        Directory for capture output.
    dispatch : int
        Dispatch index to capture (-1 = first match).
    timeout : int
        Maximum seconds to wait for the application.
    language : str, optional
        Kernel language ("hip" or "triton").  When "triton", a
        Python-level capture is used instead of HSA interception.

    Returns
    -------
    str
        Path to the capture output directory.
    """
    if language == "triton":
        from kerncap.triton_capture import run_triton_capture

        return run_triton_capture(
            kernel_name=kernel_name,
            cmd=cmd,
            output_dir=output_dir,
            dispatch=dispatch,
            timeout=timeout,
        )

    from kerncap import _get_lib_path

    lib_path = _get_lib_path()
    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    # Strip legacy HSA tool variables that can conflict with LD_PRELOAD-based capture
    env.pop("HSA_TOOLS_LIB", None)
    env.pop("HSA_TOOLS_REPORT_LOAD_FAILURE", None)
    if "LD_PRELOAD" in env:
        env["LD_PRELOAD"] = lib_path + ":" + env["LD_PRELOAD"]
    else:
        env["LD_PRELOAD"] = lib_path
    env["KERNCAP_KERNEL"] = kernel_name
    env["KERNCAP_OUTPUT"] = output_dir
    env["KERNCAP_CAPTURE_CHILD"] = "1"

    if dispatch >= 0:
        env["KERNCAP_DISPATCH"] = str(dispatch)

    try:
        proc = subprocess.run(
            cmd,
            env=env,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Application did not complete within {timeout}s")

    dispatch_file = os.path.join(output_dir, "dispatch.json")
    meta_file = os.path.join(output_dir, "metadata.json")

    if not os.path.exists(dispatch_file) and not os.path.exists(meta_file):
        stdout_preview = proc.stdout[:2000] if proc.stdout else "N/A"
        stderr_preview = proc.stderr[:2000] if proc.stderr else "N/A"
        raise RuntimeError(
            f"Capture did not produce output in {output_dir}. "
            f"App stdout: {stdout_preview}\n"
            f"App stderr: {stderr_preview}"
        )

    return output_dir
