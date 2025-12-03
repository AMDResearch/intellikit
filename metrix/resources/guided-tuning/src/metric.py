##############################################################################bl
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el

import sys, random, textwrap
from dataclasses import dataclass, asdict
from collections import defaultdict
from src.db import sql
from src.logger import console_debug, console_warning, console_log

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns

def div_(a, b): return a/b if b else 0

class MetricError(Exception):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message
    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, self.message)

class NotSupportedError(MetricError):
    def __str__(self):
        return f'{self.message} not supported!'

class NotFoundError(MetricError):
    def __str__(self):
        return f'{self.message} not found!'

class WaveCountZeroError(MetricError):
  pass

###################
# Helper Functions
###################
def generate_sql_query(table, columns, conditions, group_by=None):
    """
    Helper function to generate SQL queries dynamically.
    """
    query = f"SELECT {', '.join(columns)} FROM {table} WHERE {' AND '.join(conditions)}"
    if group_by:
        query += f" GROUP BY {', '.join(group_by)}"
    return query

def fetch_metrics(workload, metric_names, kernel=None, dispatch=None):
    """
    Fetch metrics from the database based on workload and a condition (kernel or dispatch).
    """
    if kernel:
      # Use dispatch IDs to filter by kernel names  
      dispatch_ids = sql(f"SELECT dispatch FROM dispatch WHERE workload={workload} AND kernel SIMILAR TO '{kernel}'")  
      if not dispatch_ids:  
        raise NotFoundError(f'Kernel(s) {kernel} not found')  
      condition = f"dispatch IN ({', '.join(str(d[0]) for d in dispatch_ids)})"  
    elif dispatch != None:
      condition = f"dispatch={dispatch}"
    else:
      raise MetricError('Need either dispatch or kernel')


    conditions = [
        "name IN (" + ", ".join("'{}'".format(name) for name in metric_names) + ")",
        f"workload={workload}",
        condition
    ]
    query = generate_sql_query("metric", ["name", "AVG(value) AS value"], conditions, group_by=["name"])
    return dict(sql(query))

def workload_name(workload):
  rs = sql(f'SELECT workload_name FROM workload WHERE workload_id={workload}')
  return rs[0][0]

def calc_pop(value, ns, peak):
  return 100 * value / (ns * peak)

###################
# Metric Functions
###################

def dur(workload, dispatch=None, kernel:list=None):
  if kernel:
    # Construct an SQL IN clause for the list of dispatches from kernel names
    dispatch_ids = sql(f"SELECT dispatch FROM dispatch WHERE workload={workload} AND kernel SIMILAR TO '{kernel}'")  
    if not dispatch_ids:  
      raise NotFoundError(f'Kernel(s) {kernel} not found')
    condition = f"dispatch IN ({', '.join(str(d[0]) for d in dispatch_ids)})"  
  elif dispatch is not None:
    condition = f"dispatch={dispatch}"
  else:
    raise MetricError('Need either dispatch or kernel')
  ns = sql(f'''
    SELECT
      kernel,
      AVG(end_ns - start_ns) as avg_dur,
      AVG(grid_size) as avg_grid_size,
      AVG(workgroup_size) as avg_workgroup_size,
      AVG(lds_per_workgroup) as avg_lds_per_workgroup,
      AVG(scratch_per_workitem) as avg_scratch_per_workitem,
      AVG(arch_vgpr) as avg_arch_vgpr,
      AVG(accum_vgpr) as avg_accum_vgpr,
      AVG(sgpr) as avg_sgpr
    FROM dispatch
    WHERE workload={workload} AND {condition}
    GROUP BY kernel
  ''')
  if kernel:
    ns_2 = sql(f'''
      SELECT
        kernel,
        count,
        pct
      FROM top
      WHERE workload={workload} AND kernel SIMILAR TO '{kernel}'
    ''')
    # Merge the results of ns and ns_2 on the column "kernel"
    ns_dict = {row[0]: row[1:] for row in ns}
    ns_2_dict = {row[0]: row[1:] for row in ns_2}

    merged_results = []
    for kernel_name, ns_values in ns_dict.items():
      if kernel_name in ns_2_dict:
        merged_results.append((kernel_name, *ns_values, *ns_2_dict[kernel_name]))
      else:
        merged_results.append((kernel_name, *ns_values, None))  # Handle missing values in ns_2

    console_debug(f"Merged results (adding pct and count column): {merged_results}")
    ns = merged_results
    
  console_debug(f"Number of matching kernels: {len(ns)}")
  try:
    result = ns[0]
  except IndexError:
    msg = f'Workload({workload})/dispatch({dispatch})'
    raise NotFoundError(msg)

  return result

def specs(workload):
  vals = sql(f'''
    SELECT
      gpu_series,
      cu_per_gpu,
      max_sclk,
      hbm_channels,
      lds_banks_per_cu,
      total_l2_chan,
      se_per_gpu,
      num_xcd
    FROM workload
    WHERE workload_id={workload}
  ''')
  # print(vals[0])
  try:
    result = vals[0]
  except IndexError:
    msg = f'Workload({workload})'
    raise NotFoundError(msg)

  return result

def l1(workload, gpu_series, cu_per_gpu, max_sclk, dispatch=None, kernel:list=None):
  METRIC_NAMES_L1 = [
    'TCP_TOTAL_CACHE_ACCESSES_sum',
    'TCP_TCC_READ_REQ_sum',
    'TCP_TCC_WRITE_REQ_sum',
    'TCP_TCC_ATOMIC_WITH_RET_REQ_sum',
    'TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum',
    'TCP_TCC_READ_REQ_LATENCY_sum',
    'TCP_TCC_WRITE_REQ_LATENCY_sum',
    'TCP_TOTAL_ACCESSES_sum',
    'TA_TOTAL_WAVEFRONTS_sum',
    'TCP_GATE_EN1_sum',
    'TCP_GATE_EN2_sum'
  ]

  vs = defaultdict(int, fetch_metrics(workload, METRIC_NAMES_L1, kernel=kernel, dispatch=dispatch))

  util = div_(
    100 * vs['TCP_GATE_EN2_sum'],
    vs['TCP_GATE_EN1_sum']
  )
  if gpu_series == 'MI200':
    bytes = vs['TCP_TOTAL_CACHE_ACCESSES_sum'] * 64
    peak_l1 = ((max_sclk / 1000) * 64) * cu_per_gpu
  elif gpu_series == 'MI300' or gpu_series == 'MI350':
    bytes = vs['TCP_TOTAL_CACHE_ACCESSES_sum'] * 128
    peak_l1 = ((max_sclk / 1000) * 128) * cu_per_gpu
  else:
    raise NotSupportedError(gpu_series)

  hr = 100 - div_(100 * (
    vs['TCP_TCC_READ_REQ_sum'] +
    vs['TCP_TCC_WRITE_REQ_sum'] +
    vs['TCP_TCC_ATOMIC_WITH_RET_REQ_sum'] +
    vs['TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum']),
    vs['TCP_TOTAL_CACHE_ACCESSES_sum'])
  
  if gpu_series == 'MI200':
    rd_lat = div_(vs['TCP_TCC_READ_REQ_LATENCY_sum'],
                  vs['TCP_TCC_READ_REQ_sum'] + vs['TCP_TCC_ATOMIC_WITH_RET_REQ_sum'])
    wr_lat = div_(vs['TCP_TCC_WRITE_REQ_LATENCY_sum'],
                  vs['TCP_TCC_WRITE_REQ_sum'] + vs['TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum'])
  elif gpu_series == 'MI300':
    # MI300 bug. Counters missing from series earlier than MI350
    rd_lat = None
    wr_lat = None
  elif gpu_series == 'MI350':
    rd_lat = vs["TCP_TCC_READ_REQ_LATENCY_sum"]
    wr_lat = vs["TCP_TCC_WRITE_REQ_LATENCY_sum"]
  coal = div_((vs['TA_TOTAL_WAVEFRONTS_sum'] * 64) * 100, (vs['TCP_TOTAL_ACCESSES_sum'] * 4))


  return bytes, hr, util, rd_lat, wr_lat, coal, peak_l1

def l2(workload, gpu_series, num_xcd, max_sclk, total_l2_chan, dispatch=None, kernel:list=None):
  METRIC_NAMES_L2 = [
    'GRBM_GUI_ACTIVE',
    'TCC_REQ_sum',
    'TCC_HIT_sum',
    'TCC_MISS_sum',
    'TCC_BUSY_sum', 
    'TCC_EA_RDREQ_LEVEL_sum',
    'TCC_EA_WRREQ_LEVEL_sum',
    'TCC_EA_RDREQ_sum',
    'TCC_EA_WRREQ_sum',
    'TCC_EA_ATOMIC_LEVEL_sum',
    'TCC_EA_ATOMIC_sum',
    'TCC_EA0_RDREQ_LEVEL_sum',
    'TCC_EA0_WRREQ_LEVEL_sum',
    'TCC_EA0_RDREQ_sum',
    'TCC_EA0_WRREQ_sum',
    'TCC_EA0_ATOMIC_LEVEL_sum',
    'TCC_EA0_ATOMIC_sum',
  ]
  vs = defaultdict(int, fetch_metrics(workload, METRIC_NAMES_L2, kernel=kernel, dispatch=dispatch))

  grbm_gui_active_per_xcd = vs['GRBM_GUI_ACTIVE'] / num_xcd
  bytes = vs['TCC_REQ_sum'] * 128
  hr = div_(100 * vs['TCC_HIT_sum'], vs['TCC_HIT_sum'] + vs['TCC_MISS_sum'])
  util = div_(
    vs['TCC_BUSY_sum'] * 100, 
    total_l2_chan * grbm_gui_active_per_xcd
  )
  rd_lat, wr_lat = 0, 0
  if gpu_series == 'MI200':
    rd_lat = div_(vs['TCC_EA_RDREQ_LEVEL_sum'], vs['TCC_EA_RDREQ_sum'])
    wr_lat = div_(vs['TCC_EA_WRREQ_LEVEL_sum'], vs['TCC_EA_WRREQ_sum'])
    peak_l2 = max_sclk * total_l2_chan * 64 / 1000
    atomic_lat = div_(vs['TCC_EA_ATOMIC_LEVEL_sum'], vs['TCC_EA_ATOMIC_sum'])
  elif gpu_series == 'MI300' or gpu_series == 'MI350':
    rd_lat = div_(vs['TCC_EA0_RDREQ_LEVEL_sum'], vs['TCC_EA0_RDREQ_sum'])
    wr_lat = div_(vs['TCC_EA0_WRREQ_LEVEL_sum'], vs['TCC_EA0_WRREQ_sum'])
    peak_l2 = max_sclk * total_l2_chan * 128 / 1000
    atomic_lat = div_(vs['TCC_EA0_ATOMIC_LEVEL_sum'], vs['TCC_EA0_ATOMIC_sum'])
  else:
    raise NotSupportedError(gpu_series)

  return bytes, hr, util, rd_lat, wr_lat, atomic_lat, peak_l2

def hbm(workload, gpu_series, dispatch=None, kernel:list=None):
  METRIC_NAMES_HBM = [
    'TCC_EA_RDREQ_32B_sum',
    'TCC_EA_RDREQ_sum',
    'TCC_EA_WRREQ_64B_sum',
    'TCC_EA_WRREQ_sum',
    'TCC_EA0_RDREQ_sum',
    'TCC_BUBBLE_sum',
    'TCC_EA0_RDREQ_32B_sum',
    'TCC_EA0_WRREQ_64B_sum',
    'TCC_EA0_WRREQ_sum',
  ]
  vs = defaultdict(int, fetch_metrics(workload, METRIC_NAMES_HBM, kernel=kernel, dispatch=dispatch))

  rd, wr = 0, 0
  if gpu_series == 'MI200':
    rd = vs['TCC_EA_RDREQ_32B_sum'] * 32 + \
      (vs['TCC_EA_RDREQ_sum'] - vs['TCC_EA_RDREQ_32B_sum']) * 64
    wr = vs['TCC_EA_WRREQ_64B_sum'] * 64 + \
      (vs['TCC_EA_WRREQ_sum'] - vs['TCC_EA_WRREQ_64B_sum']) * 32
  elif gpu_series == 'MI300' or gpu_series == 'MI350':
    rd = vs['TCC_BUBBLE_sum'] * 128 + \
      (vs['TCC_EA0_RDREQ_sum'] - vs['TCC_BUBBLE_sum'] - vs['TCC_EA0_RDREQ_32B_sum']) * 64 + \
      vs['TCC_EA0_RDREQ_32B_sum'] * 32
    wr = vs['TCC_EA0_WRREQ_64B_sum'] * 64 + \
          (vs['TCC_EA0_WRREQ_sum'] - vs['TCC_EA0_WRREQ_64B_sum']) * 32
  else:
    raise NotSupportedError(gpu_series)

  return rd, wr

def flops(workload, cu_per_gpu, max_sclk, dispatch=None, kernel:list=None):
  METRIC_NAMES_FLOPS = [
    'SQ_INSTS_VALU_INT32',
    'SQ_INSTS_VALU_INT64',
    'SQ_INSTS_VALU_ADD_F16',
    'SQ_INSTS_VALU_MUL_F16',
    'SQ_INSTS_VALU_TRANS_F16',
    'SQ_INSTS_VALU_FMA_F16',
    'SQ_INSTS_VALU_ADD_F32',
    'SQ_INSTS_VALU_MUL_F32',
    'SQ_INSTS_VALU_TRANS_F32',
    'SQ_INSTS_VALU_FMA_F32',
    'SQ_INSTS_VALU_ADD_F64',
    'SQ_INSTS_VALU_MUL_F64',
    'SQ_INSTS_VALU_TRANS_F64',
    'SQ_INSTS_VALU_FMA_F64',
    'SQ_INSTS_VALU_MFMA_MOPS_F16',
    'SQ_INSTS_VALU_MFMA_MOPS_BF16',
    'SQ_INSTS_VALU_MFMA_MOPS_F32',
    'SQ_INSTS_VALU_MFMA_MOPS_F64'
  ]
  vs = defaultdict(int, fetch_metrics(workload, METRIC_NAMES_FLOPS, kernel=kernel, dispatch=dispatch))

  fops = (64 * (
      (
        vs['SQ_INSTS_VALU_ADD_F16'] +
        vs['SQ_INSTS_VALU_MUL_F16'] +
        vs['SQ_INSTS_VALU_TRANS_F16'] +
        vs['SQ_INSTS_VALU_FMA_F16'] * 2) +
      (
        vs['SQ_INSTS_VALU_ADD_F32'] +
        vs['SQ_INSTS_VALU_MUL_F32'] +
        vs['SQ_INSTS_VALU_TRANS_F32'] +
        vs['SQ_INSTS_VALU_FMA_F32'] * 2) +
      (
        vs['SQ_INSTS_VALU_ADD_F64'] +
        vs['SQ_INSTS_VALU_MUL_F64'] +
        vs['SQ_INSTS_VALU_TRANS_F64'] +
        vs['SQ_INSTS_VALU_FMA_F64'] * 2)
    ) + 512 * (
        vs['SQ_INSTS_VALU_MFMA_MOPS_F16'] +
        vs['SQ_INSTS_VALU_MFMA_MOPS_BF16'] +
        vs['SQ_INSTS_VALU_MFMA_MOPS_F32'] +
        vs['SQ_INSTS_VALU_MFMA_MOPS_F64']
      )
  )
  iops = 64 * (vs['SQ_INSTS_VALU_INT32'] + vs['SQ_INSTS_VALU_INT64'])

  peak_flop = (max_sclk * cu_per_gpu * 64 * 2) / 1000

  return (fops, iops, peak_flop)

def ipc(workload, dispatch=None, kernel:list=None):
  METRIC_NAMES_IPC = [
    'SQ_INSTS',
    'SQ_BUSY_CU_CYCLES',
  ]
  vs = defaultdict(int, fetch_metrics(workload, METRIC_NAMES_IPC, kernel=kernel, dispatch=dispatch))

  quo = vs['SQ_BUSY_CU_CYCLES']
  return div_(vs['SQ_INSTS'],  quo)

def lds(workload, cu_per_gpu, num_xcd, max_sclk, lds_banks_per_cu, dispatch=None, kernel:list=None):
  METRIC_NAMES_LDS = [
    'SQ_INSTS_LDS',
    'SQ_LDS_BANK_CONFLICT',
    'SQ_LDS_IDX_ACTIVE',
    'SPI_RA_LDS_CU_FULL_CSN',
    'GRBM_GUI_ACTIVE'
  ]
  vs = defaultdict(int, fetch_metrics(workload, METRIC_NAMES_LDS, kernel=kernel, dispatch=dispatch))

  grbm_gui_active_per_xcd = vs['GRBM_GUI_ACTIVE'] / num_xcd
  ins = vs['SQ_INSTS_LDS']
  act = vs['SQ_LDS_IDX_ACTIVE']
  conf = vs['SQ_LDS_BANK_CONFLICT']
  quo = act - conf
  insufficient_cu_lds = div_(
    400 * vs['SPI_RA_LDS_CU_FULL_CSN'],
    grbm_gui_active_per_xcd * cu_per_gpu)
  util = div_(
    100 * vs['SQ_LDS_IDX_ACTIVE'],
    grbm_gui_active_per_xcd * cu_per_gpu
  )
  
  return (
    quo * 4 * lds_banks_per_cu,
    ins,
    util,
    div_(conf, quo),
    insufficient_cu_lds,
    max_sclk * cu_per_gpu * 0.128
  )

def wave(workload, cu_per_gpu, num_xcd, dispatch=None, kernel:list=None):
  METRIC_NAMES_WAVE = [
    'SPI_CSN_WAVE',
    'GRBM_GUI_ACTIVE',
    'SQ_INSTS',
    'SQ_WAVES',
    'SQ_WAVE_CYCLES',
    'SQ_WAIT_ANY',
    'SQ_WAIT_INST_ANY',
    'SQ_ACTIVE_INST_ANY',
    'SQ_ACCUM_PREV_HIRES_WAVES'
  ]
  vs = defaultdict(int, fetch_metrics(workload, METRIC_NAMES_WAVE, kernel=kernel, dispatch=dispatch))

  grbm_gui_active_per_xcd = vs['GRBM_GUI_ACTIVE'] / num_xcd
  fronts, waves = vs['SPI_CSN_WAVE'], vs['SQ_WAVES']
  cycles = vs['GRBM_GUI_ACTIVE']
  ins_per_wave = div_(vs['SQ_INSTS'], waves)
  wave_c= 4 * vs['SQ_WAVE_CYCLES']
  dep_wait_c= 4 * vs['SQ_WAIT_ANY']
  iss_wait_c= 4 * vs['SQ_WAIT_INST_ANY']
  act_c= 4 * vs['SQ_ACTIVE_INST_ANY']
  occ = div_(vs['SQ_ACCUM_PREV_HIRES_WAVES'], grbm_gui_active_per_xcd)
  max_waves = 32 * cu_per_gpu
  pop = (occ * 100)/max_waves
  wav = fronts if fronts else waves
  if wav == 0:
    msg = f'Wave count is 0 for workload({workload})/dispatch({dispatch})!'
    raise WaveCountZeroError(msg)

  return (wav,
          cycles,
          ins_per_wave,
          div_(wave_c, waves),
          div_(dep_wait_c, waves),
          div_(iss_wait_c, waves),
          div_(act_c, waves),
          occ,
          pop,
          max_waves)

def spi(workload, se_per_gpu, num_xcd, cu_per_gpu, dispatch=None, kernel:list=None):
  METRIC_NAMES_SPI = [
    'GRBM_SPI_BUSY',
    'GRBM_GUI_ACTIVE',
    'SPI_RA_RES_STALL_CSN',
    'SPI_RA_TMP_STALL_CSN',
    'SPI_RA_WAVE_SIMD_FULL_CSN',
    'SPI_RA_VGPR_SIMD_FULL_CSN',
    'SPI_RA_SGPR_SIMD_FULL_CSN',
    'SPI_RA_LDS_CU_FULL_CSN',
    'SPI_RA_BAR_CU_FULL_CSN',
    'SPI_RA_TGLIM_CU_FULL_CSN',
    'SPI_RA_WVLIM_STALL_CSN'
  ]
  vs = defaultdict(int, fetch_metrics(workload, METRIC_NAMES_SPI, kernel=kernel, dispatch=dispatch))
  
  grbm_gui_active_per_xcd = vs['GRBM_GUI_ACTIVE'] / num_xcd
  grbm_spi_busy_per_xcd = vs['GRBM_SPI_BUSY'] / num_xcd
  scheduler_pipe_stall_rate = div_(
    100 * vs['SPI_RA_RES_STALL_CSN'],
    se_per_gpu * grbm_spi_busy_per_xcd)
  scratch_stall_rate = div_(
    100 * vs['SPI_RA_TMP_STALL_CSN'],
    se_per_gpu * grbm_spi_busy_per_xcd)
  insuff_waveslots = div_(
    100 * vs['SPI_RA_WAVE_SIMD_FULL_CSN'],
    cu_per_gpu * grbm_gui_active_per_xcd)
  insuff_vgprs = div_(
    100 * vs['SPI_RA_VGPR_SIMD_FULL_CSN'],
    cu_per_gpu * grbm_gui_active_per_xcd)
  insuff_sgprs = div_(
    100 * vs['SPI_RA_SGPR_SIMD_FULL_CSN'],
    cu_per_gpu * grbm_gui_active_per_xcd)
  insuff_lds = div_(
    400 * vs['SPI_RA_LDS_CU_FULL_CSN'],
    cu_per_gpu * grbm_gui_active_per_xcd)
  insuff_barriers = div_(
    400 * vs['SPI_RA_BAR_CU_FULL_CSN'],
    cu_per_gpu * grbm_gui_active_per_xcd)
  limit_cu_workgroup = div_(
    400 * vs['SPI_RA_TGLIM_CU_FULL_CSN'],
    cu_per_gpu * grbm_gui_active_per_xcd)
  limit_cu_wavefront = div_(
    400 * vs['SPI_RA_WVLIM_STALL_CSN'],
    cu_per_gpu * grbm_gui_active_per_xcd)

  return (
    scheduler_pipe_stall_rate,
    scratch_stall_rate,
    insuff_waveslots,
    insuff_vgprs,
    insuff_sgprs,
    insuff_lds,
    insuff_barriers,
    limit_cu_workgroup,
    limit_cu_wavefront)

def instruction_mix(workload, waves, dispatch=None, kernel:list=None):
  METRIC_NAMES_INST_MIX = [
    'SQ_INSTS_VALU',
    'SQ_INSTS_MFMA',
    'SQ_INSTS_VMEM',
    'SQ_INSTS_FLAT_LDS_ONLY',
    'SQ_INSTS_LDS',
    'SQ_INSTS_SALU',
    'SQ_INSTS_SMEM',
    'SQ_INSTS_BRANCH'
  ]
  vs = defaultdict(int, fetch_metrics(workload, METRIC_NAMES_INST_MIX, kernel=kernel, dispatch=dispatch))
  # print(vs, waves)
  try:
    valu = int((vs['SQ_INSTS_VALU'] - vs['SQ_INSTS_MFMA'])/waves)
    vmem = int((vs['SQ_INSTS_VMEM'] - vs['SQ_INSTS_FLAT_LDS_ONLY'])/waves)
    lds = int(vs['SQ_INSTS_LDS']/waves)
    mfma = int(vs['SQ_INSTS_MFMA']/waves)
    salu = int(vs['SQ_INSTS_SALU']/waves)
    smem = int(vs['SQ_INSTS_SMEM']/waves)
    bran = int(vs['SQ_INSTS_BRANCH']/waves)
  except ZeroDivisionError:
    console_warning("Divide by zero error in instruction mix calculation")
    console_warning(f'Check workload {workload} dispatch {dispatch}!')

  # crude
  compute_mem_ratio = div_(valu + mfma + salu, max(vmem+lds, 1))

  return (
    valu,
    vmem,
    lds,
    mfma,
    salu,
    smem,
    bran,
    compute_mem_ratio
  )

@dataclass
class WorkloadDetails:
    name: str
    kernel: str
    count: int
    gpu_series: str
    cu_per_gpu: int
    max_sclk: int
    hbm_bw: float
    lds_banks_per_cu: int
    total_l2_chan: int
    se_per_gpu: int
    num_xcd: int
    grid: int
    workgroup: int

@dataclass
class Metrics:
    durations: dict
    flops: dict
    hbm: dict
    lds: dict
    l1: dict
    l2: dict
    atomics: dict
    ai: dict
    wave: dict
    ipc: dict
    cycles: dict
    allocations: dict
    stalls: dict
    instruction_mix: dict

def display_results(workload: WorkloadDetails, 
                    results: Metrics, 
                    kernel_filter: Metrics = None, 
                    W: int=98, 
                    wrap_table: bool=True):
    """
    Display the results in a formatted table using rich.
    """
    console = Console()

    def format_number(n, precision=2, metric_name=""):
      """
      Format a number with the specified decimal precision.
      If the value is None, return 'N/A' and print a warning with the metric name.
      """
      if n is None:
          console.print(f"[yellow]Warning: Missing metric value detected for '{metric_name}'. Displaying as 'N/A'.[/yellow]")
          return "N/A"
      if isinstance(n, (int, float)):
          if isinstance(n, int):
              return f"{n:,}"
          else:
              format_str = f"{{:.{precision}f}}"
              return format_str.format(n)
      return n

    # Header
    kernel_msg = f'Kernels like: {kernel_filter}' if kernel_filter else workload.kernel
    header = Text(f"{workload.name}\n({workload.grid}, {workload.workgroup}) : {kernel_msg}\nCall count: {workload.count}", style="bold")
    console.print(Panel(header, width=W))

    # Application Stats
    
    # Duration
    dur_table = Table(title="Duration", width=W)
    dur_table.add_column("Metric")
    dur_table.add_column("Value", justify="right")
    dur_table.add_row("Duration (ns)", format_number(results.durations["ns"], metric_name="Duration (ns)"))
    dur_table.add_row("Percentage (of runtime)", format_number(results.durations["pct"], metric_name="Percentage (of runtime)"))

    # Operations
    ops_table = Table(title="Operations", width=W)
    ops_table.add_column("Metric")
    ops_table.add_column("Count", justify="right")
    ops_table.add_column("Rate", justify="right")
    ops_table.add_column("PoP", justify="right")
    ops_table.add_row(
        "FLOPS",
        format_number(results.flops["f"], metric_name="FLOPS Count"),
        f"{format_number(results.flops['f']/results.durations['ns'], precision=3, metric_name='FLOPS Rate')} GFlops/s",
        f"{format_number(results.flops['flop_pop'], precision=2, metric_name='FLOPS PoP')}%"
    )
    ops_table.add_row(
        "IOPS",
        format_number(results.flops["i"], metric_name="IOPS Count"),
        f"{format_number(results.flops['i']/results.durations['ns'], precision=2, metric_name='IOPS Rate')} GIops/s",
        f"{format_number(results.flops['iop_pop'], precision=2, metric_name='IOPS PoP')}%"
    )

    # Arithmetic Intensity
    ai_table = Table(title="Arithmetic Intensity", width=W)
    ai_table.add_column("Level")
    ai_table.add_column("HBM", justify="right")
    ai_table.add_column("L2", justify="right")
    ai_table.add_column("L1", justify="right")
    ai_table.add_row(
        "AI (Flops/Byte)",
        f"{format_number(results.ai['hbm'], precision=2, metric_name='HBM AI')}",
        f"{format_number(results.ai['l2'], precision=2, metric_name='L2 AI')}",
        f"{format_number(results.ai['l1'], precision=2, metric_name='L1 AI')}"
    )

    # Instruction Mix
    mix_table = Table(title="Instruction Mix (per wave)", width=W)
    mix_table.add_column("VALU", justify="right")
    mix_table.add_column("VMEM", justify="right")
    mix_table.add_column("LDS", justify="right")
    mix_table.add_column("MFMA", justify="right")
    mix_table.add_column("SALU", justify="right")
    mix_table.add_column("SMEM", justify="right")
    mix_table.add_column("BRANCH", justify="right")
    mix_table.add_row(
        format_number(results.instruction_mix["valu"], metric_name="VALU Inst Mix"),
        format_number(results.instruction_mix["vmem"], metric_name="VMEM Inst Mix"),
        format_number(results.instruction_mix["lds"], metric_name="LDS Inst Mix"),
        format_number(results.instruction_mix["mfma"], metric_name="MFMA Inst Mix"),
        format_number(results.instruction_mix["salu"], metric_name="SALU Inst Mix"),
        format_number(results.instruction_mix["smem"], metric_name="SMEM Inst Mix"),
        format_number(results.instruction_mix["branch"], metric_name="Branch Inst Mix")
    )
    

    # Memory (Split into multiple tables based on available metrics)
    
    # HBM Table
    hbm_table = Table(title="HBM Memory", width=W)
    hbm_table.add_column("Type")
    hbm_table.add_column("Count", justify="right")
    hbm_table.add_column("Rate", justify="right")
    hbm_table.add_column("PoP", justify="right")
    hbm_table.add_row(
        "Read",
        format_number(results.hbm["rd"], metric_name="HBM Rd Count"),
        f"{format_number(results.hbm['rd']/results.durations['ns'], precision=2, metric_name='HBM Rd Rate')} GB/s",
        f"{format_number(results.hbm['rd_pop'], precision=2, metric_name='HBM Rd PoP')}%"
    )
    hbm_table.add_row(
        "Write", 
        format_number(results.hbm["wr"], metric_name='HBM Wr Count'),
        f"{format_number(results.hbm['wr']/results.durations['ns'], precision=2, metric_name='HBM Wr Rate')} GB/s",
        f"{format_number(results.hbm['wr_pop'], precision=2, metric_name='HBM Wr PoP')}%"
    )

    # LDS Table 
    lds_table = Table(title="LDS Memory", width=W)
    lds_table.add_column("Count", justify="right")
    lds_table.add_column("Rate", justify="right")
    lds_table.add_column("PoP", justify="right")
    lds_table.add_column("Utilization", justify="right")
    lds_table.add_column("Requests", justify="right")
    lds_table.add_column("Bank Conflict Rate", justify="right")
    lds_table.add_column("Insufficient CU LDS", justify="right")
    lds_table.add_row(
        format_number(results.lds["lds"], metric_name="LDS Count"),
        f"{format_number(results.lds['lds']/results.durations['ns'], precision=2, metric_name='LDS Rate')} GB/s",
        f"{format_number(results.lds['pop'], precision=2, metric_name='LDS PoP')}%",
        f"{format_number(results.lds['util'], precision=2, metric_name='LDS Utilization')}%",
        f"{format_number(int(results.lds['req']), metric_name='LDS Requests')}",
        f"{format_number(results.lds['bc'], precision=2, metric_name='LDS Bank Conflicts')}%",
        f"{format_number(results.lds['ins_cu_lds'], precision=2, metric_name='Insufficient CU LDS')}%"
    )

    # L1/L2 Cache Table
    cache_table = Table(title="L1/L2 Memory", width=W)
    cache_table.add_column("Level")
    cache_table.add_column("Count", justify="right")
    cache_table.add_column("Rate", justify="right") 
    cache_table.add_column("PoP", justify="right")
    cache_table.add_column("Hit Rate", justify="right")
    cache_table.add_column("Utilization", justify="right")
    cache_table.add_column("Read Latency", justify="right")
    cache_table.add_column("Write Latency", justify="right")
    cache_table.add_column("Coalescing Rate", justify="right")
    cache_table.add_row(
        "L1",
        format_number(results.l1["l1"], metric_name="L1 Count"),
        f"{format_number(results.l1['l1']/results.durations['ns'], precision=2, metric_name='L1 Rate')} GB/s",
        f"{format_number(results.l1['pop'], precision=2, metric_name='L1 PoP')}%",
        f"{format_number(results.l1['hr'], precision=2, metric_name='L1 Hit Rate')}%",
        f"{format_number(results.l1['util'], precision=2, metric_name='L1 Utilization')}%",
        f"{format_number(results.l1['rd_lat'], precision=2, metric_name='L1 Rd Latency')}",
        f"{format_number(results.l1['wr_lat'], precision=2, metric_name='L1 Wr Latency')}",
        f"{format_number(results.l1['coal'], precision=2, metric_name='L1 Coalescing Rate')}%"
    )
    cache_table.add_row(
        "L2",
        format_number(results.l2["l2"], metric_name="L2 Count"),
        f"{format_number(results.l2['l2']/results.durations['ns'], precision=2, metric_name='L2 Rate')} GB/s", 
        f"{format_number(results.l2['pop'], precision=2, metric_name='L2 PoP')}%",
        f"{format_number(results.l2['hr'], precision=2, metric_name='L2 Hit Rate')}%",
        f"{format_number(results.l2['util'], precision=2, metric_name='L2 Utilization')}%",
        f"{format_number(results.l2['rd_lat'], precision=2, metric_name='L2 Rd Latency')}",
        f"{format_number(results.l2['wr_lat'], precision=2, metric_name='L2 Wr Latency')}",
        "N/A" # No coalescing rate for L2
    )

    atomics_table = Table(title="L2 <--> Fabric", width=W)
    atomics_table.add_column("Atomic Latency", justify="right")
    atomics_table.add_row(
       f"{format_number(results.atomics['atomic_lat'], precision=2, metric_name='Atomic Latency')}"
    )
    
    # Wave Statistics
    wave_table = Table(title="Wave Statistics", width=W)
    wave_table.add_column("Total Wavefronts", justify="right")
    wave_table.add_column("Wavefront Occupancy", justify="right")
    wave_table.add_column("Instructions per Wavefront", justify="right")
    wave_table.add_row(
        format_number(results.wave["count"], metric_name="Total Wavefronts"),
        format_number(results.wave['occupancy'], metric_name="Wavefront Occupancy"),
        format_number(results.wave["ins_per_wave"], metric_name="Instructions per Wavefront")
    )

    # Cycles
    cycles_table = Table(title="Cycles", width=W)
    cycles_table.add_column("Metric")
    cycles_table.add_column("Value", justify="right")
    cycles_table.add_column("Active Cycles", justify="right")
    cycles_table.add_column("Active %", justify="right")
    cycles_table.add_row(
        "Wave Cycles",
        f"{format_number(results.cycles['wave_cycles'], precision=2, metric_name='Wave Cycles Count')} Cycles per wave",
        f"{format_number(results.cycles['active'], precision=2, metric_name='Wave Cycle Active')} Cycles",
        f"{format_number(div_(results.cycles['active']*100, results.cycles['wave_cycles']), precision=2, metric_name='Wave Cycles Active')}%"
    )
    cycles_table.add_row("Dep Wait Cycles", format_number(results.cycles["dep_wait"], metric_name="Dep Wait Cycles Count"), "-", "-")
    cycles_table.add_row("Issue Wait Cycles", format_number(results.cycles["issue_wait"], metric_name="Issue Wait Cycles Count"), "-", "-")

    # Resource Allocations
    alloc_table = Table(title="Resource Allocations", width=W)
    alloc_table.add_column("VGPRs", justify="right")
    alloc_table.add_column("AGPRs", justify="right")
    alloc_table.add_column("SGPRs", justify="right")
    alloc_table.add_column("LDS Allocation", justify="right")
    alloc_table.add_column("Scratch Allocation", justify="right")
    alloc_table.add_row(
        format_number(results.allocations["vgpr"], metric_name="VGPRs"),
        format_number(results.allocations["agpr"], metric_name="AGPRs"),
        format_number(results.allocations["sgpr"], metric_name="SGPRs"),
        format_number(results.allocations["lds"], metric_name="LDS Res Alloc"),
        format_number(results.allocations["scratch"], metric_name="Scratch Res Alloc")
    )

    # Stalls
    stalls_table = Table(title="Stalls", width=W)
    stalls_table.add_column("Metric")
    stalls_table.add_column("Rate", justify="right")
    stalls_table.add_row("Scheduler-Pipe Stall Rate", f"{format_number(results.stalls['scheduler_pipe'], precision=2, metric_name='Scheduler-Pipe Stall Rate')}%")
    stalls_table.add_row("Scratch Stall Rate", f"{format_number(results.stalls['scratch'], precision=2, metric_name='Scratch Stall Rate')}%")
    stalls_table.add_row("", "", style="dim")  # Add a dim line as separator
    stalls_table.add_row("Insufficient Resources:", "")
    stalls_table.add_row("  SIMD Waveslots", f"{format_number(results.stalls['waveslots'], precision=2, metric_name='SIMD Waveslots')}%")
    stalls_table.add_row("  SIMD VGPRs", f"{format_number(results.stalls['vgprs'], precision=2, metric_name='SIMD VGPRs')}%") 
    stalls_table.add_row("  SIMD SGPRs", f"{format_number(results.stalls['sgprs'], precision=2, metric_name='SIMD SGPRs')}%")
    stalls_table.add_row("  SIMD LDS", f"{format_number(results.stalls['lds'], precision=2, metric_name='SIMD LDS')}%")
    stalls_table.add_row("  CU Barriers", f"{format_number(results.stalls['barriers'], precision=2, metric_name='CU Barriers')}%")
    stalls_table.add_row("", "", style="dim")  # Add a dim line as separator
    stalls_table.add_row("Reached CU Workgroup Limit", f"{format_number(results.stalls['workgroup_limit'], precision=2, metric_name='Reached CU Workgroup Limit')}%")
    stalls_table.add_row("Reached CU Wavefront Limit", f"{format_number(results.stalls['wavefront_limit'], precision=2, metric_name='Reached CU Wavefront Limit')}%")
    
    if wrap_table:
       console.print("[bold blue]Application Stats[/bold blue]")
       console.print(Columns([dur_table, ops_table]))
       console.print(Columns([ai_table, mix_table]))
       console.print("[bold green]Memory Usage[/bold green]")
       console.print(Columns([hbm_table, lds_table]))
       console.print(Columns([cache_table, atomics_table]))
       console.print("[bold yellow]Wavefront Stats[/bold yellow]")
       console.print(Columns([wave_table, cycles_table]))
       console.print(Columns([alloc_table, stalls_table]))
    else:
       console.print("[bold blue]Application Stats[/bold blue]")
       console.print(dur_table)
       console.print(ops_table)
       console.print(ai_table)
       console.print(mix_table)
       console.print("[bold green]Memory Usage[/bold green]")
       console.print(hbm_table)
       console.print(lds_table)
       console.print(cache_table)
       console.print(atomics_table)
       console.print("[bold yellow]Wavefront Stats[/bold yellow]")
       console.print(wave_table)
       console.print(cycles_table)
       console.print(alloc_table)
       console.print(stalls_table)


def speed_of_light(w,
                   d=None, 
                   k=None, 
                   display=True, 
                   wrap_table=True) -> dict:
    # Fetch workload details
    name = workload_name(w)
    gpu_series, cu_per_gpu, max_sclk, hbm_channels, lds_banks_per_cu, total_l2_chan, se_per_gpu, num_xcd = specs(w)
    hbm_bw = max_sclk/1000 * 32 * hbm_channels

    count=None
    if k:
      ker, ns, grid, wg, lds_per_workitem, scratch_per_workitem, arch_vgpr, accum_vgpr, sgpr, count, pct = dur(w, dispatch=d, kernel=k)
    else:
      ker, ns, grid, wg, lds_per_workitem, scratch_per_workitem, arch_vgpr, accum_vgpr, sgpr = dur(w, dispatch=d, kernel=k)
    workload = WorkloadDetails(name, ker, count, gpu_series, cu_per_gpu, max_sclk, hbm_bw, lds_banks_per_cu, total_l2_chan, se_per_gpu, num_xcd, grid, wg)

    # Calculate metrics
    f_, i_, peak_flop = flops(w, cu_per_gpu, max_sclk, dispatch=d, kernel=k)
    hbm_rd_, hbm_wr_ = hbm(w, gpu_series, dispatch=d, kernel=k)
    lds_, lds_req_, lds_util_, bc_, ins_cu_lds_, peak_lds = lds(w, cu_per_gpu, num_xcd, max_sclk, lds_banks_per_cu, dispatch=d, kernel=k)
    l1_, l1_hr, l1_util, l1_rd_lat, l1_wr_lat, l1_coal, peak_l1 = l1(w, gpu_series, cu_per_gpu, max_sclk, dispatch=d, kernel=k)
    l2_, l2_hr, l2_util, l2_rd_lat, l2_wr_lat, l2_atomic_lat, peak_l2 = l2(w, gpu_series, num_xcd, max_sclk, total_l2_chan, dispatch=d, kernel=k)
    ipc_ = ipc(w, dispatch=d, kernel=k)
    wv = wave(w, cu_per_gpu, num_xcd, dispatch=d, kernel=k)
    im = instruction_mix(w, wv[0], dispatch=d, kernel=k)
    stalls = spi(w, se_per_gpu, num_xcd, cu_per_gpu, dispatch=d, kernel=k)

    # Populate results
    metrics = Metrics(
        durations={"ns": ns, "pct": pct if k else None},
        flops={"f": f_, "i": i_, "flop_pop": calc_pop(f_, ns, peak_flop), "iop_pop": calc_pop(i_, ns, peak_flop)},
        hbm={"rd": hbm_rd_, "wr": hbm_wr_, "rd_pop": calc_pop(hbm_rd_, ns, hbm_bw), "wr_pop": calc_pop(hbm_wr_, ns, hbm_bw)},
        lds={"lds": lds_, "req": lds_req_, "util": lds_util_, "bc": bc_, "ins_cu_lds": ins_cu_lds_, "pop": calc_pop(lds_, ns, peak_lds), "peak": peak_lds},
        l1={"l1": l1_, "hr": l1_hr, "util": l1_util, "rd_lat": l1_rd_lat, "wr_lat": l1_wr_lat, "coal": l1_coal, "pop": calc_pop(l1_, ns, peak_l1), "peak": peak_l1},
        l2={"l2": l2_, "hr": l2_hr, "util": l2_util, "rd_lat": l2_rd_lat, "wr_lat": l2_wr_lat, "pop": calc_pop(l2_, ns, peak_l2), "peak": peak_l2},
        atomics={"atomic_lat": l2_atomic_lat},
        ai={"hbm": div_(f_, hbm_rd_ + hbm_wr_), "l2": div_(f_, l2_), "l1": div_(f_, l1_)},
        wave={"count": wv[0], "cycles": wv[1], "ins_per_wave": wv[2], "wave_cycles": wv[3], "dep_wait_cycles": wv[4], "issue_wait_cycles": wv[5], "active_cycles": wv[6], "occupancy": wv[7], "pop": wv[8], "max_waves": wv[9]},
        ipc={"value": ipc_},
        cycles={"wave_cycles": wv[3], "active": wv[6], "dep_wait": wv[4], "issue_wait": wv[5]},
        allocations={"vgpr": arch_vgpr, "agpr": accum_vgpr, "sgpr": sgpr, "lds": lds_per_workitem, "scratch": scratch_per_workitem},
        stalls={"scheduler_pipe": stalls[0], "scratch": stalls[1], "waveslots": stalls[2], "vgprs": stalls[3], "sgprs": stalls[4], "lds": stalls[5], "barriers": stalls[6], "workgroup_limit": stalls[7], "wavefront_limit": stalls[8]},
        instruction_mix={"valu": im[0], "vmem": im[1], "lds": im[2], "mfma": im[3], "salu": im[4], "smem": im[5], "branch": im[6], "compute_mem_ratio": im[7]}
    )

    # Display results
    if display:
      display_results(workload, metrics, kernel_filter=k, wrap_table=wrap_table)

    # Merge workload details and metrics into a single dictionary
    merged_results = {**asdict(workload), **asdict(metrics)}
    return merged_results

def all_dispatches():
  rs = sql('select distinct workload, dispatch from dispatch')
  return rs

def todo(n=5):
  disps = all_dispatches()
  sels = random.choices(disps, k=n)
  for (w,d) in sels:
    console_log(w,d)
    name = workload_name(w)
    console_log(f'--- {name} ---')
    speed_of_light(w,d)

def sort_f(f):
  console_log(f'Running sort on all dispatches, this could take some time ...')
  raw = [(w, d, f((w,d))) for w,d in all_dispatches()]
  raw.sort(key=lambda x:x[2], reverse=True)
  for w,d,o in raw:
    console_log(f'{w},{d},{o:.2f}')

def top_occu():
  sort_f(lambda x: wave(x[0],x[1])[8])

def top_l2():
  def l2_pop(w, d):
    (_, ns, *_) = dur(w, d)
    gpu, _, max_sclk, _, _, total_l2_chan, *_ = specs(w)
    l2_, _, _, _, peak_l2 = l2(w, d, gpu, max_sclk, total_l2_chan)
    return 100 * l2_ / (ns * peak_l2)

  sort_f(lambda x: l2_pop(x[0], x[1]) )

def top_active_cycle():
  def active_cycle(w, d):
    _, cu_per_gpu, *_, num_xcd = specs(w)
    v = wave(w, d, cu_per_gpu, num_xcd)
    return div_(v[6]*100, v[3])

  sort_f(lambda x: active_cycle(x[0], x[1]) )

def top_flop():
  def flop_only(w, d):
    (_, ns, *_) = dur(w, d)
    _, cu_per_gpu, max_sclk, *_ = specs(w)
    f, i, _ = flops(w, d, cu_per_gpu, max_sclk)
    return div_(f, ns)

  sort_f(lambda x: flop_only(x[0], x[1]) )

def top_ins_per_cycle():
  def effective_ins_per_cycle(x):
    w, d = x
    _, cu_per_gpu, *_, num_xcd = specs(w)
    wv = wave(w, d, cu_per_gpu, num_xcd)
    im = instruction_mix(w, d, wv[0])
    # return div_(sum(list(im)[:-1]), wv[3]) # ignore BRANCH ops
    return div_(sum(list(im)), wv[3])

  sort_f(lambda x: effective_ins_per_cycle(x))

# top_ins_per_cycle(); sys.exit(0)

def calc_diff(vals):
  if len(vals) < 2: return 0.0

  v1, v2 = vals[0][2], vals[1][2]

  m, d = max(v1, v2), abs(v1-v2)
  if m == 0: return 0.0

  return 100 * d / m

def metric_diff(k1, k2, cutoff=0.05):
  (w1, d1), (w2, d2) = k1, k2
  m = defaultdict(list)

  ms = sql(f'''
    select workload, dispatch, name, value
    from metric
    where
      workload = {w1} and dispatch={d1}
      or
      workload = {w2} and dispatch={d2}
  ''')
  for w, d, name, value in ms:
    if name.find('[') == -1:
      m[name] += (w, d, value),

  l = [(k, m[k], calc_diff(m[k])) for k in m if calc_diff(m[k]) > cutoff]
  l.sort(key=lambda x:x[2], reverse=True)

  for n,d,t in l: console_log(f'{n:<40} {t:<6.2f} {d}')