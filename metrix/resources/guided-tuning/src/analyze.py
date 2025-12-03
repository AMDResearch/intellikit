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

import sys
from collections import defaultdict
from src.metric import speed_of_light, div_
from src.db import sql

MESSAGE = {
  'Low Expected Occupancy::Launch Parameter':
    '''Expected Occupancy is determined by launch parameters and kernel resource allocation. If this number is too low (< 60% of max waves), the work for GPU is not enough to keep it busy. Please consider to tune launch parameters to increase this number.
    ''',
  'Low Achieved Occupancy::Resource::Waveslot':
    '''This is probably due to the following resource contention:
	    . Insufficient waveslots
%s
    ''',
  'Low Achieved Occupancy::Resource::VGPR':
    '''This is probably due to the following resource contention:
	    . Insufficient VGPRs/AGPRs

%s
    ''',
  'Low Achieved Occupancy::Resource::SGPR':
    '''This is probably due to the following resource contention:
	    . Insufficient SGPRs
%s
    ''',
  'Low Achieved Occupancy::Resource::LDS':
    '''This is probably due to the following resource contention:
	    . Insufficient LDS

%s
    ''',
  'Low Achieved Occupancy::Resource::Barrier':
    '''This is probably due to the following resource contention:
	    . Insufficient Barriers
%s
    ''',
  'Low Achieved Occupancy::Resource::Workgroup Limit':
    '''This is probably due to the following resource contention:
	    . Workgroup limit reached
%s
    ''',
  'Low Achieved Occupancy::Resource::Wavefront Limit':
    '''This is probably due to the following resource contention:
	    . Wavefront limit reached
%s
    ''',
  'Low Achieved Occupancy::Imbalance':
    '''No resource contentions were observed. It is probably caused by the following:
      . imbalance of wave scheduling among shader engines and/or compute units
      . imbalance of computation across threads, i.e, early exits
To identify this, other tools like SQTT or some manual instrumentation tools can be used.
    '''
}

def analyze(workload, dispatch, silent):

  M = speed(workload, dispatch, not silent)
  name, kernel = M['name'], M['k']
  kernel = kernel if len(kernel) < 80 else kernel[:80] + '...'

  print(f'''====== ANALYSIS ======
  Workload: {name}
  Kernel:   {kernel:}
        ''')

  occ_achieved, occ_max, _ = occupancy(M)
  # effective_data_size = input('What is the effective I/O data size (in bytes) for the target kernel?\n')
  # access_pattern_check(workload, dispatch, calc(effective_data_size))
  char(M, occ_achieved/occ_max)

def wave_char(M):
  im, wv, l1 = M['im'], M['waves'][0], M['l1_']
  vmem, c2m = im[1], im[-1]

  return f'''
    Compute/Memory Ratio:       {c2m:.2f}
    L1 Data (bytes) per wave:   {int(div_(l1, wv)):,d}
    VMEM Instructions per wave: {int(vmem):,d}
    '''

def l1_char(M):
  MSG = '''
L1 uncoalesced memory access is high. Please consider the following possible solutions:
    1. make sure that threads in a warp access contiguous memory locations
    2. use better data structures (i.e. structure of arrays)
    3. align data properly
    4. optimize memory access patterns
    5. use shared memory (LDS)
  '''
  l1_coal = M['l1_coal']
  return MSG if l1_coal < 60.0 else ''

def char(M, occ):
  def _cap(x):
    # return min(99.99, x/occ)
    return min(99.99, x)

  W = M['W']
  print(f'''{"=" * W}\nMain Characteristics:{wave_char(M)}''')

  flop = (M['flop_pop'], 'FLOPS')
  iop  = (M['iop_pop'], 'IOPS')
  op = sorted([flop, iop], reverse=True)

  hbm_pop = (_cap(M['hbm_rd_pop'] + M['hbm_wr_pop']), 'HBM')
  lds_pop = (_cap(M['lds_pop']), 'LDS')
  l1_pop = (_cap(M['l1_pop']), 'L1')
  l2_pop = (_cap(M['l2_pop']), 'L2')
  mem = sorted([hbm_pop, lds_pop, l1_pop, l2_pop], reverse=True)

  overall = sorted([
    (op[0][0], f'    Compute bound:              {op[0][0]:.2f}% ({op[0][1]})'),
    (mem[0][0], f'    Memory bound:               {mem[0][0]:.2f}% ({mem[0][1]})')
  ], reverse=True)

  if overall[0][0] >= 60.0:
    print(overall[0][1])
  else:
    print(f'\nGPU utilization is low.')

def calc(int_or_expr):
  ret = 0
  try:
    ret = int(int_or_expr)
  except ValueError:
    try:
      ret = eval(int_or_expr)
    except Exception:
      print(f'Invalid expression for input: {int_or_expr}. Aborting.')
      sys.exit(1)
  return ret

def access_pattern_check(workload, dispatch, effective_data_size):
  ms = sql(f'''
  select name, value
  from metric
  where
    workload = {workload} and dispatch={dispatch}
    and name in (
      'TCP_TOTAL_CACHE_ACCESSES_sum',
    )
  ''')
  effective_data_size = max(4, effective_data_size) # make it nonzero
  vs = dict(ms)
  vs = defaultdict(int, vs)
  act = vs['TCP_TOTAL_CACHE_ACCESSES_sum'] * 64
  ratio = act / effective_data_size * 100
  print(f'\nActual cache bytes: {act}, ratio: {ratio:.2f}%')

def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()

WAVEFRONT_SIZE = 64

def register_occupancy_next_step(vgprs):
  MI_VGPR_OCCUPANCY = [
    # upper bound of num VGPRs, occupancy per CU
    (256, 8),
    (168, 12),
    (128, 16),
    (96, 20),
    (80, 24),
    (72, 28),
    (64, 32),
  ]
  for n, o in MI_VGPR_OCCUPANCY:
    if vgprs > n: return (n, o)

  return MI_VGPR_OCCUPANCY[-1]

def expected_occupancy(M):
  expected = M['v'][9]
  block_size = M['wg']
  grid_size = M['grid']/block_size
  block_size /= WAVEFRONT_SIZE
  vgprs = max(M['arch_vgpr'] + M['accum_vgpr'], 1)
  lds = max(M['lds_per_workitem'], 1) # make it non-zero
  cu_per_gpu = M['cu_per_gpu']
  launched_waves = M['v'][0]

  wave_per_cu_by_vgpr_limit = min(
    32,
    max(int(512 / vgprs), 1) * 4)

  # block_per_cu_by_lds_limit = int(65536 / lds)
  block_per_cu_by_lds_limit = 65536 / lds
  # del M; print(locals())
  # breakpoint()
  improve = ''
  if wave_per_cu_by_vgpr_limit < 32 or block_per_cu_by_lds_limit * block_size < 32:
    # limited by VGPR
    if wave_per_cu_by_vgpr_limit <= block_per_cu_by_lds_limit * block_size:
      block_per_cu = int(wave_per_cu_by_vgpr_limit / block_size)
      expected = cu_per_gpu * min(
        block_per_cu * (grid_size/block_size),
        wave_per_cu_by_vgpr_limit)
      improved_occ = register_occupancy_next_step(vgprs)
      improve = f'''Occupancy can be improved to {improved_occ[1] * cu_per_gpu}
      if VGPR/AGPR registers count({vgprs}) can be lowered to <= {improved_occ[0]}
      (Precondition: kernel is not bottlenecked by memory access)'''
    # limited by LDS
    else:
      expected = block_per_cu_by_lds_limit * block_size \
        * min(cu_per_gpu, grid_size/block_size)

  return min(launched_waves, expected), improve

def occupancy(M):
  def _not_within(a, b, percentage):
    return abs(a - b) / max(a, b) > percentage

  w = M['wave']
  # to fix
  launched_waves, achieved_occ, achieved_pop, max_wave = w[0], w[7], w[8], w[9]
  launched_waves_ = min(launched_waves, max_wave)
  expected_occ, improve = expected_occupancy(M)
  W = M['W']
  SL = '-' * W + '\n'
  print(f'''{"=" * W}\nOccupancy:
    expected: {expected_occ}   achieved: {achieved_occ:.2f}({achieved_pop:.2f}%)''')

  if launched_waves_/max_wave < 0.5:
    print(f"{SL}Based on launched wavefronts({launched_waves}):")
    print(f"{MESSAGE['Low Expected Occupancy::Launch Parameter']}")
  elif achieved_occ/max_wave < 0.6:
    print(f'''{SL}The achieved occupancy({achieved_occ:.2f}) < 60% of the peak({max_wave}). The expected occupancy is {expected_occ}.''')
    X = [
         ('insuff_waveslots', 'Waveslot'),
         ('insuff_vgprs', 'VGPR'),
         ('insuff_sgprs', 'SGPR'),
         ('insuff_lds', 'LDS'),
         ('limit_cu_workgroup', 'Workgroup Limit'),
         ('limit_cu_wavefront', 'Wavefront Limit')]
    xs = sorted([(M[x], x, y) for (x,y) in X], reverse=True)
    # print(xs)
    if xs[0][0] < 20.0:
      if _not_within(achieved_occ, expected_occ, 0.1):
        print(f"{SL}{MESSAGE['Low Achieved Occupancy::Imbalance']}")
    else:
      print(f"{SL}{MESSAGE[f'Low Achieved Occupancy::Resource::{xs[0][2]}'] % improve}")

  return (achieved_occ, max_wave, expected_occ)

def run_analyze(args):
  analyze(args.workload_id, args.dispatch_id, args.silence)

# if __name__ == '__main__':
#   import argparse
#   parser = argparse.ArgumentParser(description='Analyze the performance of a workload.')
#   parser.add_argument('workload_id', type=int, help='The workload ID to analyze.')
#   parser.add_argument('dispatch_id', type=int, help='The dispatch ID to analyze.')
#   parser.add_argument('--silence', action='store_true', help='Suppress output messages.')

#   args = parser.parse_args()
#   run_analyze(args)
    