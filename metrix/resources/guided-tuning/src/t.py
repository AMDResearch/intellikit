#!/usr/bin/env python3

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

import sys, random
from collections import defaultdict
from src.db import sql
from src.metric import *

def all_dispatches():
  rs = sql('select distinct workload, dispatch from dispatch')
  return rs

def todo(n=5):
  disps = all_dispatches()
  sels = random.choices(disps, k=n)
  for (w,d) in sels:
    print(w,d)
    name = workload_name(w)
    print(f'--- {name} ---')
    speed_of_light(w,dispatch=d)

def sort_f(f):
  raw = [(w, d, f((w,d))) for w,d in all_dispatches()]
  raw.sort(key=lambda x:x[2], reverse=True)
  for w,d,o in raw:
    print(f'{w},{d},{o:.2f}')

def top_occu():
  sort_f(lambda x: wave(x[0],x[1])[8])

def top_l2():
  def l2_pop(w, d):
    (_, ns, *_) = dur(w, d)
    l2_ = l2(w, d)
    _, max_sclk, _, total_l2_chan, *_ = specs(w)
    peak_l2 = max_sclk * total_l2_chan * 64 / 1000
    return 100 * l2_ / (ns * peak_l2)

  sort_f(lambda x: l2_pop(x[0], x[1]) )

def top_active_cycle():
  def active_cycle(w, d):
    v = wave(w, d)
    return div_(v[6]*100, v[3])

  sort_f(lambda x: active_cycle(x[0], x[1]) )

def top_flop():
  def flop_only(w, d):
    (_, ns, *_) = dur(w, d)
    f = flops(w, d)
    return div_(f, ns)

  sort_f(lambda x: flop_only(x[0], x[1]) )

def top_ai():
  def ai(w, d):
    (hbm_rd_, hbm_wr_), f_ = hbm(w,d), flops(w,d)
    return div_(f_, hbm_rd_ + hbm_wr_)

  sort_f(lambda x: ai(x[0], x[1]) )

def top_ipc():
  sort_f(lambda x: ipc(x[0], x[1]))

# top_ipc(); sys.exit(0)

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

  W = 80
  print(f'{"="*W}\n{" "*40} % DIFF   DISPATCH1       DISPATCH2\n{"="*W}')
  for n, d, t in l:
    y, x = d[0][2], d[1][2]
    print(f'{n:<40} {t:<6.2f}   {x:<15,d} {y:,d}')

# todo(); sys.exit(0)
if len(sys.argv) == 3:
  workload, dispatch = int(sys.argv[1]), int(sys.argv[2])
  speed_of_light(workload, dispatch=dispatch)
elif len(sys.argv) == 5:
  w1, d1 = int(sys.argv[1]), int(sys.argv[2])
  w2, d2 = int(sys.argv[3]), int(sys.argv[4])
  metric_diff((w1,d1), (w2,d2))
