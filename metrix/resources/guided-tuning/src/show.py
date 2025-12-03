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

import sys, io, os, json
import pandas as pd
from contextlib import redirect_stdout
from src.db import sql, init, get_db_file, set_db_file, remove
from src.metric import speed_of_light, NotFoundError
from tabulate import tabulate
from src.logger import console_error, console_log, console_warning
from src.csv_convert import export_db
import re

def to_dataframe(sql_cmd:str, columns:list)->pd.DataFrame:
  try:
    out = sql(sql_cmd)
    if not out:
      console_error(f'No data found for the query: \"{sql_cmd}\"')
  except IndexError:
    console_error(f'Workload with ID ({id}) not found!')
  return pd.DataFrame(out, columns=columns)

def pretty_print_table(df:pd.DataFrame, stralign:str="right", show_index=False):
  table = tabulate(
    df,
    headers='keys',
    tablefmt="fancy_grid",
    numalign="right",
    stralign=stralign,
    showindex=show_index,
    maxcolwidths=150
  )
  print(table)


def show_match_workloads(save: str, work=''):
  df = to_dataframe(
    f'select workload_id, workload_name from workload where workload_name like \'%{work}%\'',
    ['id', 'workload_name']
  )
  if save:
    export_db(df, save)
  pretty_print_table(df)

def get_workload_name(id:int)->str:
  try:
    out = sql(f'select workload_name from workload where workload_id={id}')
  except IndexError:
    console_error(f'Workload with ID ({id}) not found!')
  return out[0][0]

W = 98
def show_workload(id:int, save:str=None, max_kernel_len:int=None) -> None:
  # Print workload details
  df = to_dataframe(
    f'select timestamp, command, gpu_model from workload where workload_id={id}',
    ['Timestamp', 'Command', 'GPU Model']
  )
  pretty_print_table(df, stralign='center')
  print(f"{'-'*W}\n")

  # Print dispatch ranges
  df = to_dataframe(
    f'select dispatch, kernel from dispatch where workload={id}',
    ['Dispatch', 'Kernel']
  )
  # Group consecutive dispatches for each kernel
  ranges = []
  current_kernel = None
  start_dispatch = None

  for i, row in df.iterrows():
    if row['Kernel'] != current_kernel:
      if current_kernel is not None:
        ranges.append({
          'Dispatch': f"{start_dispatch}-{prev_dispatch}" if start_dispatch != prev_dispatch else f"{start_dispatch}",
          'Kernel': current_kernel[:W]
        })
      current_kernel = row['Kernel']
      start_dispatch = row['Dispatch']
    prev_dispatch = row['Dispatch']

  # Add the last range
  if current_kernel is not None:
    ranges.append({
      'Dispatch': f"{start_dispatch}-{prev_dispatch}" if start_dispatch != prev_dispatch else f"{start_dispatch}",
      'Kernel': current_kernel[:W]
    })

  range_df = pd.DataFrame(ranges)
  if max_kernel_len:
    range_df['Kernel'] = range_df['Kernel'].str.slice(0, max_kernel_len)
  pretty_print_table(range_df)

  # Print top kernels
  df = to_dataframe(
    f'select count, duration_mean, pct, kernel from top where workload={id}',
    ['Count', 'Avg-Duration', 'Percentage', 'Kernel']
  )
  df_kernel_dispatches = to_dataframe(
    f'''select dispatch, kernel from dispatch where workload={id}''',
    ['Dispatch','Kernel']
  )
  df_conflicts = to_dataframe(
    f'''select dispatch, value from metric where name like 'SQ_LDS_BANK_CONFLICT' and workload={id}''',
    ['Dispatch','SQ_LDS_BANK_CONFLICT']
  )
  df_merged = pd.merge(df_kernel_dispatches, df_conflicts, on='Dispatch', how='left').drop(["Dispatch"], axis=1).groupby('Kernel', as_index=False).sum()
  df_conflicts = df_merged[['Kernel', 'SQ_LDS_BANK_CONFLICT']]
  df_top = pd.merge(df, df_conflicts, on='Kernel', how='left')
  df_top = df_top[['Count', 'Avg-Duration', 'Percentage', 'SQ_LDS_BANK_CONFLICT', 'Kernel']]
  df_top.rename(columns={'SQ_LDS_BANK_CONFLICT': 'LDS Bank Conflicts'}, inplace=True)
  if max_kernel_len:
    df_top['Kernel'] = df_top['Kernel'].str.slice(0, max_kernel_len)
  pretty_print_table(df_top)
  if save:
    export_db(df_top, save)

def show_speed(workload, dispatch=None, kernel:str=None):
  try:
    return speed_of_light(workload, d=dispatch, k=kernel)
  except NotFoundError as e:
    console_error(f'{e}')

def compare(workload1, dispatch1, workload2, dispatch2):
  def speed_(workload, dispatch):
    output = io.StringIO()

    with redirect_stdout(output):
      speed_of_light(workload, d=dispatch, display=True, wrap_table=False)

    return output.getvalue().split('\n')

  x = speed_(workload1, dispatch1)
  y = speed_(workload2, dispatch2)
  max_len = max(len(l) for l in x)
  for i, line in enumerate(x):
     console_log(f'{line.rstrip().ljust(max_len)} {y[i].rstrip()}')

def validate_args(args):
  if args.kernel:
    try:
      re.compile(r"" + args.kernel.strip())
    except re.error:
      console_error('Invalid regex syntax for kernel argument!')
  if args.save:
    if args.dispatch or args.kernel:
      if not args.save.endswith(".json"):
        console_error('Output file must end in .json extension')
    else:
      if not args.save.endswith(".csv"):
        console_error('Output file must end in .csv extension')
  try:
    if (args.workload and args.dispatch) and (len(args.workload) != len(args.dispatch)):
      console_error('Workload and Dispatch IDs must be equal in length for comparison!')
    if len(args.workload) > 2:
      console_error('Too many workload IDs provided! Can only compare two workloads.')
    if len(args.dispatch) > 2:
      console_error('Too many dispatch IDs provided! Can only compare two dispatches.')
  except TypeError:
    pass

def show(args, parser):
  validate_args(args)

  if args.db:
    set_db_file(args.db)
  if not os.path.exists(get_db_file()):
    console_log("Initializing database...")
    init()

  if args.remove:
    console_log(f"Removing workload {args.remove}...")
    remove(args.remove)

  results = []
  if not args.workload and not args.dispatch and not args.workload_name and not args.kernel:
    # Show all workloads
    show_match_workloads(args.save)
  elif args.workload_name:
    # Show matching workloads
    show_match_workloads(args.save, args.workload_name)
  elif args.workload and not args.dispatch and not args.kernel:
    # Show workload details
    show_workload(args.workload[0], args.save, args.less)
  elif (len(args.workload) == 1 and (args.kernel or len(args.dispatch) == 1)):
    # Show dispatch/kernel details
    if args.kernel:
      # Sanitize the pattern for safe use in the SQL query. Avoid escape characters.
      sanitized_k_regex = (
        args.kernel.strip()
        .replace("'", "''")
        .replace("<", r"\<")
        .replace(">", r"\>")
        .replace("*", r"\*")
        .replace("(", r"\(")
        .replace(")", r"\)")
      )
      # Add .* to the beginning and end of each kernel in the regex
      sanitized_k_regex = "|".join(
        [f".*{part.strip()}.*" for part in re.split(r'\|', sanitized_k_regex)]
      )
      if args.separate:
        regex_list = [part.strip() for part in re.split(r'\|', sanitized_k_regex)]
        for r in regex_list:
          results.append(show_speed(args.workload[0], kernel=r))
      else:
        results.append(show_speed(args.workload[0], kernel=sanitized_k_regex))
    else:
      results.append(show_speed(args.workload[0], dispatch=args.dispatch[0]))

    # Save results to file
    if args.save:
      with open(args.save, 'w') as f:
        json.dump(results, f, indent=4)
  elif len(args.workload) == 2 and len(args.dispatch) == 2:
    # Dispatch compare
    compare(args.workload[0], args.dispatch[0], args.workload[1], args.dispatch[1])
  else:
    console_warning('Unrecognized combination of arguments. Please check the usage below:')
    parser.print_help()

# if __name__ == "__main__":
#   import argparse

#   parser = argparse.ArgumentParser(description='Show workload details and metrics.')
#   parser.add_argument('--workload', nargs='+', type=int, help='Workload IDs to show')
#   parser.add_argument('--dispatch', nargs='+', type=int, help='Dispatch IDs to show')
#   parser.add_argument('--workload-name', type=str, help='Workload name to search for')
#   parser.add_argument('--kernel', type=str, help='Kernel regex pattern to search for')
#   parser.add_argument('--save', type=str, help='File to save results (CSV or JSON)')
#   parser.add_argument('--less', action='store_true', help='Limit kernel length in output')
#   parser.add_argument('--separate', action='store_true', help='Separate regex patterns with |')
#   parser.add_argument('--remove', type=int, help='Remove workload by ID')
#   parser.add_argument('--db', type=str, help='Path to the database file')

#   args = parser.parse_args()
#   show(args, parser)
