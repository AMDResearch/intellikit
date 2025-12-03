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

#%%

import sys, pickle, csv
from pprint import pprint
import faiss
import joblib
from sklearn.linear_model import Lasso, Ridge
from src.profile.db import sql
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from src.metric import *
import xgboost as xgb
import pymc as pm
import arviz as az
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import cond
from datetime import datetime
from path import get_data_path
import statsmodels.api as sm
import statsmodels.formula.api as smf
import umap.umap_ as umap
import plotly.express as px

COLOR_MAP = {
  'cyan': 6,
  'red': 9,
  'yellow': 11,
  'green': 10,
  'blue': 12
}

def color(color, text):
  return f"\033[38;5;{COLOR_MAP[color]}m{text}\033[0m" \
    if color in COLOR_MAP else text


DATA_FILE = get_data_path() + '/data_mi200.csv'

def p(msg):
    now = datetime.now()
    print(f'{now}: {msg}')

def candidate_dispatches(gpu_model):
  ret = []
  ws = sql(f'''select workload_id from workload
           where gpu_model='{gpu_model}'
           order by workload_id
          ''')

  # to pick the best performance for a kernel, the last dispatch will be used
  # not sure why the SQL doesnot work in a single query
  for w in ws:
    ds = sql(f'''
      SELECT DISTINCT ON (kernel)
        workload,
        last_value(dispatch) OVER (
            PARTITION BY kernel
            ORDER BY workload, dispatch
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
          ) AS dispatch,
        kernel
      FROM dispatch
      WHERE workload={w[0]}
      ORDER by workload, dispatch
    ''')

    for w, d, k in ds:
      ret += (w, d, k),

  return ret

def cycle_data(workload, dispatch):
  try:
    _, cu_per_gpu, _, _, _, _, num_xcd = specs(workload)
    k, ns, grid, wg, lds_per_workitem, _, arch_vgpr, accum_vgpr, sgpr = dur(workload, dispatch)
    wav, _, _, cycles, *_  = wave(workload, dispatch, cu_per_gpu, num_xcd)
  except Exception as e:
    print(e)
    return None

  valu, vmem, lds, mfma, salu, smem, bran, _ = instruction_mix(workload, dispatch, wav)

  return (
          wav,
          lds_per_workitem,
          arch_vgpr+accum_vgpr,
          valu,
          vmem,
          lds,
          mfma,
          salu,
          smem,
          bran,
          cycles,
          k,
          f'[{workload},{dispatch}]')

COLUMNS = [
  'WAVE',
  'LDS_ALLOC',
  'VGPR_AGPR_ALLOC',
  'VALU',
  'VMEM',
  'LDS',
  'MFMA',
  'SALU',
  'SMEM',
  'BRANCH',
  'CYCLES',
  'KERNEL',
  'DISPATCH'
]

def read_csv(file):
  ret = []
  with open(file, 'r') as file:
    reader = csv.reader(file)
    _ = next(reader)
    for row in reader: ret.append(row)

  return ret

PREP_DATA_FILE = get_data_path() + '/data_mi200_prep.csv'
def cycle_data_for(gpu_model):
  ds = candidate_dispatches(gpu_model)

  result = []
  for w, d, _ in ds:
    x = cycle_data(w, d)

    if x: result += x,

  with open(DATA_FILE, 'w') as OUT:
    writer = csv.writer(OUT)
    writer.writerow(COLUMNS)
    for row in result:
        writer.writerow(row)

  f2c = {}

  for r in result:
    key, c, k = tuple(int(x) for x in r[:-3]), float(r[-3]), r[-2]
    if key not in f2c:
      f2c[key] = {k:[c]}
    else:
      k2c = f2c[key]
      if k in k2c:
        k2c[k] += c,
      else:
        k2c[k] = [c]
  # pprint(f2c)

  ret = []
  for f in f2c:
    ks = [f2c[f][k] for k in f2c[f]]
    # here use the minimum of kernel averages
    c = min(map(np.mean, ks))
    ret += f + (c,),

  with open(PREP_DATA_FILE, 'w') as OUT:
    writer = csv.writer(OUT)
    writer.writerow(COLUMNS[:-2])
    for row in ret:
        writer.writerow(row)

# TODO
def tree_size():

  X_train, X_test, y_train, y_test = get_training_data()

  dtrain = xgb.DMatrix(X_train, label=y_train)
  dval = xgb.DMatrix(X_test, label=y_test)

  params = {
      'objective': 'reg:logistic',
      'eval_metric': 'logloss',
      'eta': 0.1  # Learning rate
  }

  evals = [(dtrain, 'train'), (dval, 'eval')]
  bst = xgb.train(params, dtrain, num_boost_round=1000, evals=evals, early_stopping_rounds=10)
  return bst

SCALER_FILE = get_data_path() + '/scaler.pickle'
SCALER_FILE_2 = get_data_path() + '/scaler_2.pickle'
def get_training_data():

  data = pd.read_csv(PREP_DATA_FILE)

  X = data.drop(['CYCLES'], axis=1)
  y = data['CYCLES']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)
  scaler = StandardScaler()
  scaler.fit(X_train)

  X_train_scaled = scaler.transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  joblib.dump(scaler, SCALER_FILE)

  return X_train_scaled, X_test_scaled, y_train.to_numpy(), y_test.to_numpy()

CYCLE_MODEL_FILE = get_data_path()+'/cycle_model.json'
def train():

  X_train, X_test, y_train, y_test = get_training_data()

  def custom_asymmetric_mse(y_true, y_pred):
      residual = (y_pred - y_true).astype("float")
      grad = 2 * residual
      # Apply a higher penalty if the prediction is lower than the actual
      grad[residual < 0] *= 1.5
      hess = 2.0 * np.ones_like(y_true)
      return grad, hess

  model = XGBRegressor(
      # objective='reg:squarederror',
      objective=custom_asymmetric_mse,
      max_depth=8,
      n_estimators=300,
      learning_rate=0.1,
      random_state=43
  )

  p('training ...')
  model.fit(X_train, y_train)
  model.save_model(CYCLE_MODEL_FILE)
  # loaded_model = XGBRegressor()
  # loaded_model.load_model(CYCLE_MODEL_FILE)

  train_predictions = model.predict(X_train)
  test_predictions = model.predict(X_test)

  p('evaluating ...')
  train_mse = mean_squared_error(y_train, train_predictions)
  train_r2 = r2_score(y_train, train_predictions)
  test_mse = mean_squared_error(y_test, test_predictions)
  test_r2 = r2_score(y_test, test_predictions)

  print(f"Training MSE: {train_mse:.4f}")
  print(f"Training R-squared: {train_r2:.4f}")
  print(f"Test MSE: {test_mse:.4f}")
  print(f"Test R-squared: {test_r2:.4f}")

  importances = model.feature_importances_
  feature_importances = pd.DataFrame({'feature': COLUMNS[:-3], 'importance': importances})
  feature_importances = feature_importances.sort_values('importance', ascending=False)

  p("Feature Importances:")
  print(feature_importances)
  # xgb.plot_tree(model, num_trees=0)
  # plt.rcParams['figure.figsize'] = [20, 10]
  # plt.show()

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            # nn.Linear(len(COLUMNS)-3, 64),
            nn.Linear(len(COLUMNS)-3, 16),
            nn.ReLU(),
            # nn.Linear(64, 32),
            nn.Linear(16, 8),
            nn.ReLU(),
            # nn.Linear(32, 1),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.layers(x)

def train2():
  model = MLP()
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  X_train, X_test, y_train, y_test = [torch.FloatTensor(i) for i in get_training_data()]

  num_epochs = 10
  for epoch in range(num_epochs):
      model.train()
      optimizer.zero_grad()
      predictions = model(X_train)
      loss = criterion(predictions, y_train)
      loss.backward()
      optimizer.step()

      if (epoch + 1) % 10 == 0:
          print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

  # Evaluation
  model.eval()
  with torch.no_grad():
      test_predictions = model(X_test)
      test_loss = criterion(test_predictions, y_test)
      print(f"Test Loss: {test_loss.item():.4f}")

  torch.save(model.state_dict(), 'cycle.pth')

def slim_linear():
  data = pd.read_csv(PREP_DATA_FILE)

  X = data.drop(['WAVE', 'LDS_ALLOC', 'VGPR_AGPR_ALLOC'], axis=1)
  X['OP'] = X['VALU'] + X['MFMA'] + X['SALU'] + X['BRANCH']
  X['MEM'] = X['VMEM'] + X['SMEM']
  X = X.drop(['VALU', 'MFMA', 'SALU', 'SMEM', 'BRANCH', 'VMEM'], axis=1)
  X['CYCLES'] = X['CYCLES'].map(int)
  # print(X.to_string())
  X = X.drop(['CYCLES'], axis=1)
  X = sm.add_constant(X)
  '''
  X['PER_VMEM'] = np.where(X['VMEM'] != 0,
                          X['CYCLES'] / X['VMEM'],
                          0.0)
  X['PER_VMEM'] = X['PER_VMEM'].map(int)
  '''
  y = data['CYCLES'].map(int)

  # model = sm.GLM(np.log(y), X).fit()
  # model = sm.GLM(np.log(y), X,  family=sm.families.Poisson()).fit()
  model = sm.GLM(y, X,  family=sm.families.Poisson()).fit()
  # model = sm.GLM(np.log(y), X,  family=sm.families.NegativeBinomial()).fit()
  # print("Parameter names:", model.params.index.tolist())
  print(model.summary())
  model.save('cycle_glm.pickle')

  la = Lasso(alpha=0.1)
  la.fit(X, y)
  print(la)
  print(f"Model coefficients: {la.coef_}")
  joblib.dump(la, 'cycle_lasso.pickle')

  ri = Ridge(alpha=0.1)
  ri = ri.fit(X, y)
  print(ri)
  print(f"Model coefficients: {ri.coef_}")
  joblib.dump(ri, 'cycle_ridge.pickle')

def bay():
  data = pd.read_csv(PREP_DATA_FILE)

  X = data.drop(['WAVE', 'LDS_ALLOC', 'VGPR_AGPR_ALLOC', 'LDS'], axis=1)
  X['OP'] = X['VALU'] + X['MFMA'] + X['SALU'] + X['BRANCH'] + X['SMEM'] + X['VMEM']
  X['MEM'] = X['VMEM'] + X['SMEM']
  X = X[X['MEM'] > 0]
  X['CYCLES'] = X['CYCLES'].map(int)
  X = X.drop(['VALU', 'MFMA', 'SALU', 'SMEM', 'BRANCH', 'VMEM'], axis=1)
  X['PER_MEM'] = (X['CYCLES'] - 4 * X['OP'])/X['MEM']
  # print(X.to_string())
  X['PER_MEM'] = X['PER_MEM'].map(int)
  X = X[X['PER_MEM'] <= 2000]
  y = X['CYCLES']
  # print(X['PER_MEM'].mean())
  # print(X['PER_MEM'].std())
  num = X.shape[0]

  with pm.Model() as model:
    ops = pm.Data('ops', X['OP'])
    wait = pm.Normal('wait', mu=600, sigma=600)
    mem = pm.Normal('mem', mu=530, sigma=513)
    total1 = pm.Deterministic('total1', wait + mem)
    total2 = pm.Deterministic('total2', wait + 4 * ops)
    total = pm.Deterministic('total', pm.math.maximum(total1, total2))

    sigma = pm.HalfNormal('sigma', sigma=200)
    obs = pm.Normal('obs', mu=total, sigma=sigma, observed=y)

    # Inference: Use MCMC sampling
    trace = pm.sample(1000, return_inferencedata=True)

  az.plot_trace(trace);
  # az.to_netcdf(trace, "model_trace.nc")

def show_trace():
  trace = az.from_netcdf("/work1/amd/jchen/ai/guided-tuning/model_trace.nc")
  az.plot_trace(trace);
  plt.show()

def train3():
  X_train, X_test, y_train, y_test = get_training_data()
  print(f'Condition number of X_train: {cond(X_train):.2f}')

  model = sm.GLM(np.log(y_train), X_train).fit()
  print(model.summary())
  model.save('cycle_glm.pickle')

  model = Lasso(alpha=0.1)
  model.fit(X_train, np.log(y_train))
  print(f"Model coefficients: {model.coef_}")
  joblib.dump(model, 'cycle_lasso.pickle')

  model = Ridge(alpha=0.1)
  model.fit(X_train, np.log(y_train))
  print(f"Model coefficients: {model.coef_}")
  joblib.dump(model, 'cycle_ridge.pickle')

def svm():
  X_train, X_test, y_train, y_test = get_training_data()
  model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  mse = mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)

  print(f"Mean Squared Error: {mse:.3f}")
  print(f"R^2 Score: {r2:.3f}")

def do_pca():
  X_train, X_test, y_train, y_test = get_training_data()
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X_train)
  pca = PCA(n_components=None)
  pca.fit(X_scaled)
  X_pca = pca.transform(X_scaled)
  print("Explained variance ratio:", pca.explained_variance_ratio_)
  cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
  print("Cumulative explained variance:", cumulative_variance)
  plt.figure(figsize=(8, 5))
  plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
  plt.xlabel('Number of Components')
  plt.ylabel('Cumulative Explained Variance')
  plt.title('Scree Plot')
  plt.grid()
  plt.show()
  plt.figure(figsize=(8, 5))
  plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=50)
  plt.xlabel('Principal Component 1')
  plt.ylabel('Principal Component 2')
  plt.title('2D PCA Plot')
  plt.colorbar(label='Target')
  plt.grid()
  plt.show()

  loadings = pca.components_.T
  features = COLUMNS[:-3]

  loadings_df = pd.DataFrame(
    loadings,
    columns=[f'PC{i+1}' for i in range(loadings.shape[1])],
    index=features)
  print(loadings_df)

FAISS_FILE = get_data_path() + '/faiss.bin'
F2K_FILE = get_data_path() + '/f2k.pickle'
def build_vector_store():
  data = pd.read_csv(DATA_FILE)
  f2k = {}
  for i, row in data.iterrows():
    f2k[i] = (row[-3], row[-1], row[-2])

  with open(F2K_FILE, 'wb') as OUT:
    pickle.dump(f2k, OUT)

  X = data.drop(['CYCLES', 'KERNEL', 'DISPATCH'], axis=1)
  index = faiss.IndexFlatL2(len(COLUMNS)-3)
  index.add(X)
  faiss.write_index(index, FAISS_FILE)

def visualize_vector_store():
  index = faiss.read_index(FAISS_FILE)
  dim = len(COLUMNS) - 3
  vectors = np.zeros((index.ntotal, dim), dtype='float32')
  for i in range(index.ntotal):
      vectors[i] = index.reconstruct(i)

  num_clusters = 10
  kmeans = faiss.Kmeans(dim, num_clusters, niter=20, verbose=True)

  kmeans.train(vectors)

  D, cluster_labels = kmeans.index.search(vectors, 1)

  print("Cluster centers shape:", kmeans.centroids.shape)
  print("Cluster labels shape:", cluster_labels.shape)
  print("Cluster labels (first 10):", cluster_labels[:10].flatten())

  return
  reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3)
  embeddings_3d = reducer.fit_transform(vectors)
  fig = px.scatter_3d(
    x=embeddings_3d[:, 0],
    y=embeddings_3d[:, 1],
    z=embeddings_3d[:, 2],
    color=np.linalg.norm(vectors, axis=1),  # Optional: color by vector norm
    title="3D Visualization of FAISS Vector Store using UMAP"
  )

  fig.update_layout(scene=dict(xaxis_title="UMAP Dimension 1",
                              yaxis_title="UMAP Dimension 2",
                              zaxis_title="UMAP Dimension 3"),
                    margin=dict(l=0, r=0, b=0, t=40))

  fig.show()


F2K = {}
def nearest(x, k=3):
  global F2K
  if not F2K:
    with open(F2K_FILE, 'rb') as IN: F2K = pickle.load(IN)

  index = faiss.read_index(FAISS_FILE)
  D, ns = index.search(np.array([x]), k)

  return zip(ns[0], [F2K[i] for i in ns[0]], [np.sqrt(d) for d in D[0]])

def study_distance():
  scaler = joblib.load(SCALER_FILE)
  data = pd.read_csv(DATA_FILE)
  data = data.drop(['CYCLES', 'KERNEL', 'DISPATCH'], axis=1)
  rows, _ = data.shape
  for i in range(rows-1):
    x = data.iloc[i].to_numpy()
    ns = nearest(x)
    print(f'>>> {i}')
    for (j, (cs, p, k), d) in ns:
      y = data.iloc[j].to_numpy()
      x1, y1 = scaler.transform([x]), scaler.transform([y])
      print(f'''{str(j).ljust(6)} {d:.2f}: {cs:.2f} {p} {k[:80]}
{np.linalg.norm(x1[0]-y1[0]):.2f} {y}''')

def test_search():
  data = pd.read_csv(DATA_FILE)
  data = data.drop(['CYCLES', 'KERNEL', 'DISPATCH'], axis=1)
  rows, _ = data.shape
  try:
    for _ in range(100):
      pick = random.randint(0, rows-1)
      # pick = 0
      ns = nearest(data.iloc[pick].to_numpy())
      print(f'>>> {pick}')
      for (i, (cs, p, k), d) in ns:
        print(f'{i} {d:.2f}: {cs:.2f} {p} {k[:80]}\n{data.iloc[i].to_numpy()}')
      _ = input('Continue? (CTRL-C to quit)')
  except KeyboardInterrupt:
    pass

def distance(w1, d1, w2, d2):
  scaler = joblib.load(SCALER_FILE)
  # one point then find its neighbors
  if w1 == w2 and d2 == d2:
    x = cycle_data(w1, d1)
    ns = nearest(np.array(x[:-3]), 5)
    print('Neighbors:')
    for (i, (cs, p, k), d) in ns:
      print(f'{i} {d:.2f}: {cs:.2f} {p} {k[:70]}\n')
  else:
    x1, x2 = cycle_data(w1, d1), cycle_data(w2, d2)
    d1, d2 = x1[-1] + ' ' + x1[-2], x2[-1] + ' ' + x2[-2]
    f1, f2 = np.array(x1[:-3]), np.array(x2[:-3])
    g1, g2 = scaler.transform([f1]), scaler.transform([f2])

    print(f'''
{d1[:80]}
{d2[:80]}
{np.linalg.norm(f1 - f2):.2f}
{np.linalg.norm(g1[0] - g2[0]):.2f}
''')

def tt():
  wds = [
    (153, 0),
    (155, 0),
    (156, 0),
    (157, 0),
    (158, 0),
    (159, 0),
    (160, 0),
    (161, 0),
    (162, 0),
    (163, 0),
    (164, 0),
    (165, 0),
    (165, 1),
    (166, 2),
    (167, 0),
    (168, 17),
    (169, 0),
    (170, 54),
    (170, 56),
    (171, 0),
    (172, 0),
    (173, 0),
    (173, 1),
    (174, 0),
    (174, 1),
    (175, 2),
    (176, 2),
    (177, 2),
    (178, 2),
  ]

  data = pd.read_csv(DATA_FILE)
  data = data.drop(['CYCLES', 'KERNEL', 'DISPATCH'], axis=1)
  scaler = joblib.load(SCALER_FILE)
  m1 = XGBRegressor()
  m1.load_model(CYCLE_MODEL_FILE)
  for workload, dispatch in wds:
    x = cycle_data(workload, dispatch)
    if x == None:
      print('Failed to obtain data.')
      return

    X = scaler.transform([x[:-3]])
    target = f'{x[-3]:.2f}'
    pred = f'{m1.predict(X)[0]:.2f}'
    print(f'''\n>>> {x[:-3]}
  {x[-1]} {x[-2][:80]}
  REAL:    {color('green', target)}
  XGB:     {color('blue', pred)}
  ''')

    x = np.array(x[:-3])
    ns = nearest(x)
    for (j, (cs, p, k), d) in ns:
      y = data.iloc[j].to_numpy()
      x1, y1 = scaler.transform([x]), scaler.transform([y])
      val = f'{cs:.2f}'

      print(f'''{str(j).ljust(6)} {d:.2f}: {color('blue', val)} {p} {k[:70]}
  {np.linalg.norm(x1[0]-y1[0]):.2f} {y}
  ''')

def predict(workload, dispatch):
  data = pd.read_csv(DATA_FILE)
  data = data.drop(['CYCLES', 'KERNEL', 'DISPATCH'], axis=1)
  scaler = joblib.load(SCALER_FILE)
  m1 = XGBRegressor()
  m1.load_model(CYCLE_MODEL_FILE)
  x = cycle_data(workload, dispatch)
  if x == None:
    print('Failed to obtain data.')
    return

  X = scaler.transform([x[:-3]])
  target = f'{x[-3]:.2f}'
  pred = f'{m1.predict(X)[0]:.2f}'
  print(f'''
{color('cyan', 'Feature')}: {x[:-3]}
{color('cyan', 'Kernel')}:  {x[-1]} {x[-2][:80]}
{color('cyan', 'REAL')}:    {color('green', target)}
------------------------------
{color('cyan', 'XGB')}:     {color('blue', pred)}

{color('cyan', 'Neighbors')}:''')

  x = np.array(x[:-3])
  ns = nearest(x, 5)
  for (j, (cs, p, k), d) in ns:
    y = data.iloc[j].to_numpy()
    x1, y1 = scaler.transform([x]), scaler.transform([y])
    val = f'{cs:.2f}'
    di = f'{d:.2f}'

    print(f'''  {str(j).ljust(6)} {color('yellow', di)}: {color('blue', val)} {p} {k[:60]}
  {np.linalg.norm(x1[0]-y1[0]):.2f} {y}
''')

def shootout():
  ds = [(97, 1), (79, 61), (78, 61), (54, 0), (54, 100),
        (54, 200), (54, 300), (54, 400), (54, 500), (54, 600), (145, 2)]

  scaler = joblib.load(SCALER_FILE)
  scaler2 = joblib.load(SCALER_FILE_2)
  m1 = XGBRegressor()
  m1.load_model(CYCLE_MODEL_FILE)

  m2 = MLP()
  m2.load_state_dict(torch.load('cycle.pth'))
  m2.eval()
  m3 = sm.load('cycle_glm.pickle')
  m4 = joblib.load("cycle_lasso.pickle")
  m5 = joblib.load("cycle_ridge.pickle")
  for (workload, dispatch) in ds:
    x = cycle_data(workload, dispatch)
    if x == None:
      print('Failed to obtain data.')
      return

    print(f'''
>>>
{x[:-3]}
{x[-1]} {x[-2][:80]}
REAL:    {x[-3]:.2f}
---------------------------------''')

    X = scaler.transform([x[:-3]])
    X2 = scaler2.transform([
      (x[4], x[5], x[3]+x[6]+x[7]+x[8]+x[9])
    ])

    print(f'XGB:\t {m1.predict(X)[0]:.2f}')
    print(f'MLP:\t {m2(torch.FloatTensor(X)).item():.2f}')
    print(f'GLM:\t {np.exp(m3.predict(X2)[0]):.2f}')
    print(f'Lasso:\t {np.exp(m4.predict(X2)[0]):.2f}')
    print(f'Ridge:\t {np.exp(m5.predict(X2)[0]):.2f}\n\n')



#%%
# do_pca()
# cycle_data_for('MI200')
# train()
# train2()
# train3()
# build_vector_store()
# test_search()
# visualize_vector_store()
# study_distance()
# slim_linear()
# tt()
# svm()
import os
bay()
# show_trace()
sys.exit(0)

if len(sys.argv) == 3:
  predict(sys.argv[1], sys.argv[2])
elif len(sys.argv) == 5:
  distance(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
else:
  shootout()

# %%
