import numpy as np
import cPickle
import heapq
import os
from IPython.core.debugger import Tracer
from scipy.io import loadmat
import time
import h5py
import json
import copy
import bz2
import code
import traceback as tb

def unique_rows(a):
  b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
  _, idx = np.unique(b, return_index=True)
  return a[idx], idx;

def setdiff2d(a1, a2):
  assert a1.dtype == a2.dtype;
  #only works with numpy >= 1.7
  versplit = [int(x) for x in np.__version__.split('.')];
  assert versplit[0]>=1 and versplit[1]>=7;
  a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
  a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
  return np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1])

def argtopk(a, k):
  ind = np.argpartition(a,-k)[-k:]
  srtind = ind[np.argsort(a[ind])[::-1]];
  return srtind;


def get_dir_list(dirPath, extension = None):
    onlydirs = [ os.path.join(dirPath,f) for f in os.listdir(dirPath) if os.path.isdir(os.path.join(dirPath,f)) ];
    if extension!= None:
        onlydirs = [f for f in onlydirs if os.path.splitext(f)[1]==extension];
    onlydirs.sort();
    return onlydirs;

#extension with "." e.g. .jpg
def get_file_list(dirPath, extension = None):
    onlyfiles = [ os.path.join(dirPath,f) for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath,f)) ];
    if extension!= None:
        onlyfiles = [f for f in onlyfiles if os.path.splitext(f)[1]==extension];
    onlyfiles.sort();
    return onlyfiles;

def get_file_list_prefix(dirPath, prefix, extension=None):
    onlyfiles = [ os.path.join(dirPath,f) for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath,f)) and f.startswith(prefix) ];
    if extension!= None:
        onlyfiles = [f for f in onlyfiles if os.path.splitext(f)[1]==extension];
    onlyfiles.sort();
    return onlyfiles;

def list_to_indexed_dict(lvar):
    dvar = {};
    for ind, item in enumerate(lvar):
      dvar[item]=ind;
    return dvar;

def tic_toc_print(interval, string):
  global tic_toc_print_time_old
  if 'tic_toc_print_time_old' not in globals():
    tic_toc_print_time_old = time.time()
    print string
  else:
    new_time = time.time()
    if new_time - tic_toc_print_time_old > interval:
      tic_toc_print_time_old = new_time;
      print string
def mkdir(output_dir):
    return mkdir_if_missing(output_dir);

def mkdir_if_missing(output_dir):
  """
  def mkdir_if_missing(output_dir)
  """
  if not os.path.exists(output_dir):
    try:
      os.makedirs(output_dir)
      return True;
    except: #generally happens when many processes try to make this dir
      return False;


def recurse_get_mat_struct(v, curr_field=None):
  accum_dict = {};
  if type(v).__name__ != 'mat_struct':
    if type(v).__name__ == 'ndarray':
      #sometimes we have nested mat_structs ...
      numel = v.size;
      found_nested_structs = False;
      for x in range(numel):
        if type(v.item(x)).__name__ == 'mat_struct':
          if found_nested_structs == False:
            accum_dict[curr_field]=[];
          found_nested_structs = True;
        if found_nested_structs:
          newdict = recurse_get_mat_struct(v.item(x), curr_field);
          accum_dict[curr_field].append(newdict);
      if found_nested_structs == False:
        accum_dict[curr_field] = v;
    else:
        accum_dict[curr_field] = v;
  else:
    for field in v._fieldnames:
      local_dict = recurse_get_mat_struct( getattr(v, field), field );
      if field not in local_dict:
        accum_dict[field] = copy.deepcopy(local_dict);
      else:
        accum_dict[field] = copy.deepcopy(local_dict[field]);
    if curr_field not in accum_dict:
      ret_dict = {};
      ret_dict[curr_field] = copy.deepcopy(accum_dict);
      accum_dict = ret_dict;
  return accum_dict;

def mat_to_dict(mat_name):
  matfile = loadmat(mat_name, squeeze_me=True, struct_as_record=False);
  var_keys = matfile.keys();
  allVarDict = {};
  for v in var_keys:
    if v.startswith('__') == True:
      continue;
    dictData = {};
    for field in matfile[v]._fieldnames:
      localDict = recurse_get_mat_struct( getattr(matfile[v], field), field );
      if field not in localDict:
        dictData[field] = copy.deepcopy(localDict);
      else:
        dictData[field] = copy.deepcopy(localDict[field]);
    allVarDict[v] = dictData;
  return allVarDict;

def save_variables_h5(h5_file_name, var, info, overwrite = False):
  if info is None:
    return save_variables_h5_dict(h5_file_name, var, overwrite)
  if os.path.exists(h5_file_name) and overwrite == False:
    raise Exception('{:s} exists and over write is false.'.format(h5_file_name))
  # Construct the dictionary
  assert(type(var) == list); assert(type(info) == list);
  with h5py.File(h5_file_name, 'w') as f:
    for i in range(len(info)):
      d = f.create_dataset(info[i], data=var[i], chunks=True, compression="gzip", compression_opts=9);

def rec_get_keys(fh, src, keyList):
    if src!='' and type(fh[src]).__name__ == 'Dataset':
      keyList.append(src);
      return keyList;
    if src!='':
      moreSrcs = fh[src].keys();
    else:
      moreSrcs = fh.keys();
    for kk in moreSrcs:
      if src=='':
          keyList = rec_get_keys(fh, kk, keyList);
      else:
          keyList = rec_get_keys(fh, src+'/'+kk, keyList);
    return keyList;

def get_h5_keys(h5_file_name):
  if os.path.exists(h5_file_name):
    with h5py.File(h5_file_name,'r') as f:
      keyList = rec_get_keys(f, '', []);
    return keyList;
  else:
    raise Exception('{:s} does not exists.'.format(h5_file_name))


def save_variables_h5_dict(h5_file_name, dictVar, overwrite = False):
  if os.path.exists(h5_file_name) and overwrite == False:
    raise Exception('{:s} exists and over write is false.'.format(h5_file_name))
  # Construct the dictionary
  assert(type(dictVar) == dict);
  with h5py.File(h5_file_name, 'w') as f:
    for key in dictVar:
      d = f.create_dataset(key, data=dictVar[key], chunks=True, compression="gzip", compression_opts=9);

def load_variablesh5(h5_file_name):
  if os.path.exists(h5_file_name):
    with h5py.File(h5_file_name,'r') as f:
      d = {};
      h5keys = get_h5_keys(h5_file_name);
      for key in h5keys:
        d[key] = f[key].value;
    return d
  else:
    raise Exception('{:s} does not exists.'.format(h5_file_name))

def save_variables(pickle_file_name, var, info, overwrite = False):
  """
    def save_variables(pickle_file_name, var, info, overwrite = False)
  """
  fext = os.path.splitext(pickle_file_name)[1]
  if fext =='.h5':
    return save_variables_h5(pickle_file_name, var, info, overwrite);

  elif fext == '.pkl' or fext == '.pklz':
    if os.path.exists(pickle_file_name) and overwrite == False:
      raise Exception('{:s} exists and over write is false.'.format(pickle_file_name))
    if info is not None:
      # Construct the dictionary
      assert(type(var) == list); assert(type(info) == list);
      d = {}
      for i in xrange(len(var)):
        d[info[i]] = var[i]
    else: #we have the dictionary in var
      d = var;
    if fext == '.pkl':
      with open(pickle_file_name, 'wb') as f:
        cPickle.dump(d, f, cPickle.HIGHEST_PROTOCOL)
    else:
      with bz2.BZ2File(pickle_file_name, 'w') as f:
        cPickle.dump(d, f, cPickle.HIGHEST_PROTOCOL)
  else:
    raise Exception('{:s}: extension unknown'.format(fext))

def load_variables(pickle_file_name):
  """
  d = load_variables(pickle_file_name)
  Output:
    d     is a dictionary of variables stored in the pickle file.
  """
  fext = os.path.splitext(pickle_file_name)[1]
  if fext =='.h5':
    return load_variablesh5(pickle_file_name);

  elif fext == '.pkl' or fext == '.pklz':
    if os.path.exists(pickle_file_name):
      if fext == '.pkl':
        with open(pickle_file_name, 'rb') as f:
          d = cPickle.load(f)
      else:
        with bz2.BZ2File(pickle_file_name, 'r') as f:
          d = cPickle.load(f)
      return d
    else:
      raise Exception('{:s} does not exists.'.format(pickle_file_name))
  elif fext == '.json':
    with open(pickle_file_name, 'r') as fh:
        data = json.load(fh)
    return data
  else:
    raise Exception('{:s}: extension unknown'.format(fext))

#wrappers for load_variables and save_variables
def load(pickle_file_name):
    return load_variables(pickle_file_name);

def save(pickle_file_name, var, info, overwrite = False):
    return save_variables(pickle_file_name, var, info, overwrite);

def calc_pr_ovr_noref(counts, out):
  """
  [P, R, score, ap] = calc_pr_ovr(counts, out, K)
  Input    :
    counts : number of occurrences of this word in the ith image
    out    : score for this image
  Output   :
    P, R   : precision and recall
    score  : score which corresponds to the particular precision and recall
    ap     : average precision
  """
  #binarize counts
  out = out.astype(np.float64)
  counts = np.array(counts > 0, dtype=np.float32);
  tog = np.hstack((counts[:,np.newaxis].astype(np.float64), out[:, np.newaxis].astype(np.float64)))
  ind = np.argsort(out)
  ind = ind[::-1]
  score = np.array([tog[i,1] for i in ind])
  sortcounts = np.array([tog[i,0] for i in ind])

  tp = sortcounts;
  fp = sortcounts.copy();
  for i in xrange(sortcounts.shape[0]):
    if sortcounts[i] >= 1:
      fp[i] = 0.;
    elif sortcounts[i] < 1:
      fp[i] = 1.;

  tp = np.cumsum(tp)
  fp = np.cumsum(fp)
  # P = np.cumsum(tp)/(np.cumsum(tp) + np.cumsum(fp));
  P = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

  numinst = np.sum(counts);

  R = tp/numinst

  ap = voc_ap(R,P)
  return P, R, score, ap

def voc_ap(rec, prec):
  # correct AP calculation
  # first append sentinel values at the end
  mrec = np.concatenate(([0.], rec, [1.]))
  mpre = np.concatenate(([0.], prec, [0.]))

  # compute the precision envelope
  for i in range(mpre.size - 1, 0, -1):
    mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

  # to calculate area under PR curve, look for points
  # where X axis (recall) changes value
  i = np.where(mrec[1:] != mrec[:-1])[0]

  # and sum (\Delta recall) * prec
  ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap