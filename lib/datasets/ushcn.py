import os
import sys
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from lib import datasets_path
from .pd_dataset import PandasDataset
from ..utils import sample_mask
from ..utils.utils import geographical_distance, thresholded_gaussian_kernel


class Ushcn(PandasDataset):
    '''
    USHCN Dataset
    1218
    with 0, no nan
    '''
    def __init__(self):
        df, dist, mask = self.load()
        self.dist = dist
        super().__init__(dataframe=df, u=None, mask=mask, name="ushcn", freq='1MS', aggr='nearest')

    def load(self, impute_zeros=True):
        path = os.path.join(datasets_path["ushcn"], 'ushcn_X_wrapped.csv')
        df = pd.read_csv(path, index_col="timestamps")
        datetime_idx = sorted(df.index)
        date_range = pd.date_range(datetime_idx[0], datetime_idx[-1], freq='1MS')
        df.index = date_range
        df = df.replace(0, np.nan)
        mask = ~np.isnan(df.values)
        df = df.replace(np.nan, 0)
        dist = self.load_distance_matrix(list(df.columns))
        return df.astype('float32'), dist, mask.astype('uint8')

    def load_distance_matrix(self, ids):
        path = os.path.join(datasets_path['ushcn'], 'ushcn_dist.npy')
        try:
            dist = np.load(path)
        except:
            dist_path = os.path.join(datasets_path["ushcn"], 'latlon.csv')
            stations = pd.read_csv(dist_path, header=None, names=["lon", "lat"])
            st_coord = stations.loc[:, ['lon', 'lat']]
            dist = geographical_distance(st_coord, to_rad=True).values
            np.save(path, dist)
        return dist

    def get_similarity(self, thr=0.1, include_self=False, force_symmetric=False, sparse=False, **kwargs):
        thr = 0.9
        theta = np.std(self.dist)  # use same theta for both air and air36
        adj = thresholded_gaussian_kernel(self.dist, theta=theta, threshold=thr)
        if not include_self:
            adj[np.diag_indices_from(adj)] = 0.
        if force_symmetric:
            adj = np.maximum.reduce([adj, adj.T])
        if sparse:
            import scipy.sparse as sps
            adj = sps.coo_matrix(adj)
        return adj

    @property
    def mask(self):
        if self._mask is None:
            return self.df.values != 0.
        return self._mask


class MissingValuesUshcn(Ushcn):
    def __init__(self, p_fault=0.0015, p_noise=0.05, mode="random"):
        super(MissingValuesUshcn, self).__init__()
        self.p_fault = p_fault
        self.p_noise = p_noise
        eval_mask = sample_mask(self.numpy().shape,
                                p=p_fault,
                                p_noise=p_noise,
                                mode=mode)
        self.eval_mask = (eval_mask & self.mask).astype('uint8')

    @property
    def training_mask(self):
        return self.mask if self.eval_mask is None else (self.mask & (1 - self.eval_mask))

    def splitter(self, dataset, val_len=0, test_len=0, window=0):
        idx = np.arange(len(dataset))
        if test_len < 1:
            test_len = int(test_len * len(idx))
        if val_len < 1:
            val_len = int(val_len * (len(idx) - test_len))
        test_start = len(idx) - test_len
        val_start = test_start - val_len
        return [idx[:val_start - window], idx[val_start:test_start - window], idx[test_start:]]
