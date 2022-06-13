from sklearn.neighbors import NearestNeighbors
import itertools
import pandas as pd
import numpy as np
import sys
import typing as t

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = list(map(np.radians, [lon1, lat1, lon2, lat2]))

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def haversine_for_pairs(x1, x2):
    '''pairs(lon, lat)'''
    return haversine(*x1, *x2)

def date_filter(x):
    return (0 <= (x['date'] - x['date_analog']).days <= 365 * 2)

def date_not_class_filter(x):
    return (x['classProperty'] != x['all_classProperty_analog']) and \
            (0 <= (x['date'] - x['date_analog']).days <= 365 * 2)

def more_dist_same_class_filter(x):
    return (3 <= x['dist'] >= 10) and \
            (x['classProperty'] == x['all_classProperty_analog']) and \
            (0 <= (x['date'] - x['date_analog']).days <= 365 * 2)

class NearestNeighborsPartitioned:
    def __init__(self, radius: float, part_col: str, db_lat_col: str, db_lon_col: str, n_neighbors:int=5, 
                 n_jobs:int=8, return_columns:t.List=None,
                merge_filter:t.Union[t.Callable, str]=None):
        self.part_col = part_col
        self.db_lat_col = db_lat_col
        self.db_lon_col = db_lon_col
        self.searchers = {}
        self.data = {}
        self.radius = radius
        self.n_jobs = n_jobs
        self.return_columns = return_columns
        self.n_neighbors = n_neighbors
        self.merge_filter =  merge_filter
        
    def set_merge_filter(self, func: t.Union[t.Callable, str]):
        self.merge_filter = func
        
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['merge_filter']
        return state
    
    def _return_data(self, match, query, data):
        ret_columns = [col + '_analog' for col in self.return_columns]
        
        merged = data.merge(
            match, on='_db_index_'
        ).\
        merge(
            query,
            on='_query_index_',
            suffixes=('', '_analog')
        ).drop(['_query_index_', '_db_index_'], axis=1)
        
        if type(self.merge_filter) == str:
            merged = merged.query(self.merge_filter)
        elif self.merge_filter is not None:
            merged = merged[merged.apply(self.merge_filter, axis=1)]
            
        merged = merged[ret_columns + query.columns.difference(['_query_index_']).tolist()]
        
        return merged
        
    @staticmethod
    def _construct_match(query_res: tuple, n_neighbours: int):
        '''
            Создаем датафрейм соответствий из результата применения NearestNeighbors
        '''
        dist, indices = query_res

        match = []

        for elem_num, (di, ind) in enumerate(zip(dist, indices)):
            top_dist = di[:n_neighbours]
            top_ind = ind[:n_neighbours]
            ans_num = range(len(top_dist))

            chunk = list(zip([elem_num] * len(top_dist), top_ind, top_dist, ans_num))

            match.extend(chunk)

        match = pd.DataFrame(match, columns=['_query_index_', '_db_index_', 'dist', 'ans_num'])
        
        return match
        
    def fit(self, data):
        partitions = data[self.part_col].unique()
        partitions = [p for p in partitions if p is not None]
        
        if self.return_columns is None:
            self.return_columns = data.columns.tolist()
        
        for part in partitions:
            partition_data = data.loc[data[self.part_col] == part]
            partition_data = partition_data[~(partition_data[self.db_lat_col].isnull() | partition_data[self.db_lon_col].isnull())]
            
            if len(partition_data) == 0:
                continue
                
            partition_data.columns = [col + '_analog' for col in partition_data.columns]
            
            partition_data['_db_index_'] = list(range(len(partition_data)))
            self.data[part] = partition_data
            self.searchers[part] = NearestNeighbors(radius=self.radius, metric=haversine_for_pairs)
            self.searchers[part].fit(self.data[part][[self.db_lon_col + '_analog', self.db_lat_col + '_analog']])
            
    def transform(self, data, lat_col, lon_col):
        assert self.part_col in data.columns
        partitions = data[self.part_col].unique()
        
        result = []
        for part in partitions:
            if part not in self.data:
                continue
            query_partition_data = data.loc[data[self.part_col] == part]
            query_partition_data['_query_index_'] = list(range(len(query_partition_data)))
            
            df = query_partition_data[[lon_col, lat_col]]
            
            find = self.searchers[part].radius_neighbors(X=df, radius=self.radius, sort_results=True)
            
            match = self._construct_match(find, self.n_neighbors)
            
            matched = self._return_data(match, query_partition_data, self.data[part])
            
            result.append(matched)
            
        return pd.concat(result, ignore_index=True)