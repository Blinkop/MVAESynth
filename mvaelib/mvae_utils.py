from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class StandardScalerWithoutBinary():
    def __init__(self, binary_columns=None):
        self.binary_columns = [] if binary_columns is None else binary_columns

    def fit(self, df):
        self.columns = df.columns
        self.con_columns = list(set(self.columns) - set(self.binary_columns))

        self.scaler = StandardScaler()
        self.scaler.fit(df[self.con_columns])

    def transform(self, df):
        transformed_data = self.scaler.transform(df[self.con_columns])

        new_data = np.hstack([df[self.binary_columns].values, transformed_data])
        new_df = pd.DataFrame(data=new_data, columns=self.binary_columns + self.con_columns)

        return new_df[self.columns].values

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, data):
        tmp_df = pd.DataFrame(data=data, columns=self.columns)
        tmp_df.loc[:, self.con_columns] = self.scaler.inverse_transform(tmp_df.loc[:, self.con_columns])

        return tmp_df.values

def transform_tr(data):
    coeff_low = 0.0

    bin_cols = ['is_gamer', 'is_parent', 'is_driver', 'has_pets', 'cash_usage']
    cat_cols = []
    con_cols = set(data.columns) - set(bin_cols) - set(cat_cols)

    for col in bin_cols:
        data[col] = (data[col] >= coeff_low).astype('int')
        data.loc[data[col] == 0, col] = -1

    for col in cat_cols:
        data[col] = np.rint(data[col])

    for col in con_cols:
        data.loc[data[col] < 0, col] = 0.0
    
    return data

def transform_vk(data):
    coeff_low = 0.0

    bin_cols = ['sex', 'has_high_education', 'age_hidden', 'mobile_phone', 'twitter', 'facebook', 'instagram', 'movies', 'music', 'quotes']
    cat_cols = ['relation', 'about_topic', 'activities_topic', 'interests_topic', 'personal_alcohol', 'personal_life_main', 'personal_people_main', 'personal_political']
    cat_cols_max = [8, 32, 32, 32, 5, 8, 6, 9]
    con_cols = set(data.columns) - set(bin_cols) - set(cat_cols)

    for col in bin_cols:
        data[col] = (data[col] >= coeff_low).astype('int')
        data.loc[data[col] == 0, col] = -1

    for col, m in zip(cat_cols, cat_cols_max):
        data[col] = np.rint(data[col])
        data[col] = np.minimum(m, data[col])

    for col in con_cols:
        data.loc[data[col] < 0, col] = 0.0

    data.loc[data['age_hidden'] == 1, 'age'] = 0

    data = data.astype('int')
    
    return data

def transform_in(data):
    data[data < 0] = 0.0

    sums = data.sum(axis=1).values

    for col in data.columns:
        data.loc[:, col] = data.loc[:, col] / sums

    return data
