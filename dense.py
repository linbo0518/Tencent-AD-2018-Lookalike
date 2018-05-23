import gc
import time
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

START_TIME = time.time()
PATH = '../input/'


# macOS pickle dump and load
class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        buffer = bytearray(n)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            buffer[idx:idx + batch_size] = self.f.read(batch_size)
            idx += batch_size
        return buffer

    def write(self, buffer):
        n = len(buffer)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            self.f.write(buffer[idx:idx + batch_size])
            idx += batch_size


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))


def print_status(START_TIME, str=None):
    print('time:', int(time.time() - START_TIME), str)


def base_feature_num(dataframe, feature_list):
    '''
    count the number of feature in the dataframe

    param   dataframe: pd.DataFrame
            feature_list: list

    return  num_dict: dict
    '''
    num_dict = dict()
    n = 0
    for feature in feature_list:
        num_dict[feature] = dict()
        if type(dataframe[feature]) == 'object':
            for data in dataframe[feature]:
                value_list = data.split(' ')
                for value in value_list:
                    if value in num_dict[feature]:
                        num_dict[feature][value] += 1
                    else:
                        num_dict[feature][value] = 1
        else:
            for value in dataframe[feature]:
                if value in num_dict[feature]:
                    num_dict[feature][value] += 1
                else:
                    num_dict[feature][value] = 1
        n += 1
        print('feature num: %d/%d' % (n, len(feature_list)), end='\r')
    return num_dict


def base_feature_weight(num_dict):
    '''
    convert the number of feature to the weight of feature

    param   num_dict: dict

    return  weight_dict: dict
    '''
    weight_dict = dict()
    n = 0
    for feature in num_dict:
        weight_dict[feature] = dict()
        num_all = 0
        for value in num_dict[feature]:
            if value == '0':
                pass
            else:
                num_all += num_dict[feature][value]
        for value in num_dict[feature]:
            if value == '0':
                weight_dict[feature][value] = 0
            else:
                weight_dict[feature][value] = num_dict[feature][value] / num_all
        n += 1
        print('feature weight: %d/%d' % (n, len(num_dict)), end='\r')
    return weight_dict


def apply_base_weight(dataframe, weight_dict):
    '''
    apply base feature weight to the dataframe

    param   dataframe: pd.DataFrame
            weight_dict: dict

    return  feature_col: dict
    '''
    feature_col = dict()
    n = 0
    for feature in weight_dict:
        feature_col[feature] = list()
        if type(dataframe[feature]) == 'object':
            for data in dataframe[feature]:
                value_list = data.split(' ')
                value_weight = 0
                for value in value_list:
                    try:
                        value_weight += weight_dict[feature][value]
                    except:
                        value_weight += 0
                feature_col[feature].append(value_weight)
        else:
            for value in dataframe[feature]:
                try:
                    feature_col[feature].append(weight_dict[feature][value])
                except:
                    feature_col[feature].append(0)
        n += 1
        print('apply feature: %d/%d' % (n, len(weight_dict)), end='\r')
    return feature_col


def ad_feature_num(dataframe, feature_list):
    '''
    count the number of feature in the dataframe according to aid

    param   dataframe: pd.DataFrame
            feature_list: list
    
    return  num_dict: dict
    '''
    ad_list = dataframe['aid'].unique()
    num_dict = dict()
    n = 0
    for aid in ad_list:
        tmp = dataframe[dataframe['aid'] == aid]
        num_dict[aid] = dict()
        num_dict[aid] = base_feature_num(tmp, feature_list)
        n += 1
        print('ad num: %d/%d' % (n, len(ad_list)), end='\r')
    return num_dict


def ad_feature_weight(num_dict):
    '''
    convert the number of feature to the weight of feature according ad

    param   num_dict: dict

    return  weight_dict: dict
    '''
    weight_dict = dict()
    n = 0
    for aid in num_dict:
        weight_dict[aid] = dict()
        weight_dict[aid] = base_feature_weight(num_dict[aid])
        n += 1
        print('ad weight: %d/%d' % (n, len(num_dict)), end='\r')
    return weight_dict


def apply_ad_weight(dataframe, feature_list, weight_dict):
    '''
    apply ad feature weight to the dataframe

    param   dataframe: pd.DataFrame
            feature_list: list
            weight_dict: dict
    
    return  feature_col: dict
    '''
    feature_col = dict()
    for feature in feature_list:
        feature_col[feature] = list()
    n = 0
    for _, row in dataframe.iterrows():
        aid = row['aid']
        for feature in feature_list:
            if type(dataframe[feature]) == 'object':
                value_list = row[feature].split(' ')
                value_weight = 0
                for value in value_list:
                    try:
                        value_weight += weight_dict[aid][feature][value]
                    except:
                        value_weight += 0
                feature_col[feature].append(value_weight)
            else:
                value = row[feature]
                try:
                    feature_col[feature].append(
                        weight_dict[aid][feature][value])
                except:
                    feature_col[feature].append(0)
        n += 1
        if n % 1000 == 0:
            print('row num: %d/%d' % (n, dataframe.shape[0]), end='\r')
    return feature_col


def base_ctr(dataframe):
    '''
    count base ctr in dataframe

    param   dataframe: pd.DataFrame

    return  ctr_dict: dict
    '''
    ad_list = dataframe['aid'].unique()
    ctr_dict = dict()
    n = 0
    for aid in ad_list:
        tmp = dataframe[dataframe['aid'] == aid]
        hit_num = tmp[tmp['label'] == 1].shape[0]
        all_num = tmp.shape[0]
        ctr_dict[aid] = hit_num / all_num
        n += 1
        print('base ctr: %d/%d' % (n, len(ad_list)), end='\r')
    return ctr_dict


# def label_1_base_feature_weight(dataframe, feature_list):
#     '''
#     count feature which label equal 1 weight in dataframe

#     param   dataframe: pd.DataFrame
#             feature_list: list
    
#     return  weight_dict: dict
#     '''
#     weight_dict = dict()
#     tmp = dataframe[dataframe['label'] == 1]
#     num_dict = base_feature_num(tmp, feature_list)
#     weight_dict = base_feature_weight(num_dict)
#     return weight_dict


# def label_1_ad_feature_weight(dataframe, feature_list):
#     '''
#     count feature which label equal 1 in each ad in dataframe

#     param   dataframe: pd.DataDrame
#             feature_list: list
    
#     return  weight_dict
#     '''
#     ad_list = dataframe['aid'].unique()
#     weight_dict = dict()
#     n = 0
#     for aid in ad_list:
#         tmp = dataframe[(dataframe['aid'] == aid) & (dataframe['label'] == 1)]
#         weight_dict[aid] = dict()
#         num_dict = base_feature_num(tmp, feature_list)
#         weight_dict[aid] = base_feature_weight(num_dict)
#         n += 1
#         print('ad weight: %d/%d' % (n, len(ad_list)), end='\r')
#     return weight_dict


def feature_min_mean_max(dataframe, feature):
    '''
    param   dataframe: pd.DataFrame
            feature: str
    
    return  min_list: list
            mean_list: list
            max_list: list
    '''
    min_list = list()
    mean_list = list()
    max_list = list()
    for data in dataframe[feature]:
        min_val = float('inf')
        max_val = 0
        value_list = data.split(' ')
        n = 0
        add_all = 0
        for value in value_list:
            value = int(value)
            if min_val > value:
                min_val = value
            if max_val < value:
                max_val = value
            add_all += value
            n += 1

        min_list.append(min_val)
        mean_list.append(add_all / n)
        max_list.append(max_val)

    return min_list, mean_list, max_list


# load data from local storage
print_status(START_TIME, 'start load data...')
train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + "test2.csv")
ad_feature = pd.read_csv(PATH + 'adFeature.csv')
user_feature = pd.read_csv(PATH + 'userFeature.csv')
print_status(START_TIME, 'finish load data!')

# merge data
print_status(START_TIME, 'start merge data...')
n_train = train.shape[0]
merge = pd.concat([train, test])
merge.loc[merge['label'] == -1, 'label'] = 0
merge = pd.merge(merge, ad_feature, on='aid', how='left')
merge = pd.merge(merge, user_feature, on='uid', how='left')
merge = merge.fillna('0')
int_features = ['label', 'LBS', 'house']
for feature in int_features:
    merge[feature] = merge[feature].astype(int)
del train, test, ad_feature, user_feature
gc.collect()
print_status(START_TIME, 'finish merge data!')

# feature list
feature_list = [
    'advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId',
    'productId', 'productType', 'age', 'gender', 'marriageStatus', 'education',
    'consumptionAbility', 'LBS', 'interest1', 'interest2', 'interest3',
    'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3',
    'appIdInstall', 'appIdAction', 'ct', 'os', 'carrier', 'house'
]

# preprocessing
print_status(START_TIME, 'start base feature weight')
num_dict = base_feature_num(merge, feature_list)
weight_dict = base_feature_weight(num_dict)
feature_col = apply_base_weight(merge, weight_dict)
for feature in feature_col:
    merge[feature + '_base_weight'] = feature_col[feature]
print_status(START_TIME, 'finish base feature weight')
del num_dict, weight_dict, feature_col
gc.collect()

print_status(START_TIME, 'start ad feature weight')
num_dict = ad_feature_num(merge, feature_list)
weight_dict = ad_feature_weight(num_dict)
feature_col = apply_ad_weight(merge, feature_list, weight_dict)
for feature in feature_col:
    merge[feature + '_ad_weight'] = feature_col[feature]
print_status(START_TIME, 'finish ad feature weight')
del num_dict, weight_dict, feature_col
gc.collect()

print_status(START_TIME, 'start min mean max')
multivalue_feature = [
    'interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1',
    'kw2', 'kw3', 'topic1', 'topic2', 'topic3', 'appIdInstall', 'appIdAction',
    'ct', 'os'
]
for feature in multivalue_feature:
    min_list, mean_list, max_list = feature_min_mean_max(merge, feature)
    merge[feature + '_min'] = min_list
    merge[feature + '_mean'] = mean_list
    merge[feature + '_max'] = max_list
print_status(START_TIME, 'finish min mean max')
del multivalue_feature
gc.collect()

print_status(START_TIME, 'start base ctr')

ctr_dict = base_ctr(merge)
feature_col = list()
n = 0
for _, row in merge.iterrows():
    aid = row['aid']
    feature_col.append(ctr_dict[aid])
    n += 1
    if n % 1000 == 0:
        print('row num %d/%d' % (n, merge.shape[0]), end='\r')
merge['base_ctr'] = feature_col
print_status(START_TIME, 'finish base ctr')
del ctr_dict, feature_col
gc.collect()

# print_status(START_TIME, 'start label 1 base feature weight')
# weight_dict = label_1_base_feature_weight(merge, feature_list)
# feature_col = apply_base_weight(merge, weight_dict)
# for feature in feature_col:
#     merge[feature + '_label1_base_weight'] = feature_col[feature]
# print_status(START_TIME, 'finish label 1 base feature weight')
# del weight_dict, feature_col
# gc.collect()

# print_status(START_TIME, 'start label 1 ad feature weight')
# weight_dict = label_1_ad_feature_weight(merge, feature_list)
# feature_col = apply_ad_weight(merge, feature_list, weight_dict)
# for feature in feature_col:
#     merge[feature + '_label1_ad_weight'] = feature_col[feature]
# print_status(START_TIME, 'finish label 1 ad feature weight')
# del weight_dict, feature_col
# gc.collect()

# split merge into train and test
label = merge['label'][:n_train]
merge = merge.drop(
    [
        'aid', 'label', 'uid', 'advertiserId', 'campaignId', 'creativeId',
        'creativeSize', 'adCategoryId', 'productId', 'productType', 'age',
        'gender', 'marriageStatus', 'education', 'consumptionAbility', 'LBS',
        'interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1',
        'kw2', 'kw3', 'topic1', 'topic2', 'topic3', 'appIdInstall',
        'appIdAction', 'ct', 'os', 'carrier', 'house'
    ],
    axis=1)
train = merge[:n_train]
test = merge[n_train:]
del merge, n_train
gc.collect()

# split train into train and valid
train_X, valid_X, train_y, valid_y = train_test_split(
    train, label, test_size=0.2)
del train, label
gc.collect()

# prepare for training
train_data = lgb.Dataset(train_X, train_y)
valid_data = lgb.Dataset(valid_X, valid_y)
del train_X, train_y, valid_X, valid_y
gc.collect()

# params of lightgbm
params = {
    'application': 'binary',
    'boosting': 'gbdt',
    'num_leaves': 80,
    'min_data_in_leaf': 1000,
    'learning_rate': 0.05,
    'zero_as_missing': True,
    'metric': ['binary_logloss', 'auc']
}

# start training
print_status(START_TIME, 'start training')
model = lgb.train(
    params,
    train_data,
    1000,
    valid_sets=[train_data, valid_data],
    valid_names=['train', 'valid'],
    verbose_eval=50,
    early_stopping_rounds=100)
print_status(START_TIME, 'finish training')

# prediction
print_status(START_TIME, 'start prediction')
sub = pd.read_csv(PATH + 'test2.csv')
sub['score'] = model.predict(test)
sub.to_csv('out/submission.csv', float_format='%.8f', index=False)
print_status(START_TIME, 'finish prediction')

print_status(START_TIME, 'ALL FINISH!')