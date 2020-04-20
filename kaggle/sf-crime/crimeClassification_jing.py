import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from matplotlib import cm
import urllib.request
import shutil
import zipfile
import os
import re
import lightgbm as lgb
import eli5
from eli5.sklearn import PermutationImportance
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

def feature_engineering(data):
    data['Date'] = pd.to_datetime(data['Dates'].dt.date)
    data['n_days'] = (
            data['Date'] - data['Date'].min()).apply(lambda x: x.days)
    data['Day'] = data['Dates'].dt.day
    data['DayOfWeek'] = data['Dates'].dt.weekday
    data['Month'] = data['Dates'].dt.month
    data['Year'] = data['Dates'].dt.year
    data['Hour'] = data['Dates'].dt.hour
    data['Minute'] = data['Dates'].dt.minute
    data['Block'] = data['Address'].str.contains('block', case=False)

    data.drop(columns=['Dates','Date','Address'], inplace=True)

    return data

# Loading the data
train = pd.read_csv('train.csv', parse_dates=['Dates'])
tempTrain = train
test = pd.read_csv('test.csv', parse_dates=['Dates'], index_col='Id')

# Data cleaning
# train.drop_duplicates(inplace=True)
# train.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)
# test.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)
#
# imp = SimpleImputer(strategy='mean')
#
# for district in train['PdDistrict'].unique():
#     train.loc[train['PdDistrict'] == district, ['X', 'Y']] = imp.fit_transform(
#         train.loc[train['PdDistrict'] == district, ['X', 'Y']])
#     test.loc[test['PdDistrict'] == district, ['X', 'Y']] = imp.transform(
#         test.loc[test['PdDistrict'] == district, ['X', 'Y']])
#train_data = lgb.Dataset(train, label=y, categorical_feature=['PdDistrict'], free_raw_data=False)

# Feature Engineering
train = feature_engineering(train)
train.drop(columns=['Descript','Resolution'], inplace=True)
test = feature_engineering(test)
tempTest = test

# Encoding the Categorical Variables
le1 = LabelEncoder()
train['PdDistrict'] = le1.fit_transform(train['PdDistrict'])
test['PdDistrict'] = le1.transform(test['PdDistrict'])

le2 = LabelEncoder()
X = train.drop(columns=['Category'])
y = le2.fit_transform(train['Category'])


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test  = train_test_split(X, train['Category'])

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=100, max_depth=None)
forest.fit(x_train, y_train)

print('training set accuracy:', forest.score(x_train, y_train))
print('test set accuracy:', forest.score(x_test, y_test))

predictions = forest.predict_proba(test)

# # Creating the model
# train_data = lgb.Dataset(
#     X, label=y, categorical_feature=['PdDistrict'])
#
# params = {'boosting':'gbdt',
#           'objective':'multiclass',
#           'num_class':39,
#           'max_delta_step':0.9,
#           'min_data_in_leaf': 21,
#           'learning_rate': 0.4,
#           'max_bin': 465,
#           'num_leaves': 41
#           }
#
# bst = lgb.train(params, train_data, 100)
#
# predictions = bst.predict(test)
#
#
# Submitting the results
submission = pd.DataFrame(
    predictions,
    columns=le2.inverse_transform(np.linspace(0, 38, 39, dtype='int16')),
    index=test.index)
submission.to_csv(
    'RF_final.csv', index_label='Id')


# model = LGBMClassifier(**params).fit(X, y, categorical_feature=['PdDistrict'])
#
# pdp_Pd = pdp.pdp_isolate(
#     model=model,
#     dataset=X,
#     model_features=X.columns.tolist(),
#     feature='Hour',
#     n_jobs=-1)
#
# pdp.pdp_plot(
#     pdp_Pd,
#     'Hour',
#     ncols=3)
# plt.show()
