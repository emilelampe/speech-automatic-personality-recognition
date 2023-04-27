#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run it with:
    python3 script.py -p ipy_profile
where ipy_profile is the name of the ipython profile.
@author: hyamanieu
"""

import numpy as np
import datetime
import argparse
import logging
import os
import sys
import joblib
from math import ceil
from datetime import datetime
from random import randint
from csv import writer

from sklearn.base import BaseEstimator
sys.modules['sklearn.externals.joblib'] = joblib
from sklearn.externals.joblib import Parallel, parallel_backend
from sklearn.externals.joblib import register_parallel_backend
from sklearn.externals.joblib import delayed
from sklearn.externals.joblib import cpu_count
from ipyparallel import Client
from ipyparallel.joblib import IPythonParallelBackend
from imblearn.metrics import classification_report_imbalanced
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SVMSMOTE
from imblearn.metrics import geometric_mean_score
from joblib import dump, load
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import det_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn import clone, decomposition
from sklearn.calibration import CalibratedClassifierCV
from shutil import rmtree
from tempfile import mkdtemp
import time
from skmultilearn.dataset import load_dataset_dump
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from my_functions import my_train_test_split
import numpy as np
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import IterativeStratification
from sklearn.model_selection import GroupKFold, BaseCrossValidator
import multilabel_stratified_group_split as msgs
from custom_cv import NestedStratifiedGroupKFold
from sklearn.model_selection import PredefinedSplit


# Custom StratifiedGroupKFold cross-validator
class MyStratifiedGroupKFold(BaseCrossValidator):

    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def _iter_test_masks(self, X=None, y=None, groups=None):
        stratified_kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        group_kfold = GroupKFold(n_splits=self.n_splits)

        for train_index, test_index in group_kfold.split(X, y, groups):
            stratified_train_index, stratified_test_index = next(stratified_kfold.split(X[train_index], y[train_index]))
            test_mask = np.zeros_like(y, dtype=bool)
            test_mask[train_index[stratified_test_index]] = True
            yield test_mask


def percentage_binary_sample(arr, idx, bin=1):
    unique, counts = np.unique(arr[:,idx], return_counts=True)

    counts_per_bin = dict(zip(unique, counts))
    n_ones = counts_per_bin[1]
    n_zeros = counts_per_bin[0]
    total = n_ones + n_zeros
    if bin == 1:
        return round(n_ones / total, 3)
    else:
        return round(n_zeros / total, 3)

def output_df_results_report(y_test, y_pred, y_score, m, t):

    df1 = pd.DataFrame(classification_report_imbalanced(
        y_test, y_pred, output_dict=True)[0], index=[0])
    df2 = pd.DataFrame(classification_report_imbalanced(
        y_test, y_pred, output_dict=True)[1], index=[1])

    df3 = pd.concat([df1, df2])
    df3['roc_auc'] = roc_auc_score(y_test, y_score)
    df3['bal_acc'] = balanced_accuracy_score(y_test, y_pred)
    df3['acc'] = accuracy_score(y_test, y_pred)
    df3['Value'] = df3.index
    df3['Trait'] = t
    df3['Model'] = m

    row = []
    for i in range(6):
        row.append((df1.iloc[0, i] + df2.iloc[0, i]) / 2)
    row.append(df1.iloc[0, 6] + df2.iloc[0, 6])
    row.append(df3.iloc[0, 7])
    row.append(df3.iloc[0, 8])
    row.append(df3.iloc[0, 9])
    row.append('avg/total')
    row.append(t)
    row.append(m)

    df3.loc[len(df3)] = row

    for i in range(7, 13, 1):
        col_name = df3.columns.values[i]
        col = df3.pop(col_name)
        df3.insert(0, col_name, col)

    return df3


starttime = time.time()

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR)

random_id = randint(1, 10000)

# prepare the logger
parser = argparse.ArgumentParser()

parser.add_argument("-p", "--profile", default="ipy_profile",
                    help="Name of IPython profile to use")

parser.add_argument("-m", "--model", default="svm_rbf",
                    help="The model with which you want to train")

parser.add_argument("-t", "--trait", default="Extraversion",
                    help="The trait on which you want to train")

parser.add_argument("-f", "--featureset", default="egemaps",
                    help="The feature set to use")

parser.add_argument("-l", "--labelmode", default="half",
                    help="Whether the labels are binarized on the half point or on the median")

parser.add_argument("-d", "--database", default="nscc", help="The database to train")

parser.add_argument("-b", "--batchid", default=str(random_id), help="The ID with which different searches can be grouped")

parser.add_argument("-s", "--startcutoff", default="0",
                    help="The minimum length of a sample")

parser.add_argument("-e", "--endcutoff", default="0",
                    help="The maximum length of a sample")

args = parser.parse_args()

profile = args.profile

m = args.model

t = args.trait

f = args.featureset

l = args.labelmode

d = args.database

b = args.batchid

sc = args.startcutoff

ec = args.endcutoff

sc = int(sc)

ec = int(ec)

trait_dict = {
    'E': 'Extraversion',
    'A': 'Agreeableness',
    'C': 'Conscientiousness',
    'N': 'Neuroticism',
    'O': 'Openness'
}

feat_dict = {
    'e': 'egemaps',
    'c': 'compare'
}

label_dict = {
    'h': 'half',
    'm': 'median'
}

database_dict = {
    'n': 'nscc',
    's': 'sspn',
    'nn': 'nscc_normalized',
    'nnb': 'nscc_normalized_bfi44',
    '75': 'nscc_normalized_bfi44_75',
    'rc75': 'nscc_recalc_bfi44_75',
    'c': 'chatbot',
    'cc': 'chatbot_combined'
}

t = t.capitalize()
if t in trait_dict.keys():
    t = trait_dict[t]

f = f.lower()
if f in feat_dict.keys():
    f = feat_dict[f]

l = l.lower()
if l in label_dict.keys():
    l = label_dict[l]

d = d.lower()
if d in database_dict.keys():
    d = database_dict[d]

if d == 'sspn':
    l = 'average'



datetime_start = datetime.now().strftime("%d_%m_%y_%H_%M_%S")

logfilename = os.path.join(
    FILE_DIR, f'log/log-{datetime_start}-{b}-{d}-{f}-{l}-{m}-{t}.log')
logging.basicConfig(filename=logfilename,
                    filemode='w',
                    level=logging.DEBUG)
logging.info("number of CPUs found: {0}".format(cpu_count()))
logging.info("args.profile: {0}".format(profile))
logging.info("Batch: %s" % b)
logging.info("FILE_DIR: %s" % FILE_DIR)

# check whether the batchid folder exists
batch_path = f"{FILE_DIR}/data/{b}"
isExist = os.path.exists(batch_path)
# if it doesn't exist, create new folder
if not isExist:
   os.makedirs(batch_path)
   os.makedirs(f"{batch_path}/best-estimators")
   os.makedirs(f"{batch_path}/confusion-matrices")
   os.makedirs(f"{batch_path}/cv-results")
   os.makedirs(f"{batch_path}/results-reports")
   os.makedirs(f"{batch_path}/y-test-predictions")

logging.info("Folders for batch checked or made")

# def append_and_import(p):
#     import sys
#     sys.path.append(p)


# prepare the engines
c = Client(profile=profile)
# The following command will make sure that each engine is running in
# the right working directory to access the custom function(s).
c[:].map(os.chdir, [FILE_DIR]*len(c))

# custom_cv_path = os.path.abspath('custom_cv.py')
# c[:].apply_sync(append_and_import, custom_cv_path)

logging.info("c.ids :{0}".format(str(c.ids)))
bview = c.load_balanced_view()
register_parallel_backend('ipyparallel',
                          lambda: IPythonParallelBackend(view=bview))



# random seed for all random states
seed = 42

# CONFIG PATHS
full_df = pd.read_pickle(f"{FILE_DIR}/db/full-{d}-{f}-{l}.pkl")

# adjust time cutoff
if sc != 0:
    full_df = full_df[(full_df['Length'] >= sc * 1000)]
if ec != 0:
    full_df = full_df[(full_df['Length'] <= ec * 1000)]

X = full_df.iloc[:, 9:]
y = full_df.iloc[:,3:9]

# print(len(X))
# print(len(y))

feature_names = list(X.columns.values)
# skip gender in the label names
label_names = list(y.columns.values)

groups = full_df['Group'].to_numpy()

X = X.to_numpy()
y = y.to_numpy()



X_train, X_test, y_trains, y_tests, train_indices, test_indices = msgs.multilabel_stratified_group_train_test_split(X, y, groups, test_size=0.2, random_state=seed)

msgs.check_disjoint_groups(train_indices, test_indices, groups)

msgs.stratification_report(y, y_trains, y_tests, label_names)

# drop gender column
y = y[:,1:]
y_trains = y_trains[:,1:]
y_tests = y_tests[:,1:]
label_names = label_names[1:]

t_idx = label_names.index(t)

y_train = y_trains[:,t_idx]
y_test = y_tests[:,t_idx]

train_groups = groups[train_indices]
test_groups = groups[test_indices]

# step size for RFECV
if f == 'compare':
    # step = 0.2
    step = 400
elif f == 'egemaps':
    step = 3

from sklearn.model_selection import KFold

class ReindexedKFold:
    def __init__(self, n_splits):
        self.kfold = KFold(n_splits=n_splits)

    def get_n_splits(self, X, y, groups=None):
        return self.kfold.get_n_splits(X, y, groups)

    def split(self, X, y, groups=None):
        for train_idx, test_idx in self.kfold.split(X, y, groups):
            yield (np.arange(len(train_idx)), np.arange(len(test_idx)) + len(train_idx))


# inner_cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=seed)


# sgkf_inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
# cv_inner = list(sgkf_inner.split(X_train, y_train, groups=train_groups))

# # sgkf_outer = NestedStratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
# # cv_outer = list(sgkf_outer.split(X_train, y_train, groups=train_groups))

# nested_cv = NestedStratifiedGroupKFold(n_splits_outer=5, n_splits_inner=5, shuffle=True, random_state=seed)

# cv_outer, cv_inner = nested_cv.split(X_train, y_train, groups=train_groups)
# # sgkf2 = MyStratifiedGroupKFold(n_splits=5)
# # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)





# create sample strategy here, because passthrough + params gives an error
# first determine the minimal ratio for the resample strategy (must be higher than the current ratio)
n_zero = (y_train == 0).sum()
n_one = (y_train == 1).sum()
minclass = min(n_zero, n_one)
majclass = max(n_zero, n_one)
ratio = minclass/majclass
# round up to next even decimal and determine amount of needed steps
min_strat = ceil(ratio / 2 * 10) * 2 / 10
steps = int((10-(min_strat*10))/2+1)

# strategy from lowest round up even decimal, in steps of 0.2 to 1
s_vals = np.linspace(min_strat, 1, steps)

s_ss = []
for i, x in enumerate(s_vals):
    ss1 = SMOTETomek(sampling_strategy=x, random_state=seed)
    ss2 = SMOTEENN(sampling_strategy=x, random_state=seed)
    s_ss.append(ss1)
    s_ss.append(ss2)
s_ss.append('passthrough')

C_range = np.logspace(-3, 6, 10)
gamma_range = np.logspace(-7, 2, 10)
max_depth_range = np.logspace(1, 4, 4, base=2)
max_depth_range = list(max_depth_range)
max_depth_range = [int(x) for x in max_depth_range]
max_depth_range.append(None)

# model parameters
model_pars = {
    # SVM with linear kernel
    'svm_l': {
        # 'clf__base_estimator': [LinearSVC(random_state=seed)],
        # 'clf__base_estimator__C': C_range
    },
    # # SVM with RBF kernel
    'svm_rbf': {
    #     'clf__base_estimator__kernel': ['rbf'],
    #     # 'clf__base_estimator__C': C_range,
    #     'clf__base_estimator__C': [1, 10, 100, 1000],
    #     # 'clf__base_estimator__gamma': gamma_range,
    #     'clf__base_estimator__gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001],
    #     'clf__base_estimator__shrinking': [False]
    },
    # Random Forest
    'rf':
    {
        # 'clf__base_estimator': [RandomForestClassifier(random_state=seed, n_jobs=1)],
        # 'clf__base_estimator__n_estimators': [10, 100, 250, 500, 1000, 1500],
        # 'clf__base_estimator__max_depth': max_depth_range
    },
    # kNeirestNeighbour
    'knn':
    {
        # 'clf__base_estimator': [KNeighborsClassifier(n_jobs=1)],
        # 'clf__base_estimator__n_neighbors': list(range(1, 21, 1))
    }
}

# preprocessing parameters
# pre_pars = {
#         # 'clf__method': ['sigmoid', 'isotonic'],
#         'clf__method': ['sigmoid'],
#         # 'scaler': [StandardScaler(), MinMaxScaler()],
#         'scaler': [StandardScaler()],
#         # 'dim_red': [PCA(0.95), 'passthrough'],
#         'dim_red': ['passthrough'],
#         # 'sampling': ['passthrough', SMOTETomek(sampling_strategy=1, random_state=seed)]
#         'sampling': ['passthrough']
#     }

pre_pars = {
        # 'clf__method': ['sigmoid', 'isotonic'],
        'clf__method': ['sigmoid'],
        # 'scaler': [StandardScaler(), MinMaxScaler()],
        'scaler': [StandardScaler()],
        # 'dim_red': [PCA(0.95), 'passthrough'],
        # 'dim_red': ['passthrough'],
        # 'sampling': ['passthrough', SMOTETomek(sampling_strategy=1, random_state=seed)]
        # 'sampling': ['passthrough']
    }
    
# combine model and preprocessing parameters into one space
model_pars[m].update(pre_pars)
param_space = model_pars[m]

logging.info("Parameters: %s" % str(param_space))

# Initialize lists to store the results
probas_list = []
y_pred_list = []
y_score_list = []
test_scores = []

from sklearn.model_selection import ParameterGrid

class GridSearchNoCV(BaseEstimator):
    def __init__(self, estimator, param_grid, X_train, y_train, X_test, y_test, scoring_func=accuracy_score, n_jobs=1):
        self.estimator = estimator
        self.param_grid = param_grid
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.scoring_func = scoring_func
        self.best_params_ = None
        self.best_score_ = None
        self.n_jobs = n_jobs

    def _fit_and_score(self, estimator, X_train, X_test, y_train, y_test, params, scoring_func):
        estimator.set_params(**params)
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        score = scoring_func(y_test, y_pred)
        return score, params

    def fit(self, X, y):
        param_grid = ParameterGrid(self.param_grid)
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_and_score)(
                clone(self.estimator), self.X_train, self.X_test, self.y_train, self.y_test, params, self.scoring_func
            )
            for params in param_grid
        )
        
        scores, parameters = zip(*results)
        best_score = max(scores)
        best_params = parameters[scores.index(best_score)]
        
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.estimator.set_params(**best_params)
        self.estimator.fit(self.X_train, self.y_train)

    def predict(self, X):
        return self.estimator.predict(X)


cv_splits = 5
outer_cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

cache_list = [None] * 5

cache_idx = 0
# Loop over outer CV splits
for train_idx, test_idx in outer_cv.split(X_train, y_train, groups=train_groups):
    # Get training and test sets for this split
    X_train_outer, X_test_outer = X_train[train_idx], X_train[test_idx]
    y_train_outer, y_test_outer = y_train[train_idx], y_train[test_idx]

    # Get the inner groups for the inner CV
    train_groups_outer = train_groups[train_idx]
    test_groups_outer = train_groups[test_idx]



    inner_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    # Print some diagnostic information
    print(f"Iteration {cache_idx}")
    print("X_train_inner shape:", X_train_outer.shape)
    print("y_train_inner shape:", y_train_outer.shape)

    # create the cache to save the pipe to
    cache_list[cache_idx] = mkdtemp()

    # instantiate pipe with placeholders and cache
    pipe = Pipeline(steps=[
        ('sampling', 'passthrough'),
        ('scaler', StandardScaler()),
        ('dim_red', 'passthrough'),
        # ('feat_sel', RFECV(
        #     estimator=DecisionTreeClassifier(random_state=seed),
        #     cv=list(inner_cv.split(X_train_outer, y_train_outer, groups=train_groups_outer)),
        #     step=step,
        #     scoring=main_scoring,
        #     n_jobs=1,
        #     min_features_to_select=1)), 
        ('feat_sel', 'passthrough'), 
        ('clf', CalibratedClassifierCV(
            base_estimator=SVC(), 
            n_jobs=1,
            ensemble=True,
            # cv=list(inner_cv.split(X_train_outer, y_train_outer, groups=train_groups_outer))))
            cv=list(inner_cv.split(X_train_outer, y_train_outer))))
            ], memory=cache_list[cache_idx])
    
    # instantiate grid search
    search = GridSearchNoCV(pipe,
                        param_space, X_train=X_train_outer, y_train=y_train_outer, X_test=X_test_outer, y_test=y_test_outer,
                        n_jobs=len(c), scoring_func=roc_auc_score)
    
    # run with multiprocessing
    with parallel_backend('ipyparallel'):
        search.fit(X_train_outer, y_train_outer)

    # make predictions on test set
    probas_ = search.predict_proba(X_test_outer)
    y_pred_single = np.argmax(probas_, axis=1)
    y_score_single = probas_[:, 1]

    # Store the results for this split
    probas_list.append(probas_)
    y_pred_list.append(y_pred_single)
    y_score_list.append(y_score_single)
    

    cache_idx += 1


# Calculate the average score over all splits
y_pred = np.mean(y_pred_list)
y_score = np.mean(y_score_list)



# pipe = Pipeline(steps=[
#      ('scaler', StandardScaler()),
#         ('clf', CalibratedClassifierCV(
#         base_estimator=SVC(), 
#         n_jobs=1,
#         ensemble=True))], memory=cache)

# for fold_idx, (train_index, val_index) in enumerate(sgkf_inner.split(X=X_train, y=y_train, groups=train_groups)):
#     print(f'Fold {fold_idx}')
#     print(f'Train: {len(train_index)} samples, {len(set(train_groups[train_index]))} groups')
#     print(f'Test: {len(val_index)} samples, {len(set(train_groups[val_index]))} groups')



# print("JUST BEFORE")
# print(len(X_train), len(y_train), len(train_groups))




# search.fit(X_train, y_train)

# logging.info("Best pipeline: %s" % search.best_estimator_)
# logging.info("Best score during training: %s" % search.best_score_)

# # make predictions on test set
# probas_ = search.predict_proba(X_test)

# y_pred = np.argmax(probas_, axis=1)
# y_score = probas_[:, 1]

# y_pred = search.predict(X_test)
# y_score = search.decision_function(X_test)

acc = round(accuracy_score(y_test, y_pred),4)
bal_acc = round(balanced_accuracy_score(y_test,y_pred),4)
rocauc = round(roc_auc_score(y_test,y_score),4)
geo_m = round(geometric_mean_score(y_test, y_pred),4)
f1_s = round(f1_score(y_test,y_pred),4)
con_mat = confusion_matrix(y_test, y_pred)

logging.info("\n%s" % str(con_mat))
logging.info("Best score during test: %s" % rocauc)

db_with_times = f"{d}_{sc}_{ec})"

# create the string to add to the main results file
main_results_string = [datetime_start,b,db_with_times,f,l,m,t,acc,bal_acc,rocauc,geo_m,f1_s]
for x in search.best_estimator_:
    main_results_string.append(str(x).replace('\n', ''))

# write string to the main results file
with open(f"{FILE_DIR}/data/main_results.csv", 'a') as fw:
    w = writer(fw, delimiter=';')
    w.writerow(main_results_string)
    fw.close()

# get results
results = {
    'model': search.best_estimator_,
    'rr': output_df_results_report(y_test, y_pred, y_score, m, t),
    'cvr': pd.DataFrame(search.cv_results_),
    'cm': pd.DataFrame(con_mat),
    'ytp': (y_test, y_pred, y_score)
}

# create addition with info to the filename
fileadd = f"{b}-{datetime_start}-{db_with_times}-{f}-{l}-{m}-{t}"

# make filenames
filenames = {
    'model': f"{batch_path}/best-estimators/best_est-{fileadd}",
    'rr': f"{batch_path}/results-reports/res_rep-{fileadd}",
    'cvr': f"{batch_path}/cv-results/cv_res-{fileadd}",
    'cm': f"{batch_path}/confusion-matrices/con_mat-{fileadd}",
    'ytp': f"{batch_path}/y-test-predictions/y_test_pred-{fileadd}"
}

# save data
for x in filenames:
    if x == 'model' or x == 'ytp':
        dump(results[x], filenames[x] + '.joblib')
    else:
        results[x].to_csv(filenames[x] + '.csv')

# log that the data is succesfully saved
logging.info("Data of (%s, %s) saved", m, t)


datetime_end = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
# log the moment of finishing
logging.info("Finished at: %s" % datetime_end)

# log the total running time
logging.info("Total time: %s" % (time.time() - starttime))

# clean up the cache of the pipe
for cache in cache_list:
    rmtree(cache)
