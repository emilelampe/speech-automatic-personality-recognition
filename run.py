from datetime import datetime
import config
import apr.functions as functions
import apr.multilabel_stratified_group_split as msgs
import logging
import os
import sys
import time
from matplotlib import pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedGroupKFold, PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report, confusion_matrix, roc_auc_score, roc_curve, balanced_accuracy_score
import pandas as pd
import json
import joblib
from shutil import rmtree
from tempfile import mkdtemp
import random

# parallel processing imports
sys.modules['sklearn.externals.joblib'] = joblib
from sklearn.externals.joblib import Parallel, parallel_backend
from sklearn.externals.joblib import register_parallel_backend
from sklearn.externals.joblib import delayed
from sklearn.externals.joblib import cpu_count
from ipyparallel import Client
from ipyparallel.joblib import IPythonParallelBackend

# --- IMPORT CONFIG ---

db = config.db
t = config.t
m = config.m
begin_col_labels = config.begin_col_labels
begin_col_features = config.begin_col_features
sc = config.sc
ec = config.ec
scoring = config.scoring
seed = config.seed
n_searches = config.n_searches
pre_pars = config.pre_pars
model_pars = config.model_pars
clf_rfecv = config.clf_rfecv
step_rfecv = config.step_rfecv
cal_method = config.cal_method

# combine model and preprocessing parameters into one parameter grid
model_pars[m].update(pre_pars)
param_grid = model_pars[m]

# --- SETUP MULTIPROCESSING, LOGGING AND PATHS ---

b = random.randint(0, 10000)

profile = "ipy_profile"

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR)

starttime = time.time()
datetime_start = datetime.now().strftime("%y%m%d_%H%M")

file_prefix = f"{b}-{m}-{t}-"

logfilename = os.path.join(
    FILE_DIR, f'log/{datetime_start}_{b}.log')
logging.basicConfig(filename=logfilename,
                    filemode='w',
                    level=logging.DEBUG)
logging.info("number of CPUs found: {0}".format(cpu_count()))
logging.info("args.profile: {0}".format(profile))
logging.info("Batch: %s" % b)
logging.info("FILE_DIR: %s" % FILE_DIR)

# check whether the batchid folder exists
batch_path = f"{FILE_DIR}/results/{datetime_start}_{b}"
isExist = os.path.exists(batch_path)
# if it doesn't exist, create new folder
if not isExist:
   os.makedirs(batch_path)
   os.makedirs(f"{batch_path}/best_estimators")
#    os.makedirs(f"{batch_path}/confusion_matrices")
   os.makedirs(f"{batch_path}/cv_results")
#    os.makedirs(f"{batch_path}/results_reports")
#    os.makedirs(f"{batch_path}/y_test_predictions")

logging.info("Folders for batch checked or made")

# prepare the engines
c = Client(profile=profile)
# The following command will make sure that each engine is running in
# the right working directory to access the custom function(s).
c[:].map(os.chdir, [FILE_DIR]*len(c))

logging.info("c.ids :{0}".format(str(c.ids)))
bview = c.load_balanced_view()
register_parallel_backend('ipyparallel',
                          lambda: IPythonParallelBackend(view=bview))

# --- LOAD DATA ---

# load dataset
# full_df = pd.read_pickle(f"{FILE_DIR}/data/{d}-{f}-{l}.pkl")
full_df = pd.read_pickle(f"{FILE_DIR}/data/{db}")

# adjust time cutoff
if sc != 0 and 'Length' in full_df.columns:
    full_df = full_df[(full_df['Length'] >= sc * 1000)]
if ec != 0 and 'Length' in full_df.columns:
    full_df = full_df[(full_df['Length'] <= ec * 1000)]

# define X and y
X = full_df.iloc[:, begin_col_features:]
y = full_df.iloc[:,begin_col_labels:begin_col_features]

# define feature_names, label_names and groups
feature_names = list(X.columns.values)
label_names = list(y.columns.values)
groups = full_df['Group'].to_numpy()

# convert to numpy
X = X.to_numpy()
y = y.to_numpy()

# --- SPLIT TRAIN, VAL, TEST ---

# Number of labels
y_n_cols = y.shape[1]

if y_n_cols > 1:
    # -- Multi label split --

    # Initial train_val-test split (80% train_val, 20% test)
    X_train_val, X_test, y_train_vals, y_tests, train_val_indices, test_indices = msgs.multilabel_stratified_group_train_test_split(X, y, groups, test_size=0.2, random_state=seed)

    groups_train_val = groups[train_val_indices]
    groups_test = groups[test_indices]

    # Split train_val into train and val for calibration (of whole dataset: 60% train, 20% validation)
    X_train, X_val, y_trains, y_vals, train_indices, val_indices = msgs.multilabel_stratified_group_train_test_split(X_train_val, y_train_vals, groups_train_val, test_size=0.20, random_state=seed)

    groups_train = groups_train_val[train_indices]
    groups_val = groups_train_val[val_indices]

    msgs.check_disjoint_groups(train_val_indices=train_val_indices, test_indices=test_indices, train_indices=train_indices, val_indices=val_indices, groups=groups)

    msgs.stratification_report(y, y_trains, y_vals, y_tests, label_names)

    # Choose the label to train on
    t_idx = label_names.index(t)
    y_train = y_trains[:,t_idx]
    y_val = y_vals[:,t_idx]
    y_test = y_tests[:,t_idx]
else:
    # -- Single label split --

    # Convert single column y to 1D array
    y = y.ravel()

    # Initial train_val-test split (80% train_val, 20% test)
    cv_split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(cv_split.split(X, y, groups))
    X_train_val, X_test, y_train_val, y_test = X[train_val_idx], X[test_idx], y[train_val_idx], y[test_idx]
    groups_train_val, groups_test = groups[train_val_idx], groups[test_idx]

    # Split train_val into train and val for calibration (of whole dataset: 60% train, 20% validation)
    cv_train_val = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    train_idx, val_idx = next(cv_train_val.split(X_train_val, y_train_val, groups_train_val))
    X_train, X_val, y_train, y_val = X_train_val[train_idx], X_train_val[val_idx], y_train_val[train_idx], y_train_val[val_idx]
    groups_train, groups_val = groups_train_val[train_idx], groups_train_val[val_idx]

# --- TRAINING ---

# Variables to keep track of the best model and score
reports = [None] * n_searches

# Manually iterate over the outer StratifiedGroupKFold for GridSearchCV
cv_gs = StratifiedGroupKFold(n_splits=n_searches, shuffle=True, random_state=seed)

# -- cross-validation of the GridSearch --
for idx, (train_idx_gs, test_idx_gs) in enumerate(cv_gs.split(X_train, y_train, groups_train)):
    X_train_gs, y_train_gs, X_test_gs, y_test_gs = X_train[train_idx_gs], y_train[train_idx_gs], X_train[test_idx_gs], y_train[test_idx_gs]
    groups_train_gs = groups_train[train_idx_gs]
    groups_test_gs =  groups_train[test_idx_gs]

    # Create a StratifiedGroupKFold object for the RFECV
    cv_rfecv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    cache = mkdtemp()

    # Create the pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', 'passthrough'),
        ('feat_sel', RFECV(clf_rfecv, cv=list(cv_rfecv.split(X_train_gs, y_train_gs, groups_train_gs)), scoring=scoring, step=step_rfecv)),
        ('clf', SVC(kernel='rbf'))
    ],memory=cache)

    test_fold = np.zeros(len(y_train))
    test_fold[train_idx_gs] = -1

    cv_fold = PredefinedSplit(test_fold=test_fold)

    print("Fold ", (idx + 1))
    # Perform a Grid Search for the current fold
    gs = GridSearchCV(pipe, param_grid, scoring=scoring, n_jobs=len(c), cv=cv_fold, error_score='raise', verbose=1)

    # run with multiprocessing
    with parallel_backend('ipyparallel'):
        gs.fit(X_train, y_train)

    reports[idx] = pd.DataFrame(gs.cv_results_)

    rmtree(cache)

# Combine the cv_results_ of the different folds
cv_results_ = functions.merge_cv_results(reports)

# Save cv_results_ to file
cv_results_.to_csv(f'{batch_path}/{file_prefix}best_result.csv')

# Get the best parameters
best_params_ = cv_results_.loc[cv_results_.index[0],'params']
best_params_ = json.loads(best_params_.replace("'", '"'))
best_params_

# create a StratifiedGroupKFold object for the best estimator RFECV
cv_rfecv_best = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

# Create the best estimator pipeline
best_estimator_ = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=None)),
    ('feat_sel', RFECV(clf_rfecv, cv=list(cv_rfecv_best.split(X_train_gs, y_train_gs, groups_train_gs)), scoring=scoring)),
    ('clf', 'passthrough')
])

# Set the best parameters
best_estimator_.set_params(**best_params_)

# Fit the best estimator on the whole training set
best_estimator_.fit(X_train, y_train)

# Calibrate the model using the validation set
calibrated_estimator_ = CalibratedClassifierCV(best_estimator_, cv='prefit', method=cal_method)
calibrated_estimator_.fit(X_val, y_val)

# --- EVALUATION ---

# Predict probabilities for the test set using the calibrated model
y_probs_calibrated = calibrated_estimator_.predict_proba(X_test)[:, 1]
y_preds_calibrated = calibrated_estimator_.predict(X_test)

# assert(list(y_preds_calibrated) == list(np.argmax(y_probs_calibrated, axis=1)))

print(f"\n{str(best_params_)}")

print("\nEvaluation Results:")

# Calculate the AUC-ROC score
auc_roc = roc_auc_score(y_test, y_probs_calibrated)
print(f"AUC-ROC score: {auc_roc:.3f}")

# Calculate the balanced accuracy score
bal_acc = balanced_accuracy_score(y_test, y_preds_calibrated)
print(f"Balanced accuracy score: {bal_acc:.3f}")

# Calculate the Brier score
brier_score = brier_score_loss(y_test, y_probs_calibrated)
print(f"Brier score: {brier_score:.3f}")

# Classification report

print("\nClassification report:\n", classification_report(y_test, y_preds_calibrated))

# Confusion matrix
print("Confusion matrix:\n", confusion_matrix(y_test, y_preds_calibrated))

print("\nTotal time:", (time.time() - starttime))

# Calculate the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_probs_calibrated)

# # Plot the ROC curve
# plt.plot(fpr, tpr, label=f'AUC-ROC: {auc_roc:.3f}')
# plt.plot([0, 1], [0, 1], linestyle='--', label='Random classifier')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) curve')
# plt.legend()
# plt.show()

# # Calibration curve
# true_proportions, predicted_proportions = calibration_curve(y_test, y_probs_calibrated, n_bins=10)

# # Plot the calibration curve
# plt.plot(predicted_proportions, true_proportions, marker='o', label='Calibrated SVM')
# plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
# plt.xlabel('Predicted probability')
# plt.ylabel('True probability')
# plt.title('Calibration curve')
# plt.legend()
# plt.show()