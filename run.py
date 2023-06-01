import argparse
import contextlib
from datetime import datetime
import configs.config as config
import configs.arguments_config as arguments_config
from apr.functions import *
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
from sklearn.base import clone
from sklearn.model_selection import permutation_test_score
from sklearn.metrics import accuracy_score, auc, brier_score_loss, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, balanced_accuracy_score
import pandas as pd
import json
import joblib
from shutil import rmtree
from tempfile import mkdtemp
import random
from random import randint
import maestros.functions as mt
from csv import writer
from sklearn.utils import resample
from scipy.stats import ttest_1samp

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
sc = config.sc
ec = config.ec
label_feature_indexes = config.label_feature_indexes
scoring = config.scoring
scoring_metrics = config.scoring_metrics
save_graphs = config.save_graphs
save_model = config.save_model
seed = config.seed
n_searches = config.n_searches
# pre_pars = config.pre_pars
model_pars = config.model_pars
clf_rfecv = config.clf_rfecv
step_rfecv = config.step_rfecv
calibration = config.calibration
cal_method = config.cal_method
n_bootstrap = config.n_bootstrap
n_metadata_cols = config.n_metadata_cols
gender = config.gender
pca = config.pca
w_rfecv = config.w_rfecv
leave_out_noisy = config.leave_out_noisy
leave_out_bfi10 = config.leave_out_bfi10

if leave_out_noisy:
    leave_out_noisy = '1'
else:
    leave_out_noisy = '0'

if leave_out_bfi10:
    leave_out_bfi10 = '1'
else:
    leave_out_bfi10 = '0'

if w_rfecv:
    w_rfecv = '1'
else:
    w_rfecv = '0'

# --- ARGUMENTS SETTINGS ---

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR)

# Default random value for batch
random_id = randint(1, 10000)
# Default value for run
run = 0

# Default current time for timestamp
starttime = time.time()
timestamp = datetime.now().strftime("%y%m%d_%H%M")

# prepare the logger
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--profile", default="ipy_profile",
                    help="Name of IPython profile to use")
parser.add_argument("-m", "--model", default=m,
                    help="The model with which you want to train")
parser.add_argument("-t", "--trait", default=t,
                    help="The trait on which you want to train")
parser.add_argument("-d", "--database", default=db, help="The database to train on")
parser.add_argument("-b", "--batchid", default=str(random_id), help="The ID with which different searches can be grouped")
parser.add_argument("--cal-method", default=cal_method,
                    help="The calibration method. Can be 'sigmoid' or 'isotonic'")
parser.add_argument("-s", "--startcutoff", default=str(sc),
                    help="The minimum length of a sample")
parser.add_argument("-e", "--endcutoff", default=str(ec),
                    help="The maximum length of a sample")
parser.add_argument("--timestamp", default=timestamp,
                    help="Timestamp in format YYMMDD_hhmm, used for logging batches")
parser.add_argument("--scoring", default=scoring,
                    help="Scoring metric for training")
parser.add_argument("--run", default=run,
                    help="Scoring metric for training")
parser.add_argument("--pca", default=str(pca),
                    help="PCA to use")
parser.add_argument("--leave-bfi10", default=leave_out_bfi10,
                    help="Whether to leave out BFI-10 for NSC")
parser.add_argument("--leave-noisy", default=leave_out_noisy,
                    help="Whether to leave out very noisy for REMDE")
parser.add_argument("--rfecv", default=w_rfecv,
                    help="With RFECV or not")

# Overwrite config values with argument values
args = parser.parse_args()
profile = args.profile
m = args.model
t = args.trait
db = args.database
b = args.batchid
cal_method = args.cal_method
sc = args.startcutoff
ec = args.endcutoff
sc = float(sc)
ec = float(ec)
timestamp = args.timestamp
scoring = args.scoring
run = args.run
pca = str(args.pca)
w_rfecv = int(args.rfecv)
leave_out_bfi10 = int(args.leave_bfi10)
leave_out_noisy = int(args.leave_noisy)

if w_rfecv == 1:
    w_rfecv = True
else:
    w_rfecv = False

if leave_out_bfi10 == 1:
    leave_out_bfi10 = True
else:
    leave_out_bfi10 = False

if leave_out_noisy == 1:
    leave_out_noisy = True
else:   
    leave_out_noisy = False

if pca == '99':
    pca = PCA(0.99)
    pca_str = '99'
elif pca == '95':
    pca = PCA(0.95)
    pca_str = '95'
else:
    pca = 'passthrough'
    pca_str = 'NaN'

# Define index of labels and features after final database selection
begin_col_labels = label_feature_indexes[db][0]
begin_col_features = label_feature_indexes[db][1]
median_labels_needed = label_feature_indexes[db][2]
db_is_remde = label_feature_indexes[db][3]
db_is_nsc = label_feature_indexes[db][4]
f = label_feature_indexes[db][5]

remde_str = 'NaN'
if db_is_remde:
    if leave_out_noisy:
        remde_str = 'w/o'
    else:
        remde_str = 'w'

nsc_str = 'NaN'
if db_is_nsc:
    if leave_out_bfi10:
        nsc_str = 'w/o'
    else:
        nsc_str = 'w'

# Import custom arguments config
trait_dict = arguments_config.trait_dict
feat_dict = arguments_config.feat_dict
database_dict = arguments_config.database_dict

if t.capitalize() in trait_dict.keys():
    t = trait_dict[t.capitalize()]

if f.lower() in feat_dict.keys():
    f = feat_dict[f.lower()]

if db.lower() in database_dict.keys():
    db = database_dict[db.lower()]

# combine model and preprocessing parameters into one parameter grid
# model_pars[m].update(pre_pars)
param_grid = model_pars[m]

# --- SETUP MULTIPROCESSING, LOGGING AND PATHS ---

if calibration:
    cal_str = cal_method
else:
    cal_str = "no_cal"

# file_prefix = f"{timestamp}-{b}-{db}-{scoring}-{sc}-{ec}-{m}-{f}-{t}-{cal_str}".replace(".","_")
file_prefix = f"{b}-{run}"


logfilename = os.path.join(
    FILE_DIR, f'log/{timestamp}_{b}.log')
logging.basicConfig(filename=logfilename,
                    filemode='w',
                    level=logging.DEBUG)
logging.info("number of CPUs found: {0}".format(cpu_count()))
logging.info("args.profile: {0}".format(profile))
logging.info("Batch: %s" % b)
logging.info("FILE_DIR: %s" % FILE_DIR)

# check whether the batchid folder exists
batch_path = f"{FILE_DIR}/results/{timestamp}_{b}"
isExist = os.path.exists(batch_path)
# if it doesn't exist, create new folder
if not isExist:
    os.makedirs(batch_path)
    if save_model:
        os.makedirs(f"{batch_path}/best_estimators")
    os.makedirs(f"{batch_path}/cv_results")
    os.makedirs(f"{batch_path}/outputs")
    os.makedirs(f"{batch_path}/boots")
    if save_graphs:
        os.makedirs(f"{batch_path}/graphs")

logging.info("Folders for batch checked or made")

output_path = f"{batch_path}/outputs/{file_prefix}-output.txt"

ps = PrintSaver(output_path)

# prepare the engines
c = Client(profile=profile)
# The following command will make sure that each engine is running in
# the right working directory to access the custom function(s).
c[:].map(os.chdir, [FILE_DIR]*len(c))

logging.info("c.ids :{0}".format(str(c.ids)))
bview = c.load_balanced_view()
register_parallel_backend('ipyparallel',
                          lambda: IPythonParallelBackend(view=bview))

ps.print_save(f"\nSetup:")
ps.print_save(f"b: {b}, db: {db}, remde_noise: {remde_str}, nsc_bfi10: {nsc_str}, f: {f}, m: {m}, t: {t}, scoring: {scoring}, cal: {cal_str}")

# --- LOAD DATA ---

# load dataset
full_df = pd.read_pickle(f"{FILE_DIR}/data/{db}")

if db_is_remde:
    if leave_out_noisy:
        users = pd.read_csv('data/remde_noisy_users.csv')
        noisy_users = users[users['Usable'] == 0]['Group']
        full_df = full_df[full_df['Group'].isin(noisy_users) == False]

if db_is_nsc:
    if leave_out_bfi10:
        full_df = full_df[full_df['Group'] != 'BFI-10']

if gender == 'male':
    full_df = full_df[full_df['Gender'] == 1]
elif gender == 'female':
    full_df = full_df[full_df['Gender'] == 0]
else:
    gender = 'both'

# adjust time cutoff
if sc != 0 and 'Length' in full_df.columns:
    full_df = full_df[(full_df['Length'] >= sc)]
if ec != 0 and 'Length' in full_df.columns:
    full_df = full_df[(full_df['Length'] <= ec)]

if median_labels_needed:
    'Calculating median labels...'
    full_df = calculate_median_labels(full_df, begin_col_labels, begin_col_features)

# define X and y
X = full_df.iloc[:, begin_col_features:]
y = full_df.iloc[:,begin_col_labels - n_metadata_cols:begin_col_features]

# define feature_names, label_names and groups
feature_names = list(X.columns.values)
label_names = list(y.columns.values)
groups = full_df['Group'].to_numpy()


# convert to numpy
X = X.to_numpy()
y = y.to_numpy()

ps.print_save(f"\nTotal number of samples: {len(y[:,0])}")

# --- SPLIT TRAIN, VAL, TEST ---

# Number of labels
y_n_cols = y.shape[n_metadata_cols]

if y_n_cols > 1:
    # -- Multi label split --
    # Choose the label to train on
    t_idx = label_names.index(t)

    # If with calibration, create validation set for calibration
    if calibration:
        # Initial train_val-test split (80% train_val, 20% test)
        X_train_val, X_test, y_train_vals, y_tests, train_val_indices, test_indices = mt.multilabel_stratified_group_split(X, y, groups, test_size=0.2, random_state=seed)

        groups_train_val = groups[train_val_indices]
        groups_test = groups[test_indices]

        
        # Split train_val into train and val for calibration (of whole dataset: 60% train, 20% validation)
        X_train, X_val, y_trains, y_vals, train_indices, val_indices = mt.multilabel_stratified_group_split(X_train_val, y_train_vals, groups_train_val, test_size=0.20, random_state=seed)

        groups_train = groups_train_val[train_indices]
        groups_val = groups_train_val[val_indices]

        ps.print_save(mt.stratification_report(y, y_trains, y_tests, y_val=y_vals, labels=label_names))

        y_val = y_vals[:,t_idx]
    else:
        # Initial train-test split (80% train, 20% test)
        X_train, X_test, y_trains, y_tests, train_indices, test_indices = mt.multilabel_stratified_group_split(X, y, groups, test_size=0.2, random_state=seed)

        groups_train = groups[train_indices]
        groups_test = groups[test_indices]

        ps.print_save(mt.stratification_report(y, y_trains, y_tests, labels=label_names))

    
    y_train = y_trains[:,t_idx]
    y_test = y_tests[:,t_idx]
else:
    # -- Single label split --

    # Convert single column y to 1D array
    y = y.ravel()

    if calibration:
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
    else:
        # Initial train-test split (80% train, 20% test)
        cv_split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        train_idx, test_idx = next(cv_split.split(X, y, groups))
        X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]

# --- TRAINING ---

# Variables to keep track of the best model and score
reports = [None] * n_searches

# Manually iterate over the outer StratifiedGroupKFold for GridSearchCV
cv_gs = StratifiedGroupKFold(n_splits=n_searches, shuffle=True, random_state=seed)

ps.print_save(f"Training {t}")
# -- cross-validation of the GridSearch --
for idx, (train_idx_gs, test_idx_gs) in enumerate(StratifiedGroupKFold(n_splits=n_searches, shuffle=True, random_state=seed).split(X_train, y_train, groups_train)):
    print(f"Fold {idx+1}")
    X_train_gs, y_train_gs, X_test_gs, y_test_gs = X_train[train_idx_gs], y_train[train_idx_gs], X_train[test_idx_gs], y_train[test_idx_gs]
    groups_train_gs = groups_train[train_idx_gs]
    groups_test_gs =  groups_train[test_idx_gs]

    # Create a StratifiedGroupKFold object for the RFECV
    cv_rfecv = BalancedStratifiedGroupKFold(n_splits=5, max_attempts=1000, print_saver=ps, outer_fold=(idx+1))

    test_fold = np.zeros(len(y_train))
    test_fold[train_idx_gs] = -1

    cv_fold = PredefinedSplit(test_fold=test_fold)

    cache = mkdtemp()
    
    if w_rfecv:
        # Create the pipeline
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', pca),
            ('feat_sel', RFECV(clf_rfecv, cv=list(cv_rfecv.split(X_train_gs, y_train_gs, groups_train_gs)), scoring=scoring, step=step_rfecv)),
            ('clf', SVC(kernel='rbf'))
        ],memory=cache)
    else:
         # Create the pipeline
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', pca),
            ('feat_sel', 'passthrough'),
            ('clf', SVC(kernel='rbf'))
        ],memory=cache)

    # Perform a Grid Search for the current fold
    gs = GridSearchCV(pipe, param_grid, scoring=scoring_metrics, n_jobs=len(c), cv=cv_fold, error_score='raise', verbose=1, refit=scoring)

    # run with multiprocessing
    with parallel_backend('ipyparallel'):
        gs.fit(X_train, y_train)

    reports[idx] = pd.DataFrame(gs.cv_results_)

    rmtree(cache)

scoring_names = scoring_metrics.keys()
# Combine the cv_results_ of the different folds
cv_results_ = merge_cv_results(reports, scoring_metrics=scoring_names, main_metric=scoring)

# Get the best parameters
best_params_ = cv_results_.loc[cv_results_.index[0],'params']

for x in scoring_names:
    if x != scoring:
        second_scoring = x

cv_combined_test_score = cv_results_.loc[cv_results_.index[0],f'combined_test_{scoring}']
cv_mean_test_main = cv_results_.loc[cv_results_.index[0],f'mean_test_{scoring}']
cv_std_test_main = cv_results_.loc[cv_results_.index[0],f'std_test_{scoring}']
cv_mean_test_second = cv_results_.loc[cv_results_.index[0],f'mean_test_{second_scoring}']
cv_std_test_second = cv_results_.loc[cv_results_.index[0],f'std_test_{second_scoring}']
cv_rank_test_score = cv_results_.loc[cv_results_.index[0],f'rank_test_score']


# create a StratifiedGroupKFold object for the best estimator RFECV
cv_rfecv_best = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

if w_rfecv:
    # Create the best estimator pipeline
    best_estimator_ = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', pca),
        ('feat_sel', RFECV(clf_rfecv, cv=list(cv_rfecv_best.split(X_train_gs, y_train_gs, groups_train_gs)), scoring=scoring)),
        ('clf', 'passthrough')
    ])
else:
    # Create the best estimator pipeline
    best_estimator_ = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', pca),
        ('feat_sel', 'passthrough'),
        ('clf', 'passthrough')
    ])

# Set the best parameters
best_estimator_.set_params(**best_params_)

# Fit the best estimator on the whole training set
best_estimator_.fit(X_train, y_train)

final_estimator_ = best_estimator_

if calibration:
    # Calibrate the model using the validation set
    calibrated_estimator_ = CalibratedClassifierCV(best_estimator_, cv='prefit', method=cal_method)
    calibrated_estimator_.fit(X_val, y_val)
    final_estimator_ = calibrated_estimator_

# --- EVALUATION ---

ps.print_save(f"\n{str(best_params_)}")

# Evaluation test set using bootstrapping

# Initialize arrays to store bootstrapped metrics
boot_auc_rocs = []
boot_bal_accs = []
boot_f1_scores = []
boot_precisions = []
boot_recalls = []

# Bootstrap loop
print("\nBootstrapping...")
for bs_idx in range(n_bootstrap):
    if (bs_idx+1) % (n_bootstrap / 10) == 0:
        print(f"Bootstrap {bs_idx+1}/{n_bootstrap}")
    # Resample the test set with replacement
    X_resampled, y_resampled = resample(X_test, y_test, replace=True, random_state=bs_idx, n_samples=len(y_test))

    # Predict probabilities and labels for the resampled test set
    y_probs = final_estimator_.predict_proba(X_resampled)[:, 1]
    y_preds = final_estimator_.predict(X_resampled)

    # Compute the evaluation metrics
    fpr, tpr, _ = roc_curve(y_resampled, y_probs)
    boot_auc_rocs.append(roc_auc_score(y_resampled, y_probs))
    boot_bal_accs.append(balanced_accuracy_score(y_resampled, y_preds))
    boot_f1_scores.append(f1_score(y_resampled, y_preds))
    boot_precisions.append(precision_score(y_resampled, y_preds))
    boot_recalls.append(recall_score(y_resampled, y_preds))

print("Bootstrapping done!")

# Calculate the mean and standard deviation for each metric
auc_roc_mean = round(np.mean(boot_auc_rocs), 3)
auc_roc_std = round(np.std(boot_auc_rocs), 3)
bal_acc_mean = round(np.mean(boot_bal_accs), 3)
bal_acc_std = round(np.std(boot_bal_accs), 3)
f1_mean = round(np.mean(boot_f1_scores), 3)
f1_std = round(np.std(boot_f1_scores), 3)
precision_mean = round(np.mean(boot_precisions), 3)
precision_std = round(np.std(boot_precisions), 3)
recall_mean = round(np.mean(boot_recalls), 3)
recall_std = round(np.std(boot_recalls), 3)

# Calculate the 95% confidence intervals for each metric
alpha = 0.95
auc_roc_ci = np.percentile(boot_auc_rocs, [(1 - alpha) / 2 * 100, (1 + alpha) / 2 * 100])
bal_acc_ci = np.percentile(boot_bal_accs, [(1 - alpha) / 2 * 100, (1 + alpha) / 2 * 100])
f1_ci = np.percentile(boot_f1_scores, [(1 - alpha) / 2 * 100, (1 + alpha) / 2 * 100])
precision_ci = np.percentile(boot_precisions, [(1 - alpha) / 2 * 100, (1 + alpha) / 2 * 100])
recall_ci = np.percentile(boot_recalls, [(1 - alpha) / 2 * 100, (1 + alpha) / 2 * 100])


# Predict probabilities for the original test set
y_probs = final_estimator_.predict_proba(X_test)[:, 1]
y_preds = final_estimator_.predict(X_test)

# Scoring metrics of original test set
auc_roc = roc_auc_score(y_test, y_probs)
bal_acc = balanced_accuracy_score(y_test, y_preds)
f1 = f1_score(y_test, y_preds)
precision = precision_score(y_test, y_preds)
recall = recall_score(y_test, y_preds)

# Classification report of original tes tset
ps.print_save(f"\nClassification report original test set:\n{classification_report(y_test, y_preds)}")

# Confusion matrix of original test set
ps.print_save(f"Confusion matrix original test set:\n{confusion_matrix(y_test, y_preds)}")

ps.print_save("\nTraining set CV best model:")
ps.print_save(f"Best combined score: {round(cv_combined_test_score, 3)}")
ps.print_save(f"Best mean score: {round(cv_mean_test_main, 3)}")
ps.print_save(f"Best std score: {round(cv_std_test_main, 3)}")
ps.print_save(f"Rank of mean score: {cv_rank_test_score}")

db_name = db.split("-")[0]
# create the string to add to the main results file
main_results_string = [timestamp, b, run, db_name, remde_str, nsc_str, f, m, t]

if best_estimator_[1] != 'passthrough':
    main_results_string.append(f"{pca_str}, {best_estimator_[1].n_features_}")
else:
    main_results_string.append('NaN')

if best_estimator_[2] != 'passthrough':
    main_results_string.append(best_estimator_[2].n_features_)
else:
    main_results_string.append('NaN')
main_results_string.append(str(best_estimator_[-1]).replace('\n', ''))

main_results_string.extend([
    scoring,
    round(cv_combined_test_score, 3), cv_rank_test_score, round(cv_mean_test_main, 3), round(cv_std_test_main, 3),
])

# Order of metrics: original value, mean, std, p-value, ci[0], ci[1]
t_stat_bal_acc, p_value_bal_acc = ttest_1samp(boot_bal_accs, 0.5)
main_results_string.extend([
    round(bal_acc, 3), round(bal_acc_mean, 3), round(bal_acc_std, 3), round(p_value_bal_acc,3), round(bal_acc_ci[0], 3), round(bal_acc_ci[1], 3),
])

t_stat_auc_roc, p_value_auc_roc = ttest_1samp(boot_auc_rocs, 0.5)
main_results_string.extend([
    round(auc_roc, 3), round(auc_roc_mean, 3), round(auc_roc_std, 3), round(p_value_auc_roc, 3), round(auc_roc_ci[0], 3), round(auc_roc_ci[1], 3),
])

t_stat_f1, p_value_f1 = ttest_1samp(boot_f1_scores, 0.5)
main_results_string.extend([
    round(f1, 3), round(f1_mean, 3), round(f1_std, 3), round(p_value_f1,3), round(f1_ci[0], 3), round(f1_ci[1], 3),
])

t_stat_precision, p_value_precision = ttest_1samp(boot_precisions, 0.5)
main_results_string.extend([
    round(precision, 3), round(precision_mean, 3), round(precision_std, 3), round(p_value_precision, 3), round(precision_ci[0], 3), round(precision_ci[1], 3),
])

t_stat_recall, p_value_recall = ttest_1samp(boot_recalls, 0.5)
main_results_string.extend([
    round(recall, 3), round(recall_mean, 3), round(recall_std, 3), round(p_value_recall,3), round(recall_ci[0], 3), round(recall_ci[1], 3),
])


# Print bootstrapped results
ps.print_save("\nEvaluation bootstrap results (mean, SD, p-value, lower CI, higher CI):")

ps.print_save(f"Balanced accuracy: ({bal_acc_mean:.3f}, {bal_acc_std:.3f}, {p_value_bal_acc},{bal_acc_ci[0]:.3f}, {bal_acc_ci[1]:.3f})")
ps.print_save(f"AUC-ROC score: ({auc_roc_mean:.3f}, {auc_roc_std:.3f}, {p_value_auc_roc}, {auc_roc_ci[0]:.3f}, {auc_roc_ci[1]:.3f})")
# ps.print_save(f"F1 score: ({f1_mean:.3f}, {f1_std:.3f}, {p_v},{f1_ci[0]:.3f}, {f1_ci[1]:.3f})")
# ps.print_save(f"Precision: ({precision_mean:.3f}, {precision_std:.3f}, {precision_ci[0]:.3f}, {precision_ci[1]:.3f})")
# ps.print_save(f"Recall: ({recall_mean:.3f}, {recall_std:.3f}, {recall_ci[0]:.3f}, {recall_ci[1]:.3f})")

# --- SAVE RESULTS ---

# Save cv_results_ to file
cv_results_.to_csv(f'{batch_path}/cv_results/{file_prefix}-best_result.csv')

if save_model:
    joblib.dump(best_estimator_, f"{batch_path}/best_estimators/{file_prefix}-best_estimator.joblib")

joblib.dump(boot_bal_accs, f"{batch_path}/boots/{file_prefix}-boots.joblib")


# write string to the main results file
with open(f"{FILE_DIR}/results/main_results.csv", 'a') as fw:
    w = writer(fw, delimiter=';')
    w.writerow(main_results_string)
    fw.close()

print("\nFiles saved.")

# Print total duration of execution
ps.print_save(f"\nTotal time: {round((time.time() - starttime),1)}s")

if save_graphs:
    roc_name = f"{batch_path}/graphs/roc_curve-{file_prefix}.png"
    # Calculate the ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)

    # Plot the ROC curve
    plt.plot(fpr, tpr, label=f'AUC-ROC: {auc_roc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) curve')
    plt.savefig(os.path.join(roc_name), format='png')  # Save the plot to a file
    plt.clf()  # Clear the current plot

    # Calibration curve
    cal_name = f"{batch_path}/graphs/cal_curve-{file_prefix}.png"
    true_proportions, predicted_proportions = calibration_curve(y_test, y_probs, n_bins=10)

    # Plot the calibration curve
    plt.plot(predicted_proportions, true_proportions, marker='o', label='Calibrated model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.title('Calibration curve')
    plt.legend()
    plt.savefig(os.path.join(cal_name), format='png')  # Save the plot to a file
    plt.clf()  # Clear the current plot