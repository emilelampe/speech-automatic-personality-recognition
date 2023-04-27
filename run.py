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
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report, confusion_matrix, roc_auc_score, roc_curve, balanced_accuracy_score
import pandas as pd
import json
import joblib
from shutil import rmtree
from tempfile import mkdtemp
import random
from random import randint
import maestros.functions as mt
from csv import writer

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
f = config.f
t = config.t
m = config.m
sc = config.sc
ec = config.ec
label_feature_indexes = config.label_feature_indexes
scoring = config.scoring
save_graphs = config.save_graphs
save_model = config.save_model
seed = config.seed
n_searches = config.n_searches
pre_pars = config.pre_pars
model_pars = config.model_pars
clf_rfecv = config.clf_rfecv
step_rfecv = config.step_rfecv
cal_method = config.cal_method

# --- ARGUMENTS SETTINGS ---

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR)

# Default random value for batch
random_id = randint(1, 10000)

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
parser.add_argument("-f", "--featureset", default=f,
                    help="The feature set to use")
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

# Overwrite config values with argument values
args = parser.parse_args()
profile = args.profile
m = args.model
t = args.trait
f = args.featureset
db = args.database
b = args.batchid
cal_method = args.cal_method
sc = args.startcutoff
ec = args.endcutoff
sc = float(sc)
ec = float(ec)
timestamp = timestamp

# Define index of labels and features after final database selection
begin_col_labels = label_feature_indexes[db][0]
begin_col_features = label_feature_indexes[db][1]

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
model_pars[m].update(pre_pars)
param_grid = model_pars[m]

# --- SETUP MULTIPROCESSING, LOGGING AND PATHS ---

file_prefix = f"{b}-{m}-{t}"

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
    if save_graphs:
        os.makedirs(f"{batch_path}/graphs")

logging.info("Folders for batch checked or made")

output_path = f"{batch_path}/outputs/{file_prefix}-output.txt"

# Print to both terminala and output file
def print_save(text):
    with open(output_path, "a") as file, contextlib.redirect_stdout(file):
        print(text)
    print(text)

# prepare the engines
c = Client(profile=profile)
# The following command will make sure that each engine is running in
# the right working directory to access the custom function(s).
c[:].map(os.chdir, [FILE_DIR]*len(c))

logging.info("c.ids :{0}".format(str(c.ids)))
bview = c.load_balanced_view()
register_parallel_backend('ipyparallel',
                          lambda: IPythonParallelBackend(view=bview))

print_save(f"\nSetup:")
print_save(f"b: {b}, db: {db}, f: {f}, m: {m}, t: {t}, scoring: {scoring}, cal: {cal_method}")

# --- LOAD DATA ---

# load dataset
# full_df = pd.read_pickle(f"{FILE_DIR}/data/{d}-{f}-{l}.pkl")
print(begin_col_features, begin_col_labels)
full_df = pd.read_pickle(f"{FILE_DIR}/data/{db}")

# adjust time cutoff
if sc != 0 and 'Length' in full_df.columns:
    full_df = full_df[(full_df['Length'] >= sc)]
if ec != 0 and 'Length' in full_df.columns:
    full_df = full_df[(full_df['Length'] <= ec)]

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
    X_train_val, X_test, y_train_vals, y_tests, train_val_indices, test_indices = mt.multilabel_stratified_group_split(X, y, groups, test_size=0.2, random_state=seed)

    groups_train_val = groups[train_val_indices]
    groups_test = groups[test_indices]

    # Split train_val into train and val for calibration (of whole dataset: 60% train, 20% validation)
    X_train, X_val, y_trains, y_vals, train_indices, val_indices = mt.multilabel_stratified_group_split(X_train_val, y_train_vals, groups_train_val, test_size=0.20, random_state=seed)

    groups_train = groups_train_val[train_indices]
    groups_val = groups_train_val[val_indices]

    print_save(mt.stratification_report(y, y_trains, y_tests, y_val=y_vals, labels=label_names))

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

print_save(f"Training {t}")
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
cv_results_ = merge_cv_results(reports)

# Get the best parameters
best_params_ = cv_results_.loc[cv_results_.index[0],'params']

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

print_save(f"\n{str(best_params_)}")

print_save("\nEvaluation Results:")

# Calculate the AUC-ROC score
auc_roc = round(roc_auc_score(y_test, y_probs_calibrated), 3)
print_save(f"AUC-ROC score: {auc_roc:.3f}")

# Calculate the balanced accuracy score
bal_acc = round(balanced_accuracy_score(y_test, y_preds_calibrated), 3)
print_save(f"Balanced accuracy score: {bal_acc:.3f}")

# Calculate the accuracy score
acc = round(accuracy_score(y_test, y_preds_calibrated), 3)
print_save(f"Accuracy score: {acc:.3f}")

# Calculate the Brier score
brier_score = round(brier_score_loss(y_test, y_probs_calibrated), 3)
print_save(f"Brier score: {brier_score:.3f}")

# Classification report
print_save(f"\nClassification report:\n{classification_report(y_test, y_preds_calibrated)}")

# Confusion matrix
print_save(f"Confusion matrix:\n{confusion_matrix(y_test, y_preds_calibrated)}")

db_name = db.split("-")[0]
# create the string to add to the main results file
main_results_string = [timestamp,b,db_name,sc,ec,f,m,t,auc_roc,bal_acc, acc, brier_score]
for i, x in enumerate(best_estimator_):
    if i == 1:
        if str(x) != 'passthrough':
            x = f"{str(x).split('=')[1][:-1]}, {x.n_components_}"
    if i == 2:
        x = x.n_features_
    main_results_string.append(str(x).replace('\n', ''))
main_results_string.append(cal_method)

# --- SAVE RESULTS ---

# Save cv_results_ to file
cv_results_.to_csv(f'{batch_path}/cv_results/{file_prefix}-best_result.csv')

if save_model:
    joblib.dump(best_estimator_, f"{batch_path}/best_estimators/{file_prefix}-best_estimator.joblib")


# write string to the main results file
with open(f"{FILE_DIR}/results/main_results.csv", 'a') as fw:
    w = writer(fw, delimiter=';')
    w.writerow(main_results_string)
    fw.close()

# Print total duration of execution
print_save(f"\nTotal time: {round((time.time() - starttime),1)}s")

if save_graphs:
    roc_name = f"{batch_path}/graphs/roc_curve-{file_prefix}.png"
    cal_name = f"{batch_path}/graphs/cal_curve-{file_prefix}.png"

    # Calculate the ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_probs_calibrated)

    # Plot the ROC curve
    plt.plot(fpr, tpr, label=f'AUC-ROC: {auc_roc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) curve')
    plt.savefig(os.path.join(roc_name), format='png')  # Save the plot to a file
    plt.clf()  # Clear the current plot

    # Calibration curve
    true_proportions, predicted_proportions = calibration_curve(y_test, y_probs_calibrated, n_bins=10)

    # Plot the calibration curve
    plt.plot(predicted_proportions, true_proportions, marker='o', label='Calibrated model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.title('Calibration curve')
    plt.legend()
    plt.savefig(os.path.join(cal_name), format='png')  # Save the plot to a file
    plt.clf()  # Clear the current plot