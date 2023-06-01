import numpy as np
from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, make_scorer, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

# ------------
# ---CONFIG---
# ------------

'''The config is divided in 2 sections:
    - The first section contains specific parameters such as the dataset filename
    - The second section contains the parameter grid for the grid search
    
    Config values will be overwritten by command line arguments if they are given.'''

# database to use
# For single label example dataset: 'single_label_example_dataset.pkl' (begin_col_labels = 2, begin_col_features = 3)
# Example from https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data
db = "spc-egemaps.pkl"
# db = "nsc-egemaps.pkl"

# Info on the datasets:
# - index of column where labels start
# - index of column where features start
# - whether the labels are raw personality scores and need to be binarized
# - whether the dataset is REMDE or not
# - whether the dataset is NSC or not
label_feature_indexes = {
    'single_label_example_dataset.pkl': (2, 3, False, False, False, 'egemaps'),
    'spc-egemaps.pkl': (3, 8, False, False, False, 'egemaps'),
    'spc-embeddings.pkl': (3, 8, False, False, False, 'embeddings'),
    'nsc-egemaps.pkl': (5, 10, False, False, True, 'egemaps'),
    # 'own-egemaps.pkl': (4, 9, False, True, False),
    # 'own_combined-egemaps.pkl': (4, 9, False, True, False),
    # 'own_combined_new-egemaps.pkl': (4, 9, False, True, False),
    # 'own_combined_raw-egemaps.pkl': (4, 9, True, True, False),
    # 'own_1015-egemaps.pkl': (4, 9, True, True, False),
    'reduced-egemaps.pkl': (4, 9, True, True, False, 'egemaps'),
    'noisy-egemaps.pkl': (4, 9, True, True, False, 'egemaps')
}

# For REMDE db: whether to leave out the speakers that were noisy by ear
# Is only applicable to datasets that are set as REMDE above
leave_out_noisy = True

# For NSC db: whether to leave out the BFI-10 speakers
leave_out_bfi10 = True

# Columns to include with stratified train/test split, but delete afterwards
# For example only 'Gender' column, set n_metadata_cols to 1
# Metadata columns must be place directly before the label columns
n_metadata_cols = 1

# Choose to only include one gender in the data ('both', 'male', 'female')
# Gender column must be called 'Gender', 'male' = 1, 'female' = 0
gender = 'both'

# Model to use ('svm_rbf', 'knn', 'rf', 'svm_l')
# m = 'svm_rbf'
m = 'svm_rbf'

# Label column to choose in case of multi-label dataset
t = 'Extraversion'

# minimum and maximum length cutoff in seconds ('Length' column needed)
sc = 0
ec = 0

# main scoring metric ('balanced_accuracy', 'roc_auc', 'f1')
scoring = 'balanced_accuracy'

# all scoring metrics
scoring_metrics = {
    'balanced_accuracy': make_scorer(balanced_accuracy_score),
    'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
}

# if you want to save graphs
save_graphs = True

# if you want to save the trained model
save_model = True

# whether to calibrate the model
calibration = False

# random seed
seed = 42

# number of bootstrap samples
n_bootstrap = 100

# number of GridSearch folds
n_searches = 5

# RFECV base estimator
clf_rfecv = DecisionTreeClassifier(random_state=seed)

# Calibration method
cal_method = 'no_cal'

# PCA method ('passthrough' for no PCA, '95' for 95% variability, '0.99' for 99% variability)
pca = 'passthrough'

w_rfecv = False

# # step size for RFECV
step_rfecv = 1

# --- PARAMETER GRID ---

# Range of C parameter
C_range = np.logspace(-3, 6, 10)

# Range of gamma parameter
gamma_range = np.logspace(-7, 2, 10)

# Range of RF max depth
max_depth_range = np.logspace(1, 4, 4, base=2)
max_depth_range = list(max_depth_range)
max_depth_range = [int(x) for x in max_depth_range]
max_depth_range.append(None)

# Preprocessing parameters
# pre_pars = {
#         'scaler': [StandardScaler()],
#         # 'pca': [PCA(0.95), PCA(0.99), 'passthrough'],
#         'pca': ['passthrough']
#     }

# Model parameters
model_pars = {
    # SVM with linear kernel
    'svm_l': {
        'clf': [LinearSVC(random_state=seed)],
        'clf__C': C_range
    },
    # # SVM with RBF kernel
    'svm_rbf': {
        'clf': [SVC(random_state=seed, probability=True)],
        'clf__kernel': ['rbf'],
        # 'clf__C': C_range,
        'clf__C': [10, 100],
        # 'clf__gamma': gamma_range,
        'clf__gamma': [0.01, 0.001]
    },
    # Random Forest
    'rf':
    {
        'clf': [RandomForestClassifier(random_state=seed, n_jobs=1)],
        # 'clf__n_estimators': [10, 100, 250, 500, 1000],
        'clf__n_estimators': [100, 250],
        # 'clf__max_depth': max_depth_range
        'clf__max_depth': [1, 2, 4, None]
    },
    # kNeirestNeighbour
    'knn':
    {
        'clf': [KNeighborsClassifier(n_jobs=1)],
        'clf__n_neighbors': list(range(1, 21, 1))
    }
}

