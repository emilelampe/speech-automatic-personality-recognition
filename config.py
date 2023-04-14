import numpy as np
from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ------------
# ---CONFIG---
# ------------

'''The config is divided in 2 sections:
    - The first section contains specific parameters such as the dataset filename
    - The second section contains the parameter grid for the grid search'''

# database to use
# For single label example dataset: 'single_label_example_dataset.pkl' (begin_col_labels = 2, begin_col_features = 3), from ("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data")
# db = "full-sspn-egemaps-average.pkl"
db = "single_label_example_dataset.pkl"

# Column indices where the labels and the features begin
begin_col_labels = 2
begin_col_features = 3

# Label column to choose in case of multi-label dataset
t = 'Label'

# model to use ('svm_rbf', 'knn', 'rf', 'svm_l')
m = 'svm_rbf'


# minimum and maximum length cutoff ('Length' column needed)
sc = 0
ec = 0

# scoring metric
scoring = 'roc_auc'

# random seed
seed = 42

# number of GridSearch folds
n_searches = 5

# RFECV base estimator
clf_rfecv = DecisionTreeClassifier(random_state=seed)

# Calibration method
cal_method = 'sigmoid'

# # step size for RFECV
# if f == 'compare':
#     # step = 0.2
#     step = 400
# elif f == 'egemaps':
#     step = 3

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
pre_pars = {
        'scaler': [StandardScaler()],
        # 'pca': [PCA(0.95), 'passthrough'],
        'pca': ['passthrough'],
    }

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
        # 'clf__base_estimator__C': C_range,
        'clf__C': [1, 10, 100, 1000],
        # 'clf__base_estimator__gamma': gamma_range,
        'clf__gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001]
    },
    # Random Forest
    'rf':
    {
        'clf': [RandomForestClassifier(random_state=seed, n_jobs=1)],
        'clf__n_estimators': [10, 100, 250, 500, 1000, 1500],
        'clf__max_depth': max_depth_range
    },
    # kNeirestNeighbour
    'knn':
    {
        'clf': [KNeighborsClassifier(n_jobs=1)],
        'clf__n_neighbors': list(range(1, 21, 1))
    }
}

