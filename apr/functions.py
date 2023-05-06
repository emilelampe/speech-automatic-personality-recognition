import pandas as pd
import numpy as np
import re
import contextlib
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import norm
from sklearn.metrics import make_scorer, balanced_accuracy_score

def calculate_median_labels(df, begin_labels_col, begin_features_col):
    temp_df = pd.merge(df['Group'], df.iloc[:,begin_labels_col:begin_features_col], left_index=True, right_index=True)
    temp_df = temp_df.drop_duplicates('Group').sort_values('Group').reset_index(drop=True)
    medians_greater_equal = (temp_df.iloc[:,1:] >= temp_df.iloc[:,1:].median()).astype('int')
    medians_greater = (temp_df.iloc[:,1:] > temp_df.iloc[:,1:].median()).astype('int')
    median_greater_equal_scores = pd.merge(temp_df['Group'], medians_greater_equal, left_index=True, right_index=True)
    median_greater_scores = pd.merge(temp_df['Group'], medians_greater, left_index=True, right_index=True)
    labels = median_greater_equal_scores.columns.values[1:]
    merged = pd.DataFrame(median_greater_scores['Group'].copy())
    for l in labels:
        one_greater_equal = (median_greater_equal_scores[l] == 1).sum()
        zero_greater_equal = (median_greater_equal_scores[l] == 0).sum()
        minc_greater_equal = min(one_greater_equal, zero_greater_equal)
        majc_greater_equal = max(one_greater_equal, zero_greater_equal)
        one_greater = (median_greater_scores[l] == 1).sum()
        zero_greater = (median_greater_scores[l] == 0).sum()
        minc_greater = min(one_greater, zero_greater)
        majc_greater = max(one_greater, zero_greater)
        r1 = minc_greater_equal / majc_greater_equal
        r2 = minc_greater / majc_greater
        if r1 > r2:
            print(f"{l}, r1: {round(r1, 3)}, r2: {round(r2, 3)}, so r1: greater or equal")
            
            merged[l] = median_greater_equal_scores[l].copy()

        else:
            print(f"{l}, r1: {round(r1, 3)}, r2: {round(r2, 3)}, so r2: only greater than")

            merged[l] = median_greater_scores[l].copy()
    final = df.copy()
    for l in labels:
        final = final.drop(l, axis=1)
    final = final.merge(merged, on='Group')
    reversed_labels = list(labels)
    reversed_labels.reverse()
    for l in reversed_labels:
        col = final.pop(l)
        final.insert(begin_labels_col, l, col)
    
    return final


def merge_cv_results(cv_results_list, scoring_metrics, main_metric):
    '''Takes a list of k cv_results_ DataFrames and returns a single merged DataFrame as if k-fold GridSearchCV was performed'''
    # assert len(cv_results_list) == 5, "There must be 5 cv_results_ DataFrames."

    # Combine DataFrames and rename split0_test_score columns
    for i, df in enumerate(cv_results_list):
        df = df.copy()
        for j in scoring_metrics:
            df.rename(columns={f"split0_test_{j}": f"split{i}_test_{j}"}, inplace=True)
        cv_results_list[i] = df

    combined_df = pd.concat(cv_results_list, ignore_index=True)

    # Convert "params" column to string column
    combined_df["params_str"] = combined_df["params"].apply(str)

    # Define dictionary for aggregation
    aggregate_dict = {
        "mean_fit_time": "mean",
        "std_fit_time": "mean",
        "mean_score_time": "mean",
        "std_score_time": "mean",
        "params": "first"
    }

    for i in range(len(cv_results_list)):
        for j in scoring_metrics:
            aggregate_dict[f"split{i}_test_{j}"] = "first"

    # Add columns with specific parameters to aggregation dictionary
    param_cols = [col for col in combined_df.columns if col.startswith("param_")]
    for col in param_cols:
        aggregate_dict[col] = "first"

    # Group by 'params' column, and aggregate the other columns
    grouped_df = combined_df.groupby("params_str", as_index=False).agg(aggregate_dict)

    # Calculate mean_test_score and std_test_score
    for j in scoring_metrics:
        test_score_columns = [f"split{i}_test_{j}" for i in range(5)]
        grouped_df[f"mean_test_{j}"] = grouped_df[test_score_columns].mean(axis=1)
        grouped_df[f"std_test_{j}"] = grouped_df[test_score_columns].std(axis=1)

    # Calculate rank_test_score
    grouped_df["rank_test_score"] = grouped_df[f"mean_test_{main_metric}"].rank(ascending=False, method="min")

    # Keep only the columns we need and return the result
    final_columns = [
        "mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time",
        "params"] + param_cols
    
    for j in scoring_metrics:
        for i in range(len(cv_results_list)):
            final_columns.append(f"split{i}_test_{j}")
        final_columns.append(f"mean_test_{j}")
        final_columns.append(f"std_test_{j}")
    final_columns.append("rank_test_score")

    final_df = grouped_df.loc[:, final_columns]

    final_df['rank_test_score'] = final_df['rank_test_score'].astype(int)

    # Custom scorer that takes standard deviation into account
    final_df[f'combined_test_{main_metric}'] = final_df[f'mean_test_{main_metric}'] - 0.5 * final_df[f'std_test_{main_metric}']

    final_df["rank_test_combined"] = final_df[f"combined_test_{main_metric}"].rank(ascending=False, method="min")
    final_df['rank_test_combined'] = final_df['rank_test_combined'].astype(int)

    # Select the first row to take std_dev into account
    final_df = final_df.sort_values(by=f"combined_test_{main_metric}", ascending=False, ignore_index=True)
    # final_df = final_df.sort_values(by="rank_test_score", ascending=True, ignore_index=True)

    return final_df

class PrintSaver():
    '''Both prints a message and saves it to a file.'''
    def __init__(self, output_path):
        self.output_path = output_path
    
    def print_save(self, text):
        with open(self.output_path, "a") as file, contextlib.redirect_stdout(file):
            print(text)
        print(text)

def calculate_p_value(observed, mean, std_dev, n_bootstraps):
    # Calculate the z-score
    z_score = (observed - mean) / (std_dev/ np.sqrt(n_bootstraps))

    # Calculate the two-tailed p-value
    p_value = 2 * norm.sf(abs(z_score))
    return p_value

class BalancedStratifiedGroupKFold:
    '''StratifiedGroupKFold that ensures that each fold has both classes in y_train and y_test, if the data allows it.'''
    def __init__(self, n_splits=5, max_attempts=1000, print_saver=None, outer_fold=None):
        self.n_splits = n_splits
        self.max_attempts = max_attempts
        self.print_saver = print_saver
        self.outer_fold = outer_fold

    def _check_folds(self, X, y, groups, cv):
        for train_idx, test_idx in cv.split(X, y, groups):
            y_test = y[test_idx]
            y_train = y[train_idx]
            if (len(np.unique(y_test)) < 2) or (len(np.unique(y_train)) < 2):
                return False
        return True

    def split(self, X, y, groups):
        for seed in range(self.max_attempts):
            cv = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=seed)
            if self._check_folds(X, y, groups, cv):
                if self.print_saver:
                    if self.outer_fold:
                        self.print_saver.print_save(f"Found suitable inner folds on seed {seed} for outer fold {self.outer_fold}.")
                    else:
                        self.print_saver.print_save(f"Found suitable inner folds on seed {seed}.")
                return cv.split(X, y, groups)

        raise ValueError(f'Failed to find suitable folds after {self.max_attempts} attempts')