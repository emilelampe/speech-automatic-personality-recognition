import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedGroupKFold



def merge_cv_results(cv_results_list):
    '''Takes a list of 5 cv_results_ DataFrames and returns a single merged DataFrame as if 5-fold GridSearchCV was performed'''
    assert len(cv_results_list) == 5, "There must be 5 cv_results_ DataFrames."

    # Combine DataFrames and rename split0_test_score columns
    for i, df in enumerate(cv_results_list):
        df = df.copy()
        df.rename(columns={"split0_test_score": f"split{i}_test_score"}, inplace=True)
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
        "params": "first",
        "split0_test_score": "first",
        "split1_test_score": "first",
        "split2_test_score": "first",
        "split3_test_score": "first",
        "split4_test_score": "first",
    }

    # Add columns with specific parameters to aggregation dictionary
    param_cols = [col for col in combined_df.columns if col.startswith("param_")]
    for col in param_cols:
        aggregate_dict[col] = "first"

    # Group by 'params' column, and aggregate the other columns
    grouped_df = combined_df.groupby("params_str", as_index=False).agg(aggregate_dict)

    # Calculate mean_test_score and std_test_score
    test_score_columns = [f"split{i}_test_score" for i in range(5)]
    grouped_df["mean_test_score"] = grouped_df[test_score_columns].mean(axis=1)
    grouped_df["std_test_score"] = grouped_df[test_score_columns].std(axis=1)

    # Calculate rank_test_score
    grouped_df["rank_test_score"] = grouped_df["mean_test_score"].rank(ascending=False, method="min")

    # Keep only the columns we need and return the result
    final_columns = [
        "mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time",
        "params"] + param_cols + ["split0_test_score", "split1_test_score", "split2_test_score",
        "split3_test_score", "split4_test_score", "mean_test_score",
        "std_test_score", "rank_test_score"
    ]
    final_df = grouped_df.loc[:, final_columns]

    final_df['rank_test_score'] = final_df['rank_test_score'].astype(int)

    final_df = final_df.sort_values(by="rank_test_score", ascending=True, ignore_index=True)

    return final_df


class BalancedStratifiedGroupKFold:
    '''StratifiedGroupKFold that ensures that each fold has both classes in y_train and y_test, if the data allows it.'''
    def __init__(self, n_splits=5, max_attempts=1000):
        self.n_splits = n_splits
        self.max_attempts = max_attempts

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
                return cv.split(X, y, groups)

        raise ValueError(f'Failed to find suitable folds after {self.max_attempts} attempts')