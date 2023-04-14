import pandas as pd

def merge_cv_results(cv_results_list):
    '''Takes a list of 5 cv_results_ DataFrames and returns a single merged DataFrame as if 5-fold GridSearchCV was performed'''
    assert len(cv_results_list) == 5, "There must be 5 cv_results_ DataFrames."

    # Combine DataFrames and rename split0_test_score columns
    for i, df in enumerate(cv_results_list):
        df = df.copy()
        df.rename(columns={"split0_test_score": f"split{i}_test_score"}, inplace=True)
        cv_results_list[i] = df

    combined_df = pd.concat(cv_results_list, ignore_index=True)

    # Convert "params" column to string
    combined_df["params"] = combined_df["params"].apply(str)

    # Define dictionary for aggregation
    aggregate_dict = {
        "mean_fit_time": "mean",
        "std_fit_time": "mean",
        "mean_score_time": "mean",
        "std_score_time": "mean",
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
    grouped_df = combined_df.groupby("params", as_index=False).agg(aggregate_dict)

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