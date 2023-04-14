import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_array, check_consistent_length, column_or_1d
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def multilabel_stratified_group_train_test_split(
    X, y, groups, test_size=0.2, random_state=None, include_indices=True
):
    # Check input
    X = check_array(X, "csr", ensure_min_features=2)
    y = check_array(y, "csr", ensure_2d=False, ensure_min_features=2)
    groups = column_or_1d(groups, warn=True)
    check_consistent_length(X, y, groups)

    target_type = type_of_target(y)
    if target_type != "multilabel-indicator":
        raise ValueError(
            f"Multilabel stratification is not supported for target type {target_type}"
        )

    # Check that test_size is within the valid range
    if not (0 < test_size < 1):
        raise ValueError(f"Test size must be between 0 and 1, got {test_size}")

    # Shuffle groups and stratify multilabel data
    rng = check_random_state(random_state)
    group_indices = np.arange(groups.shape[0])
    rng.shuffle(group_indices)
    unique_groups, group_counts = np.unique(groups[group_indices], return_counts=True)

    mskf = MultilabelStratifiedKFold(n_splits=int(1/test_size), random_state=random_state, shuffle=True)
    
    # Create a 2D array representing the labels for each unique group
    group_labels = np.zeros((unique_groups.shape[0], y.shape[1]))
    for idx, group in enumerate(unique_groups):
        group_labels[idx] = np.any(y[groups == group], axis=0)
    
    # Assign train and test indices based on shuffled groups
    for train_group_idx, test_group_idx in mskf.split(unique_groups, group_labels):
        train_indices = group_indices[np.isin(groups[group_indices], unique_groups[train_group_idx])]
        test_indices = group_indices[np.isin(groups[group_indices], unique_groups[test_group_idx])]
        break

    if include_indices:
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices], train_indices, test_indices
    else:
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def label_distribution(y):
    return np.sum(y, axis=0) / y.shape[0]

def stratification_report(y, y_train, y_val, y_test, label_names=None):
    complete_set_distribution = label_distribution(y)
    train_set_distribution = label_distribution(y_train)
    val_set_distribution = label_distribution(y_val)
    test_set_distribution = label_distribution(y_test)

    print("Label distribution:")
    print("\n{:<20} {:<10} {:<10} {:<10} {:<10}".format("Label", "Complete", "Train", "Val", "Test"))
    for i, (complete, train, val, test) in enumerate(zip(complete_set_distribution, train_set_distribution, val_set_distribution, test_set_distribution)):
        if label_names:
            print(f"{label_names[i]:<20} {complete:.3f}{' '*5} {train:.3f}{' '*5} {val:.3f}{' '*5} {test:.3f}")
        else:
            print(f"Label {i:<10} {complete:.3f}{' '*5} {train:.3f}{' '*5} {val:.3f}{' '*5} {test:.3f}")

    print("\nDifferences:")
    print("\n{:<20} {:<15} {:<15} {:<15}".format("Label", "Train-Complete","Val-Complete", "Test-Complete"))
    for i, (train_diff, val_diff, test_diff) in enumerate(zip(np.abs(train_set_distribution - complete_set_distribution), np.abs(val_set_distribution - complete_set_distribution), np.abs(test_set_distribution - complete_set_distribution))):
        if label_names:
            print(f"{label_names[i]:<20} {train_diff:.3f}{' '*10} {val_diff:.3f}{' '*10} {test_diff:.3f}")
        else:
            print(f"Label {i:<10} {train_diff:.3f}{' '*10} {val_diff:.3f}{' '*10} {test_diff:.3f}")

    train_diff = np.abs(train_set_distribution - complete_set_distribution)
    val_diff = np.abs(val_set_distribution - complete_set_distribution)
    test_diff = np.abs(test_set_distribution - complete_set_distribution)
    print(f"\nMean of Train-Complete: {round(np.mean(train_diff),3):<15}")
    print(f"Mean of Val-Complete: {round(np.mean(val_diff),3):<15}")
    print(f"Mean of Test-Complete: {round(np.mean(test_diff),3)}\n")


def check_disjoint_groups(train_val_indices, test_indices, train_indices, val_indices, groups):
    train_val_groups = groups[train_val_indices]
    train_groups = train_val_groups[train_indices]
    val_groups = train_val_groups[val_indices]
    test_groups = groups[test_indices]
    assert np.intersect1d(train_val_groups, test_groups).size == 0
    assert np.intersect1d(train_groups, val_groups).size == 0