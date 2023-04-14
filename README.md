# Automatic Personality Recognition (APR) from Speech

## Introduction

This repository contains the code for a thesis project on speech-based Automatic Personality Recognition (APR) at TU Delft. The project aims to explore the challenges of APR and investigate two new databases for this purpose: Nautilus Speaker Characterization (NSC) corpus (scripted speech) and a spontaneous speech dataset collected by a research team from Universitat Politècnica de València. The performance of the models is compared with those trained on a widely-used Automatic Personality Perception (APP) database: the Speaker Personality Corpus (SPC) from the INTERSPEECH 2012 Speaker Trait Challenge. The machine learning algorithms used for this project are Support Vector Machine (SVM) with RBF kernel, k-Nearest Neighbors (kNN), and Random Forest.

As APR is a multi-label classification problem, this project can be used as well for other multi-label classification problems. Furthermore, the data is always split speaker-indepent, both in the train, validation and test split, as in the inner cross-validation (using StratifiedGroupKFold). It could therefore also be of use for other projects that need speaker/group-independence.

## Specifications

### HPC capabilities

The project is build so that it supports multiprocessing on a HPC controlled by Slurm, using iPyParallel. An .sh file is included that can be used to execute run.py with different configurations. When this .sh script executes run.py multiple times, the same batch number is assigned to all instances of run.py. This causes all results of that batch to be stored in the same directory, making it easy to find and compare results from the same batch. If run.py is executed directly, it will create a new random batch number every time. If you don't want to use iPyParallel for multiprocessing, you can comment out the imports and code at the beginning of run.py where the ipyparallel is configured. The n_jobs of the grid search should be set to -1 instead of len(c).

### Conventions

The project is built with the following naming conventions:

```
dataset:    {name}-{featureset}-{binary_threshold_type}.pkl
result dir: YYMMDD_hhmm_{batch_id}
log file:   YYMMDD_hhmm_{batch_id}.log
```

Columns with metadata should be placed first, then the labels, and then the features.
The following is an example:

| **ID**   | **Group**  | **Length** | **Label 1** | **Label 2** | **Feature 1** | **Feature 2** |
|----------|------------|------------|-------------|-------------|---------------|---------------|
| clip_001 | speaker_01 | 5204       | 1           | 0           | 0.6753        | 0.3563        |
| clip_002 | speaker_02 | 1107       | 0           | 0           | 0.2476        | 0.1375        |
| clip_003 | speaker_01 | 4593       | 1           | 0           | 0.4674        | 0.1378        |

## Structure

The folder structure of the repository is as follows:

```
speech-automatic-personalty-recognition
├── apr
│   ├── functions.py
│   └── multilabel_stratified_group_split.py
├── config.py
├── data
│   ├── single_label_example_dataset.pkl
│   └── spc-egemaps-average.pkl
├── log
│   └── 230414_1155_8388.log
├── README.md
├── results
│   ├── 230414_1155_8388
│   │   ├── 8388-svm_rbf-Extraversion-best_result.csv
│   │   ├── best_estimators
│   │   └── cv_results
└── run.py

```

## Requirements

To run the code, you need to have Python 3.7+ installed along with the following libraries:

- matplotlib
- numpy
- pandas
- scikit-learn
- joblib
- ipyparallel

You can install these libraries using pip:

```python
bashCopy code
pip install matplotlib numpy pandas scikit-learn joblib ipyparallel
```

## Datasets

The datasets used in this project are:

1. Nautilus Speaker Characterization (NSC) corpus (scripted speech)
2. A spontaneous speech dataset collected by a research team from Universitat Politècnica de València
3. Speaker Personality Corpus (SPC) from the INTERSPEECH 2012 Speaker Trait Challenge

These datasets are not included in this repository due to privacy and licensing issues. However, you can replace them with your own datasets by updating the paths in the `train.py` script.

## Usage

To use the training script, follow these steps:

1. Clone this repository:

   ```python
   bashCopy codegit clone https://github.com/user/repo.git
   cd repo
   ```

2. Update the dataset paths in the `train.py` script to match your local paths:

   ```python
   pythonCopy code
   db = "path/to/your/dataset.pkl"
   ```

3. Run the training script:

   ```python
   bashCopy code
   python train.py
   ```

### Config

The config is divided in two parts:
- Parameters for specific settings
- A parameter grid for the grid search

You can customize the training script by modifying the following parameters:

- `db`: The name of the dataset file (in .pkl)
- `t`: Target label to train and evaluate on
- `begin_col_labels`: Column index where labels start
- `begin_col_features`: Column index where features start
- `sc` and `ec`: Minimum and maximum cutoff lengths for speech segments
- `scoring`: Scoring metric for model evaluation
- `seed`: Random seed for reproducibility
- `n_searches`: Number of grid search folds
