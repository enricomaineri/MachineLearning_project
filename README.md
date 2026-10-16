# MachineLearning_project

A documented machine-learning workflow for exploring customer churn from Spotify user activity and account attributes.

## Project

The notebook loads the supplied dataset, checks its structure and target balance, preprocesses mixed numerical and categorical features, compares several classifiers with stratified cross-validation, and evaluates the selected pipeline against a majority-class baseline on a held-out test set.

The pipeline keeps preprocessing and optional random oversampling inside each training fold. The identifier is excluded from model inputs, and the test set is not used to choose a model. Evaluation includes precision, recall, F1, ROC-AUC, a classification report, and a confusion matrix.

## Files

- [MachineLearning_project.ipynb](MachineLearning_project.ipynb): complete analysis with explanations and annotated workflow.
- [data/spotify_churn_dataset.csv](data/spotify_churn_dataset.csv): dataset used by the notebook.
- [requirements.txt](requirements.txt): Python dependencies.

## Run

Open the notebook in Jupyter or Google Colab from the repository root. Install the dependencies in `requirements.txt` first. Keep the dataset at `data/spotify_churn_dataset.csv`; the notebook uses this relative path. Cross-validation and model search can take several minutes.

## Data source

[Spotify Dataset for Churn Analysis — Kaggle](https://www.kaggle.com/datasets/nabihazahid/spotify-dataset-for-churn-analysis). Please consult the source page for its license and terms of use.

## Interpretation

The notebook prints metrics when run; this README does not claim a precomputed model score. Results from this dataset are exploratory and should not be treated as evidence of production readiness. Performance on a held-out split does not guarantee transfer to different users or time periods.
