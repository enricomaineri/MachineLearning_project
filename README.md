# MachineLearning_project

A documented machine-learning workflow for exploring customer churn from Spotify user activity and account attributes.

## Project

The notebook loads the supplied dataset, checks its structure and target balance, preprocesses mixed numerical and categorical features, compares several classifiers with stratified cross-validation, tests whether the features carry any signal about churn, and evaluates the selected pipeline against simple baselines on a held-out test set.

The pipeline keeps preprocessing and optional random oversampling inside each training fold. The identifier is excluded from model inputs, and the test set is not used to choose a model. Evaluation includes precision, recall, F1, ROC-AUC, a classification report, and a confusion matrix.

**Tools:** Python, pandas, scikit-learn (pipelines, `ColumnTransformer`, `GridSearchCV`), imbalanced-learn, SciPy, matplotlib.

## Results

| Model (held-out test set, 1,600 users) | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- |
| Baseline: majority class | 0.000 | 0.000 | 0.000 | 0.500 |
| Baseline: always predict churn | 0.259 | 1.000 | 0.411 | 0.500 |
| Baseline: stratified random | 0.260 | 0.258 | 0.259 | 0.501 |
| Selected model (logistic regression) | 0.260 | 0.486 | 0.339 | 0.492 |

**Main finding: the dataset does not contain usable signal for predicting churn.** The selected model performs at chance level on the held-out set and does not beat the trivial *always predict churn* rule. A dedicated check on the training data explains why:

- no feature is significantly associated with churn (univariate Mann–Whitney U and chi-square tests, all p > 0.05), and churn rates are about 25–27% in every category;
- a permutation test shows that the cross-validated ROC-AUC with the real labels (0.51) is not significantly better than with randomly shuffled labels (p ≈ 0.15).

The pipeline itself is sound: the same diagnostics that guard against data leakage are what made it possible to detect the lack of signal instead of over-tuning the model on the test set. The numbers above are produced by the notebook with fixed random seeds.

## Files

- [MachineLearning_project.ipynb](MachineLearning_project.ipynb): complete analysis with explanations and annotated workflow.
- [data/spotify_churn_dataset.csv](data/spotify_churn_dataset.csv): dataset used by the notebook.
- [requirements.txt](requirements.txt): Python dependencies.

## Run

Open the notebook in Jupyter or Google Colab from the repository root. Install the dependencies in `requirements.txt` first:

```bash
pip install -r requirements.txt
jupyter notebook MachineLearning_project.ipynb
```

Keep the dataset at `data/spotify_churn_dataset.csv`; the notebook uses this relative path. A full run takes about a minute on a laptop.

## Data source

[Spotify Dataset for Churn Analysis — Kaggle](https://www.kaggle.com/datasets/nabihazahid/spotify-dataset-for-churn-analysis). Please consult the source page for its license and terms of use.

## Limitations and next steps

Results from this dataset are exploratory and should not be treated as evidence of production readiness. Because the features carry no signal here, the natural next step is to apply the same pipeline to churn data with real behavioral history (for example, trends in listening activity over time), engineer time-based features, and choose the decision threshold from the explicit cost of false positives and false negatives.
