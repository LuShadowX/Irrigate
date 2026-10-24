# Irrigate

Applied machine learning on a real Kaggle competition, from raw data to a submission. Built for **CSE274 Applied Machine Learning** (LPU).

**Site:** https://lushadowx.github.io/Irrigate/ (every notebook with its outputs, plus a course map)

The data is the Kaggle [Playground Series S6E4: Predicting Irrigation Need](https://www.kaggle.com/competitions/playground-series-s6e4) competition: 630,000 fields, three classes (Low 58.7%, Medium 37.9%, High 3.3%), scored on **balanced accuracy**. My first entry in April 2026 (one-hot features, XGBoost, argmax) scored 0.96021. This project rebuilds the pipeline one syllabus unit at a time and ends with a new submission:

| | CV balanced accuracy | Kaggle public / private |
|---|---|---|
| April 2026: one-hot XGBoost, argmax | 0.9617 | 0.96021 / 0.96077 |
| October 2026: LightGBM + XGBoost + CatBoost, domain features, tuned decision weights | 0.9703 (held-out fold) | **0.96985 / 0.97126** |

| Notebook | Unit | What's in it |
|---|---|---|
| [01 Pre-processing](notebooks/01_preprocessing.ipynb) | I | types of data, two kinds of leakage demonstrated, MCAR/MAR/MNAR and imputation, outlier detectors on injected faults, encodings, scaling, class imbalance (resampling, SMOTE, class weights, threshold moving) |
| [02 Features and dimensionality](notebooks/02_features_dimensionality.ipynb) | II | filter / wrapper / embedded selection, domain and aggregation features (leak-free transformers), the curse of dimensionality, PCA vs LDA |
| [03 Classification](notebooks/03_classification.ipynb) | III | perceptron from scratch, logistic regression, Naive Bayes, k-NN, decision tree, SVM, confusion matrix, ROC and PR curves |
| [04 Regression](notebooks/04_regression.ipynb) | IV | on the Kaggle crop-yield data: SLR, MLR and coefficients, polynomial terms, Ridge / Lasso / Elastic Net paths, tree regression, time-series forecasting |
| [05 Ensembles and the final model](notebooks/05_ensembles_final_model.ipynb) | V | voting, bagging, random forest, AdaBoost, GBM, XGBoost / LightGBM / CatBoost, CV strategies, pipelines, grid / random / halving search, the final submission and its Kaggle score |
| [06 Unsupervised learning](notebooks/06_unsupervised.ipynb) | VI | distance metrics, K-Means and K-Medoids from scratch, hierarchical clustering, DBSCAN, anomaly detection, silhouette and Davies–Bouldin |

## The package

`src/irrigate/`:

- `features.py`: `DomainFeatures` (dryness, evaporation drive, thirsty stage × no mulch …), `GroupStats` and `FrequencyEncoder`, all fit/transform so nothing leaks inside cross-validation
- `preprocess.py`: the column transformer and pipelines used everywhere
- `decision.py`: per-class decision weights that maximise balanced accuracy
- `evaluate.py`: metrics, confusion matrices, ROC / PR curves
- `submit.py`: trains the final models on all 630,000 rows and writes the Kaggle submission
- `data.py`, `plotting.py`

## Run it

```bash
git clone https://github.com/LuShadowX/Irrigate && cd Irrigate
pip install -e ".[dev]"
kaggle competitions download -c playground-series-s6e4 -p data && unzip data/playground-series-s6e4.zip -d data
kaggle datasets download -d patelris/crop-yield-prediction-dataset -p data/yield --unzip
pytest
python -m irrigate.submit --baseline    # final models, submissions/submission.csv and report.json
jupyter lab notebooks/
python tools/build_site.py              # rebuild docs/
```

Neither dataset is committed (Kaggle's rules don't allow redistributing competition data).

Lukshya Singh Supyal · B.Tech CSE
