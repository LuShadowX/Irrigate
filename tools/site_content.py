"""Text and figures for the Irrigate site (used by build_site.py)."""

NAME = "Irrigate"
REPO = "https://github.com/LuShadowX/Irrigate"
ACCENT = "#2f6db5"
SOFT = "#e8f0fa"
TAGLINE = "Applied machine learning on a real Kaggle competition: pre-processing, features, classifiers, regression, ensembles and clustering, ending in a new submission."
HEADLINE = "From my first Kaggle submission to a model that beats it."
INTRO = [
    "Six notebooks, one per unit of CSE274, built on the Kaggle Playground S6E4 <i>Predicting Irrigation Need</i> "
    "competition (630,000 fields, scored on balanced accuracy). My first entry in April 2026, a plain XGBoost on one-hot "
    "features, scored 0.960. This project rebuilds the pipeline properly, unit by unit: leakage-safe pre-processing, "
    "engineered features, every classifier in the syllabus, ensembles of LightGBM, XGBoost and CatBoost, and decision "
    "weights tuned for the rare High class.",
    "Regression uses a second Kaggle dataset from the same domain (crop yields for 101 countries, 1990–2013).",
]
STATS = [("0.9713", "new Kaggle private score (was 0.9608)"), ("6", "notebooks, one per unit"), ("50", "syllabus topics mapped"), ("5", "unit tests")]
FIGURES = [
    ("i05-final", "Cross-validated balanced accuracy, from my first submission to the final blend."),
    ("i01-imbalance", "What each class-imbalance fix does to the recall of each class."),
    ("i03-roc-pr", "ROC and precision–recall curves of the decision tree, one per class."),
    ("i04-regularisation", "Ridge and Lasso coefficient paths, and cross-validated R² on unseen countries."),
    ("i05-boundaries", "Decision surfaces of a tree, a forest and two boosted models (DecisionBoundaryDisplay)."),
    ("i06-dbscan", "DBSCAN core, border and noise points; K-Means vs DBSCAN on two moons."),
]
RUN = """git clone https://github.com/LuShadowX/Irrigate && cd Irrigate
pip install -e ".[dev]"
kaggle competitions download -c playground-series-s6e4 -p data && unzip data/*.zip -d data
kaggle datasets download -d patelris/crop-yield-prediction-dataset -p data/yield --unzip
pytest
python -m irrigate.submit --baseline    # final models + submissions/submission.csv
jupyter lab notebooks/"""
FOOTER = "Lukshya Singh Supyal · B.Tech CSE · built for CSE274 Applied Machine Learning · data: Kaggle Playground Series S6E4, Crop Yield Prediction Dataset"
OUTCOME_TITLE = "What the course description says students learn"
