"""CSE274 Applied Machine Learning: every syllabus topic and where it is covered.

Used by build_site.py for the course map. Each entry is (topic, notebook, section).
"""

COURSE = "CSE274 · Applied Machine Learning"

NOTEBOOKS = {
    "01_preprocessing": "Data pre-processing",
    "02_features_dimensionality": "Feature engineering and dimensionality reduction",
    "03_classification": "Linear and probabilistic classification models",
    "04_regression": "Regression",
    "05_ensembles_final_model": "Ensemble learning, tuning and the final submission",
    "06_unsupervised": "Unsupervised learning",
}

UNITS = [
    ("I", "Data Pre-processing", "01_preprocessing", [
        ("Introduction to data and the pre-processing pipeline", "1"),
        ("Types of data: qualitative, quantitative, structured vs unstructured", "2"),
        ("Data leakage: causes and prevention", "3"),
        ("Missing values: mechanisms (MCAR, MAR, MNAR) and handling techniques", "4"),
        ("Outliers: detection and treatment", "5"),
        ("Categorical data: one-hot, label (ordinal), frequency encoding", "6"),
        ("Standardisation (z-score) and min-max normalisation", "7"),
        ("Class imbalance: resampling, SMOTE, algorithmic approaches", "8"),
    ]),
    ("II", "Feature Engineering and Dimensionality Reduction", "02_features_dimensionality", [
        ("Feature selection: filter methods", "1A"),
        ("Feature selection: wrapper methods", "1B"),
        ("Feature selection: embedded methods", "1C"),
        ("Feature extraction and creating new features", "2A–2B"),
        ("Aggregation features", "2C"),
        ("Curse of dimensionality; dimensionality reduction", "3"),
        ("Principal component analysis (PCA)", "4A"),
        ("Linear discriminant analysis (LDA); PCA vs LDA", "4B"),
    ]),
    ("III", "Linear & Probabilistic Classification Models", "03_classification", [
        ("Introduction to supervised learning", "1"),
        ("Perceptron (from scratch)", "2.1"),
        ("Logistic regression", "2.2"),
        ("Naïve Bayes classifier", "3.1"),
        ("K-nearest neighbours", "4.1"),
        ("Decision tree", "4.2"),
        ("Support vector machine (linear and RBF)", "4.3"),
        ("Confusion matrix and basic metrics", "5.1–5.2"),
        ("ROC and precision–recall curves", "5.3"),
    ]),
    ("IV", "Regression", "04_regression", [
        ("Regression vs classification", "1"),
        ("Bias–variance considerations in regression", "2"),
        ("Simple linear regression", "3"),
        ("Multiple linear regression", "4"),
        ("Interpretation of coefficients", "5"),
        ("Polynomial feature expansion", "6"),
        ("Ridge, Lasso and Elastic Net", "7"),
        ("Effect of regularisation on model complexity", "8"),
        ("Tree-based regression models", "9"),
        ("Time-series regression models", "10"),
    ]),
    ("V", "Ensemble Learning", "05_ensembles_final_model", [
        ("Introduction; bias–variance trade-off in ensembles", "1"),
        ("Majority voting classifier: hard and soft voting", "2"),
        ("Bagging and random forest", "3"),
        ("AdaBoost and gradient boosting machines", "4"),
        ("XGBoost, LightGBM, CatBoost", "5"),
        ("Ensemble regression models", "6"),
        ("Cross-validation strategies, pipelines, hyperparameter tuning", "7"),
        ("Final model and Kaggle submission", "8–9"),
    ]),
    ("VI", "Unsupervised Learning", "06_unsupervised", [
        ("Role of unsupervised learning; supervised vs unsupervised", "1"),
        ("Distance metrics: Euclidean, Manhattan, cosine; choosing a metric", "2"),
        ("K-Means and K-Medoids (PAM)", "3"),
        ("Hierarchical clustering: types, dendrogram, linkage criteria", "4"),
        ("Density-based clustering (DBSCAN)", "5"),
        ("Anomaly detection", "6"),
        ("Evaluating clusters: inertia, silhouette, Davies–Bouldin", "7"),
    ]),
]

# From the course description ("Through this course, students learn to ..."). The course page
# gives no numbered COs; official ones from UMS can replace these.
OUTCOMES = [
    ("D1", "explain types of data and data pre-processing concepts", "01"),
    ("D2", "apply feature engineering", "02"),
    ("D3", "use dimensionality reduction techniques", "02"),
    ("D4", "compare and implement nonlinear (classification) models with appropriate evaluation metrics", "03"),
    ("D5", "compare and implement regression models with appropriate evaluation metrics", "04"),
    ("D6", "compare and implement ensemble models with appropriate evaluation metrics", "05"),
    ("D7", "compare and implement clustering models with appropriate evaluation metrics", "06"),
]
