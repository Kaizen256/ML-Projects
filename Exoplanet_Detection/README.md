# 🪐 Exoplanet Detection with ML Classifiers (Kepler Dataset)

This project focuses on detecting whether a Kepler object of interest is a true exoplanet or a false positive. Using NASA's Kepler observations I trained and evaluated various classifiers to distinguish real exoplanets from noise and false detections.

---

## 📄 About the Dataset

The data originates from the **Kepler Space Observatory**, a NASA mission launched in 2009 to discover Earth-like exoplanets by observing the brightness of stars.

- **KOI**: Object identified by a transit-like dip in light.
- **Disposition**: Indicates whether an object is CONFIRMED, FALSE POSITIVE, or CANDIDATE.
- **Data Shape**: ~10,000 objects of interest, with ~50 astrophysical features per object.

I removed several non-informative or leakage columns were removed (like `koi_pdisposition`, `koi_fpflag_*`, etc.).

📚 Original dataset: [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)

---

## 🧠 Models Used

We trained and compared multiple models with and without hyperparameter tuning:

- **Untuned XGBoost Classifier Hyperparameters:** Default
- **Tuned XGBoost Classifier Hyperparameters:** colsample_bytree: 0.8, learning_rate: 0.25, max_depth: 8, n_estimators: 200, subsample: 1
- **Untuned LightGBM Classifier Hyperparameters:**: learning_rate: 0.1, max_depth: -1, min_child_samples: 20, n_estimators: int = 100, num_leaves: 31
- **Tuned LightGBM Classifier Hyperparameters:** learning_rate: 0.05, max_depth: -1, min_child_samples: 50, n_estimators: 300, num_leaves: 63
- **Untuned Random Forest Hyperparameters:** n_estimators: 300, max_depth: 9
- **Tuned Random Forest Hyperparameters:** n_estimators: 500, max_depth: 9
- **SGDClassifier Hyperparameters:** loss: 'hinge', penalty: 'l2', alpha: 0.0001

---

## 🔧 Preprocessing

- Removed rows where 'koi_disposition' was 'CANDIDATE'
- Mapped 'koi_disposition' to True or False (0,1)
- Removed human-annotated flags and disposition labels to avoid leakage.
    'rowid', 'kepid', 'kepoi_name', 'kepler_name','koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_pdisposition', 'koi_score', 'koi_tce_delivname', 'ra', 'dec', 'koi_tce_plnt_num'
- Found 377 rows that had mostly null features so I dropped them.
- Created a correlation heatmap with plotly.
- Split dataframe into train and test frames.
- Filled remaining null values with the mean of the column.
- Standardized features
- Applied 5-Fold Cross Validation for performance evaluation.

---

## 📊 Model Performance

| Model                       | Cross-Validation Accuracy  | Test Accuracy |
|-----------------------------|----------------------------|---------------|
| LGBM Classifier (Untuned)   | 94.40%                     | 95.03%        |
| XGBoost (Untuned)           | 94.09%                     | 95.02%        |
| LGBM Classifier (Tuned)     | 94.76%                     | 94.67%        |
| XGBoost (Tuned)             | 94.65%                     | 94.52%        |
| Random Forest (Tuned)       | 93.17%                     | 93.73%        |
| Random Forest (Untuned)     | 93.12%                     | 93.66%        |
| SGD Classifier (hinge)      | 91.44%                     | 92.51%        |

*Sorted by Test Accuracy
---

## 📦 Libraries Used

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `plotly`
- `IPython.display`

---
