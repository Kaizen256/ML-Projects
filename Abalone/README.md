# Abalone Age Prediction with Machine Learning Regressors

Predicting the age of abalone from physical measurements.  The age of
abalone is determined by cutting the shell through the cone, staining it,
and counting the number of rings through a microscope -- a boring and
time-consuming task.  Other measurements, which are easier to obtain, are
used to predict the age.  Further information, such as weather patterns
and location (hence food availability) may be required to solve the problem.

---

## 📄 About the Dataset

From the original data examples with missing values were removed (the
majority having the predicted value missing), and the ranges of the
continuous values have been scaled for use with an ANN (by dividing by 200).

Data comes from an original (non-machine-learning) study:

Warwick J Nash, Tracy L Sellers, Simon R Talbot, Andrew J Cawthorn and
Wes B Ford (1994) "The Population Biology of Abalone (_Haliotis_
species) in Tasmania. I. Blacklip Abalone (_H. rubra_) from the North
Coast and Islands of Bass Strait", Sea Fisheries Division, Technical
Report No. 48 (ISSN 1034-3288)

📚 Original dataset: [UCI Abalone Dataset](https://archive.ics.uci.edu/ml/datasets/abalone)

---

## 🧠 Models Used

Here are the models trained and their hyperparameters.

- **SGDRegressor (Default)**: penalty: `'l2'`, alpha: `0.0001`, loss: `'squared_error'`
- **SGDRegressor (Tuned)**: penalty: `'l1'`, alpha: `0.00105`, loss: `'squared_error'`
- **XGBoost Regressor (Default)**: default parameters
- **XGBoost Regressor (Tuned)**: learning_rate: `0.05`, max_depth: `5`, n_estimators: `100`, colsample_bytree: `0.8`, subsample: `0.8`
- **Random Forest Regressor (Default)**: n_estimators: `100`, max_depth: `None`

---

## 🔧 Preprocessing

- Create age column by adding 1.5 to the rings column
- One-hot encoded the categorical `Sex` feature
- Standardized numerical features using `StandardScaler`
- Applied 5-Fold Cross Validation
- Evaluated with **Root Mean Squared Error (RMSE)**

---

## 📊 Model Performance

| Model                       | Cross-Validation RMSE  | Test Set RMSE |
|-----------------------------|------------------------|----------------|
| XGBoost Regressor (Tuned)   | 2.1282                 | 2.2005         |
| Random Forest (Default)     | 2.1845                 | 2.2245         |
| SGD Regressor (Tuned)       | 2.2540                 | 2.2420         |
| SGD Regressor (Default)     | 2.2545                 | 2.2422         |
| XGBoost Regressor (Default) | 2.3355                 | 2.3810         |

*Sorted by Cross-Validation RMSE

---

## 📦 Libraries Used

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `plotly`
