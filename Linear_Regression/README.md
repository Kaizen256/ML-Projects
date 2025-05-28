# Linear Regression Projects

This folder contains multiple regression-based projects using `scikit-learn` and `xgboost`. Each project uses a different dataset and focuses on building predictive models and evaluating their performance.

---

## 📁 Projects

### 1. **Advertising Dataset**
- **Data**: Synthetic dataset with advertising budgets (`TV`, `Radio`, `Newspaper`) and resulting `Sales`.
- **Goal**: Predict `Sales` based on marketing budget.
- **Model**: Scikit-learn's `LinearRegression`

### 2. **Diabetes Dataset**
- **Data**: Scikit-learn’s built-in diabetes dataset.
- **Goal**: Predict disease progression based on clinical features.
- **Model**: Scikit-learn's `LinearRegression`

### 3. **House Prices**
- **Data**: Kaggle’s Advanced House Prices dataset
- **Goal**: Predict house sale prices and improve leaderboard score
- **Models**: `LinearRegression`, `XGBoostRegressor`

---

## 📦 Libraries Used

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `plotly`

---

## 📊 Model Performance Summary

| Project        | Model(s)                | Metric               | Value        |
|----------------|-------------------------|----------------------|--------------|
| Advertising    | Linear Regression       | Mean Squared Error   | ~1.45        |
| Diabetes       | Linear Regression       | Mean Squared Error   | ~3058.09     |
| House Prices   | Linear, XGBoost         | RMSE (XGBoost)       | **0.12961**  |

---

