# Pulsar Detection with Linear Regression and XGBoost

Pulsars are rare types of neutron stars that emit detectable radio waves as they rotate. These emissions are of great scientific interest, but identifying them is challenging due to the overwhelming presence of noise and radio frequency interference (RFI) in telescope data.

The dataset used in this project comes from the HTRU 2 survey and contains:
- **17,898 total candidates**
- **1,639 positive (real pulsars)**
- **16,259 negative (RFI/noise)**

Each candidate is described by 8 continuous variables:
1. Mean of the integrated profile  
2. Standard deviation of the integrated profile  
3. Excess kurtosis of the integrated profile  
4. Skewness of the integrated profile  
5. Mean of the DM-SNR curve  
6. Standard deviation of the DM-SNR curve  
7. Excess kurtosis of the DM-SNR curve  
8. Skewness of the DM-SNR curve  

📄 **Dataset Source**  
R. J. Lyon et al., *Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach*, MNRAS, 2016.

---

## 🧠 Models Used

1. **Stochastic Gradient Descent (SGDClassifier)**
2. **XGBoost Classifier**

---

## 📦 Libraries Used

- `pandas`  
- `numpy`  
- `scikit-learn`  
- `xgboost`  
- `IPython`

---

## 📊 Model Performance

| Model            | Train Accuracy   | Test Accuracy |
|------------------|------------------|---------------|
| SGD Classifier   | 97.98%           | 97.71%        |
| XGBoost          | 98.08%           | 97.84%        |
