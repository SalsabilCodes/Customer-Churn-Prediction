# 📌 Customer Churn Prediction Project

## 1️⃣ Project Overview
This project predicts whether a **telecom customer will churn** (leave the service) based on customer demographics, account information, and usage patterns. The goal is to provide insights and predictions to help the company reduce churn and retain customers.  

- **Data Source:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
- **Tools & Libraries:** Python, Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn (SMOTE), Matplotlib, Seaborn, Flask, Pickle  

---

## 2️⃣ Dataset Description
The dataset contains **7043 customers** with the following attributes:

| Column | Description |
|--------|-------------|
| customerID | Unique customer identifier (dropped during processing) |
| gender | Male or Female |
| SeniorCitizen | Whether the customer is a senior (0 = No, 1 = Yes) |
| Partner | Customer has a partner (Yes/No) |
| Dependents | Customer has dependents (Yes/No) |
| tenure | Number of months the customer has stayed |
| PhoneService | Whether the customer has phone service |
| MultipleLines | Whether the customer has multiple lines |
| InternetService | Type of internet service (DSL, Fiber optic, None) |
| OnlineSecurity | Yes/No/No internet service |
| OnlineBackup | Yes/No/No internet service |
| DeviceProtection | Yes/No/No internet service |
| TechSupport | Yes/No/No internet service |
| StreamingTV | Yes/No/No internet service |
| StreamingMovies | Yes/No/No internet service |
| Contract | Type of contract (Month-to-month, One year, Two year) |
| PaperlessBilling | Yes/No |
| PaymentMethod | Electronic check, Mailed check, Bank transfer, Credit card |
| MonthlyCharges | The monthly bill amount |
| TotalCharges | Total bill amount |
| Churn | Target variable (Yes = churn, No = retain) |

---

## 3️⃣ Data Preprocessing
1. **Dropped** `customerID` column as it is not useful for prediction.  
2. **Checked for missing values**:
   - `TotalCharges` had blank strings → replaced with `0.0` and converted to float.  
3. **Converted categorical target**:
   - `Churn` → 1 (Yes), 0 (No)  
4. **Encoded categorical features** using `LabelEncoder` and saved encoders in `encoder.pkl`.  
5. **Standardized numerical features**: `tenure`, `MonthlyCharges`, `TotalCharges` using `StandardScaler` (saved as `scaler.pkl`).  
6. **Checked correlation** and distributions using **Matplotlib** and **Seaborn**.  

---

## 4️⃣ Handling Class Imbalance
- The target variable `Churn` is imbalanced.  
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** on training data to balance the classes.  

---

## 5️⃣ Model Training
Two machine learning models were trained:

1. **Random Forest Classifier**
2. **XGBoost Classifier**

- **Hyperparameter Tuning** performed with `GridSearchCV`.  
- **Random Forest** achieved the best performance and is saved as `best_model.pkl`.

---

## 6️⃣ Making Predictions
- Input data is provided as a dictionary (from a form in Flask or directly in Python).  
- Data is encoded using the saved encoders, scaled with the saved scaler, and passed to the trained model.  
- Output:
  - **Prediction:** `"Churn"` or `"No Churn"`  
  - **Probability:** likelihood of churn  

Example:

```python
example_input = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}
prediction, prob = make_prediction(example_input)
print(prediction, prob)
