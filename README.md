# 📊 Customer Churn Prediction - Machine Learning Project

This project focuses on building a machine learning model to predict customer churn using a real-world dataset. The goal is to identify customers likely to leave a service so the business can take proactive retention actions.

---

## 📌 Objective

To analyze customer behavior and build a predictive model that classifies whether a customer is likely to churn or not.

---

## 🛠️ Tools & Technologies

- **Language:** Python  
- **IDE:** Jupyter Notebook / Google Colab  
- **Libraries:** 
  - `pandas`, `numpy` – Data handling
  - `matplotlib`, `seaborn` – Visualization
  - `scikit-learn` – Machine Learning
  - `xgboost` – Advanced model performance
  - `joblib` – Model saving

---

## 📁 Dataset

- **Source:** [Public churn datasets from Kaggle / Telco customer churn]
- **Sample Features:**
  - Customer ID
  - Gender, Age, Tenure
  - Contract Type, Monthly Charges
  - Payment Method
  - Churn (Target variable)

---

## 🔍 Exploratory Data Analysis (EDA)

- Univariate & bivariate analysis
- Correlation heatmap to identify relationships
- Visualizations to analyze:
  - Churn by contract type
  - Monthly charges vs churn
  - Tenure and churn relationship

---

## 🔄 Data Preprocessing

- Handled missing and duplicate values
- Converted categorical variables using:
  - Label Encoding
  - One-Hot Encoding
- Feature scaling using `StandardScaler`

---

## 🤖 Model Building

Models tested:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

Techniques used:
- Train-test split
- Cross-validation
- Hyperparameter tuning using GridSearchCV

---

## 🧪 Model Evaluation

Metrics:
- Accuracy
- Precision, Recall, F1-score
- ROC-AUC Score
- Confusion Matrix

Best model: **XGBoost**
- Accuracy: ~85%
- ROC-AUC Score: ~0.88
- Feature Importance extracted and visualized

---

## 📈 Key Insights

- Customers with short tenure, high monthly charges, and month-to-month contracts are more likely to churn.
- Electronic check payment users churn more frequently.
- Long-term contracts and paperless billing reduced churn likelihood.

---

## ✅ Conclusion

- Built a reliable churn prediction model using real-world data.
- Can be used by businesses to identify high-risk customers and reduce churn through targeted actions.

---

## 🚀 How to Run This Project

1. Clone the repository:
   ```bash
   git clone https://github.com/Pavani-Reddy111/churn-prediction-ml.git
# churn-prediction-
