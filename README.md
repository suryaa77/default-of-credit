# 💳 Default of Credit Card Clients - Machine Learning Project

This project predicts whether a credit card customer will **default on their payment** in the upcoming month. It involves end-to-end data analysis, feature engineering, and applying various classification models to determine credit risk.

> Built by **Surya** using Python and Scikit-learn.

---

## 🔍 Problem Statement

Credit card companies want to reduce their risk by identifying customers who are likely to **default** (fail to pay).  
This project explores a real-world dataset and uses supervised machine learning to predict the `default.payment.next.month` status based on customer demographics and payment history.

---

## 📊 Dataset Information

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- **Size**: 30,000 records
- **Target variable**: `default.payment.next.month` (1 = Default, 0 = No Default)

### Key Features:
| Feature            | Description                          |
|--------------------|--------------------------------------|
| `LIMIT_BAL`        | Amount of credit given (NT dollar)   |
| `SEX`              | Gender (1 = Male, 2 = Female)         |
| `EDUCATION`        | Education level (1 to 4)             |
| `MARRIAGE`         | Marital status (1 = Married, etc.)   |
| `AGE`              | Age in years                         |
| `PAY_0` to `PAY_6` | Repayment status for past 6 months   |
| `BILL_AMT1-6`      | Monthly bill statements              |
| `PAY_AMT1-6`       | Monthly payments made                |

---

## 🧠 Goal

To train machine learning models that **accurately predict customer default** using:
- Historical payment behavior
- Credit limit
- Demographic data

---

## 🛠️ Tech Stack & Tools

- Python 3.x
- NumPy
- Pandas
- Matplotlib & Seaborn (for EDA)
- Scikit-learn (ML algorithms)
- Jupyter Notebook / Colab

---

## 🧪 ML Algorithms Used

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- (Optional) XGBoost

Models were evaluated using:
- Accuracy
- Precision / Recall / F1-Score
- Confusion Matrix
- ROC-AUC Score

---

## 📈 Exploratory Data Analysis (EDA)

Key steps:
- Checked class distribution and handled imbalance
- Correlation heatmap of features
- Visualizations of default rate vs features (age, gender, education, etc.)
- Feature scaling where needed (e.g., StandardScaler)

---

## ✅ Final Results (Sample)

> *(Update this based on your actual results)*

| Model              | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 81%      | 0.79      | 0.71   | 0.75     |
| Random Forest      | 85%      | 0.83      | 0.76   | 0.79     |
| SVM                | 82%      | 0.80      | 0.72   | 0.76     |

---

## 📁 Project Structure

default-of-credit/
├── data/ # Raw dataset (Excel or CSV)
├── notebook.ipynb # Main analysis and model building
├── README.md # Project overview
└── requirements.txt # (Optional) Python dependencies

---

## 📌 How to Run the Project

1. Clone the repository:
```bash
git clone https://github.com/suryaa77/default-of-credit.git
cd default-of-credit

