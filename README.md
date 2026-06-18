# 👨‍💼 Employee Attrition Prediction System

> Predicting whether an employee is likely to leave the organization using Machine Learning.



![Python](https://img.shields.io/badge/Python-3.11-blue)




![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)




![Model](https://img.shields.io/badge/Model-Random%20Forest-green)




![Accuracy](https://img.shields.io/badge/Accuracy-81.29%25-yellow)




![License](https://img.shields.io/badge/License-Educational-orange)



---

## 📌 Table of Contents
- [Overview](#overview)
- [Demo](#demo)
- [Dataset](#dataset)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Model Performance](#model-performance)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Developer](#developer)

---

## 📖 Overview
Employee attrition is one of the biggest challenges faced by HR departments. This project uses the **IBM HR Analytics Dataset** to build a machine learning model that predicts whether an employee will leave the company or not.

The application is built using **Streamlit** and deployed on **Streamlit Cloud** for interactive predictions.

---

## 🎥 Demo
🔗 **Live App:** [Click Here](https://your-app-link.streamlit.app)

---

## 📊 Dataset

| Property      | Details |
|----------     |---------|
| **Source**    | IBM HR Analytics — Kaggle |
| **Rows**      | 1,470 employees |
| **Columns**   | 35 features |
| **Target**    | Attrition (Yes/No) |
| **Class Imbalance** | 84% No — 16% Yes |

---

## 🎯 Top 10 Features Used

| # | Feature               | Description |
|---|-------------|         |--------------------|
| 1 | OverTime              | Whether employee works overtime |
| 2 | MaritalStatus         | Marital status of employee |
| 3 | StockOptionLevel      | Stock options given to employee |
| 4 | Age | Age of employee |
| 5 | JobLevel              | Seniority level in company |
| 6 | YearsWithCurrManager  | Years working with current manager |
| 7 | JobSatisfaction       | Job satisfaction level (1-4) |
| 8 | YearsAtCompany        | Total years spent in company |
| 9 | MonthlyIncome         | Monthly salary of employee |
| 10 | TotalWorkingYears    | Total work experience in years |

---

## 🛠️ Tech Stack

| Technology                    | Purpose |
|-------------|             |-----------------|
| Python 3.11               |Programming Language |
| Scikit-learn              | Machine Learning |
| Random Forest Classifier  | Prediction Model |
| SMOTE                     | Handle Class Imbalance |
| StandardScaler            | Feature Scaling |
| Streamlit                 | Web Application |
| Pandas & NumPy            | Data Processing |
| Matplotlib & Seaborn      | Data Visualization |
| Pickle                    | Model Saving |

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| ✅ Accuracy | 81.29% |
| ✅ ROC-AUC | 75.77% |
| 🌲 Algorithm | Random Forest Classifier |
| 🌳 Trees | 200 |
| ⚖️ Balancing | SMOTE |

---

## 🚀 How to Run Locally

### Step 1 — Clone Repository
```bash
git clone https://github.com/Rishabh-18-tech/EmployeeAttritionPrediction.git
cd EmployeeAttritionPrediction