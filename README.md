# 🧠 Diabetes Prediction Model Optimization and Deployment

<div align="center">
  
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Dataset](https://img.shields.io/badge/dataset-Pima%20Indians%20Diabetes-blue?style=for-the-badge)
![ML](https://img.shields.io/badge/machine%20learning-classification-orange?style=for-the-badge)



</div>


## ✨ Project Highlights

<div align="center">
  
🔍 **Comparative Analysis** | 🎯 **5+ ML Models** | ⚙️ **Hyperparameter Tuning**  
📊 **Interactive Dashboard** | 🏆 **Performance Metrics** | 🚀 **Deployment Ready**

</div>

---

## 🌟 Project Overview

This project focuses on developing, optimizing, and comparing multiple machine learning models to predict diabetes risk using clinical diagnostic measurements. The goal is to identify the best-performing classifier through rigorous evaluation and hyperparameter tuning, ultimately creating a deployable prediction system.

---

## 🎯 Problem Statement

<div align="center">
  
| Objective | Key Actions |
|-----------|-------------|
| Model Comparison | Train 5+ classification algorithms |
| Performance Evaluation | Measure accuracy, precision, recall, F1-score |
| Optimization | Hyperparameter tuning with Grid/Random Search |
| Visualization | Interactive dashboard for model comparison |

</div>

---

## 📊 Dataset Description

- Pima Indians Diabetes Dataset (768 samples, 9 features).
- Target: Outcome (1 = diabetic, 0 = non-diabetic).
- Class imbalance: 65% non-diabetic vs. 35% diabetic.

Example structure of dataset features:

| Feature                   | Type       | Description                           |
|---------------------------|------------|---------------------------------------|
| Age                       | Numeric    | Age of the individual                 |
| BMI                       | Numeric    | Body Mass Index                       |
| Insulin                   | Numeric    | Insulin value in the blood            |
| Glucose                   | Numeric    | Glucose level                         |
| SkinThickness             | Numeric    | Thickness of the skin of individual   |
| DiabetesPredgreeFunction  | Numeric    | Gender of the individual              |
| Pregnancies               | Numeric    | No of pregenancies                    |
| Bloodpressure             | Numeric    | Blood pressure of the individual      |
| Outcome                   | Binary     | Target variable (1 = Diabetic, 0 = Not)|

---


## Data Preparation

-  Cleaning: Handle missing values (e.g., Glucose = 0 → median imputation).

-  Scaling: StandardScaler for normalization.

-  Split: 80% training, 20% testing.

---


## Model Training 

| Model	  | Key Hyperparameters Tuned |
|---------|---------------------------|
| Logistic Regression |	C, penalty    |
| Decision Tree	      | max_depth, min_samples_split |
| Random Forest	      | n_estimators, max_features |
| SVM	                | C, kernel                 |
| Gradient Boosting	  | learning_rate, n_estimators |

----


## 🔧 Evaluation Metrics:


-  Accuracy, Precision, Recall, F1 Score, R2 Score.
-  Confusion matrices.


---


## ⚡ Hyperparameter Optimization

-  Used RandomizedSearchCV for efficient optimization.


🌳 Why Random Forest?

- Handles Imbalance: Built-in bagging reduces overfitting to majority class.
- Feature Importance: Quantifies clinical risk factors (e.g., Glucose > BMI > Age).
- Robustness: Works well with small datasets and mixed data types.


---


## 🛠️ Technologies & Tools

- Python (Google Colab)
- NumPy, Pandas
- Scikit-learn
- Seaborn, Matplotlib, Plotly
- GridSearchCV, RandomizedSearchCV
- Streamlit / Plotly Dash (for optional dashboard)

---

## 📈 Dashboard Features

- Visualize metrics like Accuracy, F1-Score, and R² Score across all models
- Compare performance before and after hyper-parameter tuning
- Highlight top-performing models for specific data conditions

---

## ▶️ How to Run (Google Colab)

1. *Open the notebook* directly in Google Colab:  
   [Diabetes_Prediction_Model_Optimization_and_Deployment.ipynb](https://colab.research.google.com/github/JaiRamteke/Diabetes_Prediction_Model_Optimization_and_Deployment/blob/main/Diabetes_Prediction_Model_Optimization_and_Deployment.ipynb)

2. *Run all cells* sequentially for:
   - Data preprocessing
   - Model training & evaluation
   - Hyper-parameter tuning
   - Results visualization

3. *(Optional)*: Download notebook as .py or export results if needed.

---


## ✅ Deliverables

- ✔️ Cleaned and split dataset
- ✔️ Trained 5+ classification models
- ✔️ Performance evaluation logs
- ✔️ Hyper-parameter tuning results
- ✔️ Visual comparison dashboard

---


## ✅ Results


![Screenshot 2025-04-12 201819](https://github.com/user-attachments/assets/369c91e0-8f1a-43e9-bb6b-5db2a67bfe48)




![Screenshot 2025-04-12 201954](https://github.com/user-attachments/assets/8d4ef7e4-b961-43ea-96fc-25f4995dd0e4)

---


## 📌 Future Improvements

- Integrate a real-time prediction dashboard.
- Add more advanced models (XGBoost, LightGBM).
- Enable model deployment via Flask or FastAPI.

---


## 🎯 Key Takeaways

-  Random Forest outperformed others after hyperparameter tuning.
-  Glucose levels were the most predictive feature.
-  Deployment-ready pipeline with modular components.

---


## 📌 Why This Project?

-  Clinical Impact: Early diabetes prediction can improve patient outcomes.
-  End-to-End ML Pipeline: Demonstrates data cleaning → training → tuning → deployment.
-  Open-Source: Fully reproducible code for healthcare/ML communities.

---


## 📌 Repository Link

🔗 GitHub Repo: [https://github.com/yourusername/diabetes-model-comparison](https://github.com/JaiRamteke/Diabetes_Prediction_Model_Optimization_and_Deployment.git)



📁 [Google Drive - Resources](https://drive.google.com/drive/folders/1u0PrWS-AEzMIpOLhnx8Oc6uLfPVZXiyp?usp=sharing)  

-  Group video of the presentation of the Project.

---


## 👨‍💻 Team Contributions

| Member  	| Focus Area	   | Key Contribution           |
|-----------|----------------|----------------------------|
| Jai Ramteke	| ML Engineering   | Model architecture, tuning |
| Nandeesh Puri	| Data Pipeline	   | Data Analysis              |
| Dyanna Joshi	| Visualization	   | Dashboard development      |



---

💡“The best model is the one that tells you the most about your data.”

---
