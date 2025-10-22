# HR Data Prediction (Machine Learning Classification Project)

## Project Overview

* The **HR Data Prediction** project focuses on predicting employee outcomes using classification techniques.
* It covers **data preprocessing**, **exploratory data analysis (EDA)**, **feature engineering**, and **machine learning modeling**.
* The goal is to create a reliable model to support HR decisions using data-driven insights.

🔗 <a href=""View the Project

---

## Objective

* To develop a **classification model** that predicts HR-related outcomes.
* To understand data patterns, remove inconsistencies, and improve model accuracy through proper preprocessing and feature handling.

---

## Tools and Libraries Used

* **Programming Language:** Python
* **Libraries:**

  * Pandas
  * NumPy
  * Matplotlib
  * Seaborn
  * Scikit-learn (sklearn)
  * SMOTE (Synthetic Minority Oversampling Technique)

---

## Steps and Tasks Performed

### ## Data Preprocessing

* Loaded dataset and analyzed:

  * `.shape`, `.info()`, `.dtypes`, `.describe()`, `.nunique()`
* Removed:

  * **Null values**
  * **Duplicate rows**
* Checked **statistical summary** of numerical columns for data consistency.

---

### ## Exploratory Data Analysis (EDA)

* Visualized distributions using **count plots** and **histograms** for:

  * Age
  * Education
  * Department
  * Business Travel
* Checked for **skewness** in numerical columns.
* Performed **box plots** to identify and understand **outliers**.
* Detected and **removed outliers** using the **Z-score method**.

---

### ## Feature Engineering

* Separated:

  * **Categorical columns**
  * **Numerical columns**
* Applied **Label Encoding** for categorical variables.
* Standardized numerical features using **StandardScaler** to remove biasness.
* Balanced the dataset using **SMOTE** to handle class imbalance.
* Converted skewed numerical columns into categorical bins when needed.

---

### ## Model Building

* Split the dataset into **Training** and **Testing sets** using:

  ```python
  from sklearn.model_selection import train_test_split
  ```
* Implemented **Logistic Regression** as the main classification algorithm:

  ```python
  from sklearn.linear_model import LogisticRegression
  ```
* Evaluated model using:

  * **Accuracy Score**
  * **Confusion Matrix**
  * **Classification Report**
  * **ROC Curve**
  * **Cross-Validation (KFold)**

  ```python
  from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve
  from sklearn.model_selection import cross_val_score, KFold
  ```

---

### ## Model Evaluation

* Assessed model accuracy and compared training vs testing results.
* Used **cross-validation** for consistency checking.
* Verified balanced performance using **confusion matrix** and **ROC curve analysis**.

---

## Results

* Developed a **highly accurate classification model** using **Logistic Regression**.
* Improved dataset quality by:

  * Removing null values and duplicates
  * Handling outliers
  * Reducing skewness
  * Balancing data using SMOTE
* Ensured stable model results through scaling and proper encoding techniques.

---

## Conclusion

* This project successfully demonstrates an **end-to-end machine learning workflow** for HR analytics.
* Covers data cleaning, preprocessing, feature transformation, model training, and evaluation.
* Provides a foundation for similar HR-based predictive analytics and business intelligence tasks.

---

🔗 View the Project

---

