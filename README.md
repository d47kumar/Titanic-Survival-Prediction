# Titanic Survival Prediction Model

This repository presents a predictive model built to forecast Titanic passenger survival using logistic regression. The project covers data cleaning, missing value imputation via two methods (MICE & KNN), feature engineering, and model training—all encapsulated in one clear pipeline.

---

## Table of Contents

- [Overview](#overview)
- [Dataset Description](#dataset-description)
- [Installation & Dependencies](#installation--dependencies)
- [Data Preprocessing](#data-preprocessing)
  - [Loading & Exploring Data](#loading--exploring-data)
  - [Handling Missing Values](#handling-missing-values)
  - [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
  - [Logistic Regression](#logistic-regression)
- [Generating Submission](#generating-submission)
- [Usage Guide](#usage-guide)
- [Further Improvements](#further-improvements)
- [Acknowledgments](#acknowledgments)

---

## Overview

The project uses the Titanic dataset to predict whether a passenger would survive. The workflow includes:

1. **Data Loading & Inspection:**  
   Examining the dataset structure and identifying missing values.
2. **Data Cleaning:**  
   Mapping categorical values (like `Sex`) to numerical indicators and addressing missing entries for features such as `Age` using two imputation strategies.
3. **Visualization:**  
   Comparing the `Age` distribution post-imputation with Kernel Density Estimation (KDE) plots.
4. **Feature Engineering:**  
   Dropping extraneous columns and applying one-hot encoding.
5. **Model Training:**  
   Fitting a logistic regression model and generating predictions.
6. **Submission:**  
   Combining predictions with Passenger IDs and outputting a CSV file.

---

## Dataset Description

The dataset includes:
- **Train Dataset (`train.csv`):** 891 passengers, 12 columns (e.g., `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Cabin`, `Embarked`).
- **Test Dataset (`test.csv`):** Similar structure but without the `Survived` column.

---

## Installation & Dependencies

Ensure you have Python 3 and the following libraries installed:

- **numpy**
- **pandas**
- **seaborn**
- **matplotlib**
- **scikit-learn**

Install them via pip:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn

## Modeling

### Logistic Regression

After cleaning and preprocessing the data, we fit a logistic regression model to predict passenger survival.

#### Data Preparation

We first separate the target variable and drop unnecessary columns. Then, we apply one-hot encoding to handle categorical features:

```python
# Prepare training features and target
y_train = pd.DataFrame(df['Survived'])
df_x = df.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket'])
df_test_x = df_test.drop(columns=['PassengerId', 'Name', 'Ticket'])

# Convert categorical variables using one-hot encoding
x_train = pd.get_dummies(df_x)
x_test = pd.get_dummies(df_test_x)
