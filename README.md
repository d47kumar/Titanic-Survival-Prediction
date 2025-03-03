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
- [Accuracy](#accuracy)
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
```

## Data Preparation

### Exploratory Data Analysis

Before any modifications, the dataset is loaded and inspected using:

```python
import pandas as pd
df = pd.read_csv("train.csv")
print(df.shape)
print(df.info())
print(df.isnull().sum())
```

This helps identify missing values (notably in the Age, Cabin, and Embarked fields).

### Handling Missing Values

Two distinct imputation strategies are explored on the Age feature:

#### MICE Imputation
The code selects a core group of features and uses the Iterative Imputer:
```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imputer = IterativeImputer(random_state=100, max_iter=10)
df_train_mice = df.loc[:, ["Pclass", "Age", "Sex_Number", "SibSp", "Parch", "Fare"]]
imputer.fit(df_train_mice)
df_imputed_mice = imputer.transform(df_train_mice)
df_mice = pd.DataFrame(df_imputed_mice, columns=["Pclass", "Age", "Sex_Number", "SibSp", "Parch", "Fare"])
```

#### KNN Imputation
Using KNN with 2 neighbors for comparison:
```python
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2)
df_train_knn = df.loc[:, ["Pclass", "Age", "Sex_Number", "SibSp", "Parch", "Fare"]]
df_imputed_knn = imputer.fit_transform(df_train_knn)
df_knn = pd.DataFrame(df_imputed_knn, columns=["Pclass", "Age", "Sex_Number", "SibSp", "Parch", "Fare"])
```

#### Visualization Comparison
The density distribution of the original Age, MICE-imputed Age, and KNN-imputed Age are overlaid using seaborn:
```python
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))
sns.kdeplot(data=df["Age"], color='crimson', ax=ax, fill=True)
sns.kdeplot(data=df_mice["Age"], color='limegreen', ax=ax, fill=True)
sns.kdeplot(data=df_knn["Age"], color='blue', ax=ax, fill=True)
plt.xlabel('Age')
plt.ylabel('Density')
plt.show()
```

After visual comparison, the KNN imputation is chosen to replace the original Age values.

Additional steps include:

• Mapping the Sex column to a numerical representation.

• Filling in missing values in the Embarked column using the mode.

• Dropping columns that are either irrelevant or contain too many missing entries (like Cabin).

### Feature Engineering
Convert the Sex categorical value into a numeric indicator:

```python
new_data = {'male': 0, 'female': 1}
df["Sex_Number"] = df["Sex"].map(new_data)
```
Once the appropriate imputation is applied, update the dataset and drop intermediate columns:

```python
df.loc[:, ["Pclass", "Age", "Sex_Number", "SibSp", "Parch", "Fare"]] = df_knn
df.fillna(df["Embarked"].mode()[0], inplace=True)
df = df.drop(columns=['Cabin', 'Sex_Number'])
```

## Modeling
### Logistic Regression

The model is built using scikit-learn’s LogisticRegression:

#### Data Splitting and Encoding:

Drop non-essential features (like PassengerId, Name, Ticket) and convert categorical variables using one-hot encoding:

```python
import pandas as pd
x = df.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket'])
x = pd.get_dummies(x)
y = df['Survived']
```
####Model Training:

The logistic regression model with an increased maximum iteration limit is fitted on the training data:

```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
lr.fit(x, y)
```
#### Prediction on Test Data:

Preprocess the test data similarly and generate predictions:

```python
df_test_x = df_test.drop(columns=['PassengerId', 'Name', 'Ticket'])
df_test_x = pd.get_dummies(df_test_x)
predictions = lr.predict(df_test_x)
```

## Generating Submission
The predicted survival outcomes are combined with the corresponding Passenger IDs and written to a CSV file:

```python
pred = pd.DataFrame(predictions, columns=['Survived'])
final = pd.concat([df_test['PassengerId'], pred], axis=1)
final.to_csv("submission.csv", index=False)
```
## Accuracy
The accuracy score is 76.794%.

## Acknowledgments
Thanks to the contributors of the Titanic dataset.

The project is powered by open-source libraries and tools, including Pandas, NumPy, Seaborn, Matplotlib, and Scikit-learn.
