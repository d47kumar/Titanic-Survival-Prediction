# Titanic Survival Prediction
This repository presents a predictive model built to forecast the survival of Titanic passengers. The project integrates data cleaning, missing value imputation, feature engineering, and model training wrapped in one clean pipeline. The primary goal is to demonstrate how different imputation methods—MICE and KNN—affect the distribution of the critical feature Age, and to create a logistic regression model for the prediction of the survival outcome.

Table of Contents
Overview

Dataset Description

Installation & Dependencies

Data Preprocessing

Exploratory Data Analysis

Handling Missing Values

Feature Engineering

Modeling

Logistic Regression

Generating Submission

Usage Guide

Further Improvements

Acknowledgments

Overview
This project leverages the Titanic dataset (train.csv and test.csv) to predict whether a passenger would survive based on features such as passenger class, sex, age, fare, and more. The workflow consists of the following key steps:

Loading and Inspecting Data: Reading in data files using pandas and quickly inspecting the dataset structure to understand data types, dimensions, and missing values.

Data Cleaning: Addressing missing data by:

Mapping the Sex feature to numerical values.

Imputing missing values for Age using two strategies: MICE (Multiple Imputation by Chained Equations) and KNN imputation.

Filling in or dropping other missing fields.

Visualization: Comparing the distribution of Age after both imputation methods using Kernel Density Estimation (KDE) plots via seaborn.

Feature Engineering and Encoding: Dropping extraneous columns and applying one-hot encoding to convert categorical variables into a numerical format suitable for model training.

Model Training: Fitting a logistic regression model on the cleaned and transformed training data.

Prediction Output: Producing a final CSV file that combines Passenger IDs with the survival predictions.

Dataset Description
The dataset contains 891 training samples and 12 columns. Some key features include:

PassengerId: Unique identifier

Survived: Target variable indicating survival (0 = No, 1 = Yes)

Pclass: Ticket class (1, 2, or 3)

Name, Sex, Age: Passenger details (with Age having missing values)

SibSp & Parch: Family relation counts

Ticket, Fare: Ticket and fare information

Cabin & Embarked: Cabin information (many missing values) and port of embarkation

Installation & Dependencies
Ensure you have Python 3 installed. The code utilizes the following main libraries:

numpy – for numerical operations

pandas – for data processing and CSV handling

seaborn & matplotlib – for visualization

scikit-learn – for imputation methods and model training

Install dependencies via pip:

bash
pip install numpy pandas seaborn matplotlib scikit-learn
Data Preprocessing
Exploratory Data Analysis
Before any modifications, the dataset is loaded and inspected using:

python
import pandas as pd
df = pd.read_csv("train.csv")
print(df.shape)
print(df.info())
print(df.isnull().sum())
This helps identify missing values (notably in the Age, Cabin, and Embarked fields).

Handling Missing Values
Two distinct imputation strategies are explored on the Age feature:

MICE Imputation

The code selects a core group of features and uses the Iterative Imputer:

python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imputer = IterativeImputer(random_state=100, max_iter=10)
df_train_mice = df.loc[:, ["Pclass", "Age", "Sex_Number", "SibSp", "Parch", "Fare"]]
imputer.fit(df_train_mice)
df_imputed_mice = imputer.transform(df_train_mice)
df_mice = pd.DataFrame(df_imputed_mice, columns=["Pclass", "Age", "Sex_Number", "SibSp", "Parch", "Fare"])
KNN Imputation

Using KNN with 2 neighbors for comparison:

python
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2)
df_train_knn = df.loc[:, ["Pclass", "Age", "Sex_Number", "SibSp", "Parch", "Fare"]]
df_imputed_knn = imputer.fit_transform(df_train_knn)
df_knn = pd.DataFrame(df_imputed_knn, columns=["Pclass", "Age", "Sex_Number", "SibSp", "Parch", "Fare"])
Visualization Comparison

The density distribution of the original Age, MICE-imputed Age, and KNN-imputed Age are overlaid using seaborn:

python
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))
sns.kdeplot(data=df["Age"], color='crimson', ax=ax, fill=True)
sns.kdeplot(data=df_mice["Age"], color='limegreen', ax=ax, fill=True)
sns.kdeplot(data=df_knn["Age"], color='blue', ax=ax, fill=True)
plt.xlabel('Age')
plt.ylabel('Density')
plt.show()
After visual comparison, the KNN imputation is chosen to replace the original Age values.

Additional steps include:

Mapping the Sex column to a numerical representation.

Filling in missing values in the Embarked column using the mode.

Dropping columns that are either irrelevant or contain too many missing entries (like Cabin).

Feature Engineering
Convert the Sex categorical value into a numeric indicator:

python
new_data = {'male': 0, 'female': 1}
df["Sex_Number"] = df["Sex"].map(new_data)
Once the appropriate imputation is applied, update the dataset and drop intermediate columns:

python
df.loc[:, ["Pclass", "Age", "Sex_Number", "SibSp", "Parch", "Fare"]] = df_knn
df.fillna(df["Embarked"].mode()[0], inplace=True)
df = df.drop(columns=['Cabin', 'Sex_Number'])
A similar procedure is followed for the test dataset.

Modeling
Logistic Regression
The model is built using scikit-learn’s LogisticRegression:

Data Splitting and Encoding:

Drop non-essential features (like PassengerId, Name, Ticket) and convert categorical variables using one-hot encoding:

python
import pandas as pd
x = df.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket'])
x = pd.get_dummies(x)
y = df['Survived']
Model Training:

The logistic regression model with an increased maximum iteration limit is fitted on the training data:

python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
lr.fit(x, y)
Prediction on Test Data:

Preprocess the test data similarly and generate predictions:

python
df_test_x = df_test.drop(columns=['PassengerId', 'Name', 'Ticket'])
df_test_x = pd.get_dummies(df_test_x)
predictions = lr.predict(df_test_x)
Generating Submission
The predicted survival outcomes are combined with the corresponding Passenger IDs and written to a CSV file:

python
pred = pd.DataFrame(predictions, columns=['Survived'])
final = pd.concat([df_test['PassengerId'], pred], axis=1)
final.to_csv("submission.csv", index=False)
Note: A DataConversionWarning may appear during model training due to the shape of y. This is a common scenario when a column vector is passed instead of a 1D array. You can resolve this by using y.ravel() if necessary.

Usage Guide
Clone the Repository:

bash
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction
Install Dependencies:

bash
pip install -r requirements.txt
Place the Data Files: Ensure that train.csv and test.csv are located in the project root directory.

Run the Pipeline: Execute the main Python script:

bash
python main.py
View the Submission: The predictions will be saved to submission.csv, ready for upload or further review.

Further Improvements
Hyperparameter Tuning: Experiment with different parameters for the logistic regression model to potentially increase accuracy.

Feature Engineering: Explore additional features or transformations (such as combining SibSp and Parch) that could provide deeper insights.

Advanced Models: Consider using ensemble methods or more complex classification models.

Cross-Validation: Implement a cross-validation procedure to better assess model performance and reduce overfitting.

An interesting idea is to visualize the data flow using an ASCII diagram:

[Data Loading] --> [Exploratory Analysis] --> [Missing Value Imputation]
        |                                               |
        v                                               v
[Feature Engineering] ------------------> [Visualization (KDE Plots)]
        |
        v
[Model Training (Logistic Regression)]
        |
        v
[Prediction Generation] --> [Submission File]
This pipeline ensures a robust approach to handling missing data and evaluates the imputation's impact before modeling.

Acknowledgments
Thanks to the contributors of the Titanic dataset.

The project is powered by open-source libraries and tools, including Pandas, NumPy, Seaborn, Matplotlib, and Scikit-learn.
