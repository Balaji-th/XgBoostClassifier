# XgBoostClassifier : Predicting Bank Client Financial Product Subscriptions
This project aims to predict whether a bank client will subscribe to a financial product (term deposit) using machine learning techniques, specifically focusing on the challenges posed by imbalanced datasets. The solution leverages Gradient Boosted Trees (XGBoost) and Scikit-learn for robust model building and evaluation.

Table of Contents
Introduction
Dataset
Data Set Information
Feature Information
Statement of the Classification Problem
Software and Tools
Data Exploration
ML Pipeline for Data Processing
Model Training
Generalization and Prediction
Summary
Pointers to Other Advanced Techniques
References
Introduction
This project illustrates a machine learning classification approach using Gradient Boosted Trees (XGBoost) on a real-world, highly imbalanced dataset. The goal is to predict client subscription to a bank term deposit. The notebook guides the user through the following conceptual steps:
Dataset Description: Understanding the characteristics of the banking marketing dataset.
Exploratory Data Analysis (EDA): Gaining insights into the data to inform feature engineering.
Data Preprocessing: Cleaning and preparing the data using Scikit-learn pipelines and custom transformers.
Naive XGBoost Classification: Initial model training with cross-validation.
Model Evaluation: Plotting precision-recall curves and ROC curves to assess performance.
Model Tuning: Improving classification performance using weighted positive samples.
Advanced Techniques Discussion: Overview of oversampling/undersampling and SMOTE algorithms to handle class imbalance.
Dataset
The dataset used for this project is sourced from the UCI Machine Learning Repository for Bank Marketing Data Set.
Source:
[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014
Data Set Information
The data relates to direct marketing campaigns conducted by a Portuguese banking institution via phone calls. The objective is to predict whether a client will subscribe to a bank term deposit ('yes' or 'no').
Feature Information
The dataset includes various features categorized as follows:
A. Bank Client Information


Col Num
Feature Name
Feature Description
1
age
(numeric)
2
job
type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3
marital
marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4
education
(categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5
default
has credit in default? (categorical: 'no','yes','unknown')
6
balance
how much credit card balance
7
housing
has housing loan? (categorical: 'no','yes','unknown')
8
loan
has personal loan? (categorical: 'no','yes','unknown')

B. Attributes related to the last contact of the current campaign
Col Num
Feature Name
Feature Description
9
contact
contact communication type (categorical: 'cellular','telephone')
10
day
last contact day of month (categorical: '1', '2', '3', ..., '30', '31')
11
month
last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
12
duration
last contact duration, in seconds (numeric). Important note: This attribute highly affects the output target (e.g., if duration=0 then y='no'). However, the duration is not known before a call is performed. After the call, y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

C. Other attributes
Col Num
Feature Name
Feature Description
13
campaign
number of contacts performed during this campaign and for this client (numeric, includes last contact)
14
pdays
number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
15
previous
number of contacts performed before this campaign and for this client (numeric)
16
poutcome
outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

E. Output variable (desired target)
Col Num
Feature Name
Feature Description
17
y
has the client subscribed a term deposit? (binary: 'yes','no')

Statement of the Classification Problem
The primary goal is to build a machine learning model that accurately predicts whether a client is likely to subscribe to a financial product (term deposit).
Challenges in building ML model
Machine learning model building often encounters several challenges, many of which are explored in this project:
Non-representative data
Insufficient data
Poor quality data
Imbalanced dataset (a major focus of this project)
Irrelevant features
Overfitting the model on training data
Underfitting of the model on training data
Ensuring model generalization for unseen data
Software and Tools
This project utilizes the following Python libraries:
Pandas: For data loading, manipulation, and analysis.
NumPy: For numerical operations.
Scikit-learn: A comprehensive machine learning library for preprocessing, model selection, metrics, and pipeline creation.
XGBoost: An optimized distributed gradient boosting library designed for speed and performance.
Matplotlib: For creating static, animated, and interactive visualizations in Python.
Seaborn: Built on Matplotlib, providing a high-level interface for drawing attractive and informative statistical graphics.
The code also includes custom BaseEstimator and TransformerMixin classes for building a flexible machine learning pipeline.
Data Exploration
The initial step involves extensive data exploration to gain insights and inform feature engineering. This includes:
Loading the dataset using Pandas.
Previewing the data's shape, columns, and data types.
Analyzing the distribution of the target variable (y) to identify class imbalance.
Examining the mean of numerical features grouped by the target variable to uncover potential patterns.
Plotting distributions and correlations of numerical columns using Seaborn's pairplot and heatmap to identify relationships and redundant features.
ML Pipeline for Data Processing
A robust ML pipeline is constructed using Scikit-learn's Pipeline and FeatureUnion to preprocess the data consistently for both training and testing. This pipeline includes:
Data Cleaning: Handling missing values.
Feature Selection: Custom transformers (DataFrameSelector) to select specific subsets of features.
Categorical Imputation: A custom transformer (DataFrameCatImputer) to impute missing values in categorical columns with the most frequent value.
Categorical Encoding: A custom CategoricalEncoder (adapted from Scikit-learn's development branch to handle a bug in older versions) for one-hot encoding categorical features, ensuring numerical compatibility for ML algorithms without introducing bias.
Model Training
The project details multiple attempts at model training to demonstrate how to improve performance, especially with imbalanced data:
XGBoost Introduction: A brief explanation of XGBoost and its advantages.
Performance Metrics: Discussion of key metrics for evaluating classification models (e.g., precision, recall, ROC AUC).
First Attempt (Naive XGBoost): Initial model training without specific handling for imbalance.
Strategy for Imbalance: Exploration of strategies to address imbalanced data.
Second Attempt (Weighted Samples): Improving the model by using weighted positive samples during training.
Third Attempt (Weighted Samples & Feature Selection): Further refinement by combining weighted samples with feature selection techniques.
Cross-validation is extensively used throughout the training process to ensure robust model evaluation and prevent overfitting.
Generalization and Prediction
The project emphasizes the importance of splitting data into training and testing sets early in the process to ensure the model generalizes well to unseen data. The final test set is used only for evaluating the generalization performance of the chosen model.
Summary
This project provides a comprehensive walkthrough of building a classification model for an imbalanced dataset, highlighting data exploration, robust preprocessing pipelines, and iterative model training and tuning with XGBoost and Scikit-learn.
Pointers to Other Advanced Techniques
The code briefly mentions and points to other advanced techniques for handling imbalanced datasets:
Oversampling of minority class and Undersampling of majority class: Basic strategies for rebalancing datasets.
SMOTE (Synthetic Minority Over-sampling Technique): An advanced technique that generates synthetic samples for the minority class, addressing the limitations of simple oversampling.
References
Scikit-learn
Pandas
Matplotlib
Seaborn
XGBoost
SMOTE Paper
SMOTE Example
Over Sampling Example
Under Sampling Example
