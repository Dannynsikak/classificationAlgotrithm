# Titanic Survival Prediction

This project applies a Random Forest Classifier to predict which passengers were most likely to survive the Titanic disaster. The dataset is sourced from Kaggle's Titanic dataset.

# Project Overview

The goal of this project is to train a classification model using scikit-learn to determine survival likelihood based on passenger features such as age, sex, ticket class, and embarkation point.

# Dataset

The dataset used is Titanic.csv, which contains information on passengers, including:

survived (target variable: 0 = No, 1 = Yes)

sex, age, class, embarked, alone (features used for training)

# Installation

To run this project, you need Python 3.x and the following dependencies:

pip install pandas scikit-learn

# Usage

Place the Titanic.csv dataset in the project directory.

Run the classification script:

python titanicDataClassification.py

The script will train a Random Forest model and print the accuracy and classification report.

# Code Explanation

The script performs the following steps:

Load Data: Reads the Titanic dataset using pandas.

Preprocessing:

Drops unnecessary columns.

Encodes categorical variables (sex, embarked, class, alone) using LabelEncoder.

Fills missing values in age with the median.

Model Training:

Splits data into 80% training and 20% testing.

Trains a Random Forest Classifier with 100 trees.

Evaluation:

Predicts survival on test data.

Prints accuracy score and classification report.

Expected Output

    `Model Accuracy: 0.81

    Classification Report:
    precision recall f1-score support

            0       0.83      0.87      0.85       XXX
            1       0.80      0.74      0.77       XXX

        accuracy                           0.81       XXX

    macro avg 0.81 0.81 0.81 XXX
    weighted avg 0.81 0.81 0.81 XXX`

(Note: XXX represents actual values that will vary based on dataset splits.)

# License

This project is for educational purposes only. Feel free to modify and experiment!
