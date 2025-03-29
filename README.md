# Credit Card Fraud Detection Project

## Project Overview: Understanding the Basics and Goals

### What are we trying to find out?
The primary goal of this project is to build a robust machine learning model that can accurately detect fraudulent credit card transactions. Fraudulent transactions are rare, making this a highly imbalanced classification problem that requires careful handling to achieve satisfactory performance.

### Existing Knowledge:
Credit card fraud detection is a critical application of anomaly detection and binary classification in the financial industry. Traditional approaches include rule-based systems, logistic regression, and decision trees. However, these techniques often struggle with class imbalance and evolving fraud tactics. Due to a limited number of columns with information relevant to fraud, there is a need for advanced feature engineering to extract meaningful insights from the data.

### What are we aiming to achieve?
The aim is to develop a high-performing model with the following objectives:
- High recall (to capture as many fraudulent transactions as possible).
- Reasonable precision (to reduce the number of false positives).
- Robustness against class imbalance.

### What factors affect our results?
- Class imbalance (very few fraudulent transactions compared to legitimate ones).
- Feature selection and engineering.
- Quality of data preprocessing (handling of missing values, categorical encoding, etc.).
- Choice of machine learning model and hyperparameter tuning.

### Is there something new we can use?
In addition to using traditional machine learning algorithms, this project will explore:
- Advanced feature engineering techniques.
- Ensemble methods.
- Resampling techniques (ROS, SMOTE, ADASYN, etc.) to address class imbalance.
- Hyperparameter tuning using Optuna for optimal performance.

## Project Stages

### 1. Data Preparation:
- Data was collected from a credit card transaction dataset.
- Due to the size of the dataset, only a sample of 300,000 transactions from 2020 was taken.
- Tables were united, large categories reduced, and text fields cleaned.

### 2. EDA – Exploratory Data Analysis:
- Data visualization techniques were used to understand the distribution of features and identify patterns related to fraudulent activity.
- Relationships between features were analyzed to detect potential correlations.

### 3. Data Cleansing:
- Outlier detection and removal where appropriate.
- Handling of missing values through various imputation techniques.

### 4. Encoding:
- Categorical variables were transformed using one-hot encoding and label encoding where necessary.

### 5. Feature Engineering:
- Creation of new features such as transaction frequency, card type, and geographical mapping.
- Dimensionality reduction using techniques like PCA when necessary.

### 6. Handling Imbalanced Data:
- Applied various techniques such as Random Oversampling (ROS), SMOTE, ADASYN, and others to address class imbalance.

### 7. Model Selection and Fine-Tuning:
- Multiple algorithms were tested, including XGBoost, Gradient Boosting Classifier, and Random Forest Classifier.
- Hyperparameter tuning was performed using Optuna to enhance model performance.

### Conclusion
The project aims to deliver a high-performance fraud detection model with improved recall and precision by leveraging advanced feature engineering, handling of class imbalance, and optimized hyperparameters.
