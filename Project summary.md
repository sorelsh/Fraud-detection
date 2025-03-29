# Project summary 

# 1. Data Preparation

**Project Overview:**

This project aims to predict credit card fraud in the United States using transaction data from 2019–2020. The primary dataset, `credit_card_fraud.csv`, includes 26 columns of features covering:

* **Personal Information:** ssn, cc\_num, first/last names, gender, address details, etc.
* **Transaction Details:** trans\_date, trans\_time, amt, is\_fraud
* **Merchant Information:** merchant, merch\_lat, merch\_long

A secondary file, `customers.csv`, was found to be redundant as its information is already present in the main file. 

The main dataset contains over **34 million records** with a significant class imbalance of approximately **0.56% fraud**. To manage this large dataset, a representative random sample of **300,000 transactions from 2020** was selected. This sampling strategy preserved the monthly fraud distribution, particularly the peaks observed in January and February and the dip in December.

**Preparation Steps:**

1.  **Data Loading:** Performed using scalable tools (e.g., Dask).
2.  **Initial Inspection:** Checked for data integrity, and **no missing values** were found.
3.  **Data Cleaning:**
    * Standardized text fields (e.g., removed "fraud\_" prefix from merchant names).
    * Applied proper type conversion for numerical and date fields.
    * Flagged redundant identifiers (ssn, cc\_num, acct\_num) for potential consolidation.
4.  **Feature Engineering:**
    * Extracted temporal features: year, month.
    * Computed geographic distances between customer and merchant locations.

These steps ensured a clean and enriched dataset for subsequent analysis and modeling.

## 2. Exploratory Data Analysis (EDA) & Statistical Testing

EDA revealed a severe class imbalance in the dataset—with only 0.56% fraud cases against 99.44% non-fraud—making it imperative to employ specialized modeling techniques. Fraud rates vary only slightly across regions (0.46%–0.64%), suggesting geography is not a primary fraud predictor. However, demographic insights are notable: Education & Research Professionals face higher fraud risk (up to 0.68%), urban adults aged 50+ show an increased risk (~0.84%), while urban females aged 25–50 have the lowest (~0.34%). Temporal analysis indicates that fraud is predominantly concentrated during nighttime transactions, and certain transaction categories stand out, with the “other” category at 100% and “shopping_pos” at 2.33%, far exceeding rates for other categories (0.06%–0.15%).

A Spearman correlation analysis further underscored key predictors. The `trans_time_group` (0.092) and transaction amount `amt` (0.087) exhibited the strongest positive correlations with fraud, while `category` (0.063) also contributed moderately. Despite weaker correlations for demographic features like `profile` (0.021) and `age` (0.018), their interactions could enhance predictive power. Significant interdependencies—such as the high correlation between `trans_time` and `trans_time_group` (0.809), and between `category` and `trans_month` (0.823)—highlight potential redundancies that need addressing during feature selection.

Complementing these findings, Chi-Square tests identified that categorical variables including `profile`, `category`, `trans_time`, `trans_time_group`, and `trans_month` are statistically significant predictors (p < 0.05) of fraud. In contrast, features like `state`, `job`, and `trans_day` did not show significant associations and may be excluded to reduce noise. Together, these analyses guide the prioritization of time-related features and transaction categories while suggesting that demographic factors play a supplementary role, thereby refining the feature set for a more interpretable and streamlined fraud detection model.


# 3. Data Cleansing

**Dataset Overview & Initial Inspection:**

* **300,000 transactions** across **21 features**.
* **No missing values** were found.
* Several numerical features exhibited notable outliers.

**Outlier Handling & Missing Value Imputation:**

* **City Population (`city_pop`):** Outliers retained. outliers were believed important for distinguishing between large and small cities in fraud analysis.
* **Transaction Amount (`amt`):** Extreme values retained.
* **Distance (`distance_km`) and Age:** Outliers set to NaN if removal didn't affect distribution; missing age values imputed using **Multiple Imputation by Chained Equations (MICE)**.

**Documentation & Data Saving:**

* The cleansed and imputed DataFrame was saved in pickle format.

# 4. Feature Engineering & Selection

**Feature Engineering:**

* **Temporal Features:** `trans_year`, `trans_month`, `trans_time_group` (Morning/Afternoon/Evening/Night).
* **Geographic Insights:** `distance_km`, customer coordinates mapping.
* **Behavioral Indicators:** card brand/type, time since last transaction per account, statistical aggregates (mean, standard deviation, Z-scores of transaction amounts for `amt`).

**Feature Selection Process:**

* Model-based approach using **five methods** (Lasso, SVM with L1, Gradient Boosting, Random Forest, Ridge).
* Features selected by at least **four of the five models** were retained.
* **Final set of 15 key features:** ssn, state, city\_pop, acct\_num, profile, category, amt, distance\_km, age, trans\_month, trans\_day, trans\_time\_group, card\_type, card\_brand, time\_since\_last\_trans, and amt\_zscore.
* Categorical variables were one-hot encoded.
* Redundancies (e.g., high correlation between `trans_time` and `trans_time_group`) were addressed by selecting one representative feature.

**Conclusion:**

The refined DataFrame was saved as `df_model_data_after_FeatureEngineering_23.3.25.pkl`.

# 5. Model Selection and Fine-Tuning

**Resampling Strategy & Baseline Evaluation:**

* Severe class imbalance (**0.56% fraud**).
* Baseline model (no resampling): F1-score of **~0.875**.
* Random Over Sampling (ROS): F1-score of **~0.864**.
* More aggressive techniques (RUS, SMOTE, ADASYN, BorderlineSMOTE, SMOTETomek) resulted in lower performance.

**Hyperparameter Optimization:**

* Optuna was used to fine-tune an **XGBoost classifier**.
* **5-fold stratified cross-validation** was employed.
* Optimized hyperparameters led to a balanced precision and recall.

**Final Model Checks:**

* Two models constructed: one on original imbalanced data (baseline) and one with ROS.
* Tuned XGBoost model achieved an F1-score of **0.9342** on the test set.
* Confusion matrices showed high recall for fraud detection.

**Conclusion:**

The hyperparameter-tuned XGBoost model was saved as a Pickle file for deployment.

# 6. Model Evaluation

**Evaluation Overview:**

* Model performance assessed on two test datasets:
    * **2020 sample:** 1,000,000 transactions
    * **2019 sample:** 300,000 transactions

**2020 Data Performance:**

* Both baseline and ROS-enhanced models showed robust performance with high precision and F1-scores.
* ROC Model (with resampling) improved sensitivity.

**2019 Data Performance:**

* Baseline model: Performance degraded with **recall = 0.0** (failed to predict fraud).
* ROC Model: Maintained strong performance (**recall = 0.9135**, **precision = 0.9818**, **ROC-AUC = 0.99**).

**Key Insights & Next Steps:**

* Training data impact: Model trained on 2020 data requires retraining for older or evolving datasets.
* Continuous improvement: Regular model updates are essential.
* Deployment: ROS-tuned model is preferred for real-time fraud detection.

**Conclusion:**

The ROC Model's resampling approach provides superior generalization across time, highlighting the need for ongoing evaluation and dynamic updates.

# 7. Model Deployment 

#### Model Deployment and Real-World Application:

* The final phase of this project involves deploying the optimized fraud detection model into a real-world environment where it can operate in near real-time to identify suspicious credit card transactions. The deployment strategy is designed to integrate seamlessly with financial institutions' existing IT infrastructures and risk management systems.

#### Deployment Architecture
- **Model Packaging:** The hyperparameter-tuned XGBoost model, along with its feature preprocessing pipeline, is serialized and saved as a Pickle file. This encapsulation ensures that the model, including all necessary data transformations, can be reliably loaded and executed in production.
- **Real-Time Inference:** The deployed model will operate as a REST API service, receiving streaming transaction data, processing it through the preprocessing pipeline, and generating fraud probability scores. This enables rapid decision-making, where high-risk transactions trigger immediate alerts for further investigation.
- **Scalability & Integration:** Utilizing containerization (e.g., Docker) and orchestration tools (e.g., Kubernetes), the deployment system is scalable to handle high transaction volumes. It integrates with the financial institution’s core systems, such as transaction processing and customer relationship management (CRM), to allow seamless risk assessment.

#### Monitoring and Maintenance
- **Performance Monitoring:** Continuous monitoring systems will track key metrics like latency, F1-score, precision, and recall to ensure consistent model performance. Automated alerts will be set up to detect deviations from expected behavior, prompting immediate review.
- **Model Retraining:** Given the dynamic nature of fraudulent behavior, a periodic retraining strategy is in place. The model will be updated with the latest data to capture emerging fraud patterns, ensuring that the system remains robust over time.
- **Feedback Loop:** A feedback mechanism collects user and analyst insights on flagged transactions, enabling continuous improvement of the model. This iterative process ensures that the fraud detection system adapts to changes in customer behavior and evolving fraud tactics.

#### Real-World Impact
In practice, this deployment will empower financial institutions to:
- **Mitigate Losses:** By detecting fraud in real-time, institutions can halt fraudulent transactions before significant financial loss occurs.
- **Improve Customer Trust:** Enhanced fraud detection capabilities contribute to a safer transaction environment, increasing customer confidence.
- **Optimize Resource Allocation:** Automated alerts allow human investigators to focus on the most critical cases, improving overall operational efficiency.

This model deployment strategy thus transforms a robust predictive model into an actionable tool, effectively bridging the gap between data science and operational risk management in the financial sector.

