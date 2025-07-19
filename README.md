# Predicting Bank Term Deposit Subscriptions (CAIE tech test 2025)

## Background

AI-Vive-Banking, a Portuguese financial institution, has been running outbound telephone campaigns to persuade existing and potential clients to subscribe to a term-deposit product. Over multiple campaign waves, the bank’s IT and marketing teams have logged both demographic details (age, occupation, marital status, education, credit default history, existing loans) and call-centric metrics (contact method, number of calls this campaign, number of days since last contact) into a SQLite database (bmarket.db). This rich dataset—while noted to possibly contain synthetic or noisy entries—forms the basis for building and evaluating predictive models that can pinpoint which customers are most likely to say “yes” to a term deposit, thereby helping the bank streamline its marketing spend and boost conversion rates.

## Objective

Main Objective: Develop a predictive pipeline to identify customers most likely to subscribe to a term deposit. This will help the bank optimize call lists, allocate marketing resources more effectively, and improve campaign conversion rates.

Full Name: Tan Hao Wen
Email address: 234733T@mymail.nyp.edu.sg

Folder structure:
├── src
│ ├── config.py # For easy configuration for GridSearchCV parameters
│ └── evaluate.py # Evaluate for the model
│ ├── load_data.py # Load data from sqlite
│ └── main.py # main
│ ├── preprocess.py # Data preprocessing and feature engineering
│ └── train_model.py # Model training
├── .gitignore # dont update the virtual environment and the data folder
├── eda.ipynb # data preprocessing and visualization
├── README.md # Overview and explain of the model used and evaluation
├── requirements.txt # For the libraries needed
└── run.sh # To run the pipeline

## Environment & Dependencies

Programming Language used: Python 3.11.9
List of libraries used:
pandas == 2.3.0
numpy == 2.3.1
scikit-learn == 1.6.1
matplotlib == 3.10.3
seaborn == 0.13.2
plotly = 6.2.0
xgboost == 3.0.2
imbalanced-learn == 0.13.0

## Exploratory Data Analysis

Our exploratory analysis surfaced several key insights that directly informed our preprocessing pipeline:

- **Class imbalance:** Only ~11–12% of customers subscribed. We applied **SMOTE OR SMOTE-Tomek** to rebalance the training set, improving recall without excessively inflating false positives.
- **Age outliers:** Records with `Age < 18` or `Age > 100` are invalid (minors can’t open term deposits; centenarians are likely data errors). We filtered ages to the **18–100** range, then applied **StandardScaler** to normalize the remaining distribution.
- **Campaign Calls extremes:** A few customers had ≥ 50 calls. Rather than discard these outliers, we retained them and scaled the feature so that our models handle extreme values naturally.
- **Previous Contact sentinel:** A value of `999` in **Previous Contact Days** indicates “never contacted.” We converted this into a binary **Had Previous Contact** flag.
- **Missing/unknown flags:** For loan and default fields, we mapped `yes → 1`, `no → 0`, and `unknown → –1` so that “unknown” remains an informative category.
- **Low numeric correlation:** All numeric features showed |r| < 0.3, so we retained the full feature set rather than drop potentially useful predictors.

### Evaluation & Conclusion

Filtering out-of-range ages improved data integrity and prevented scaling distortions. Standardizing continuous variables ensured stable convergence for both linear and gradient-based models. Our binary/ternary mappings preserved sentinel information without bloating the feature space. Addressing class imbalance with SMOTE-Tomek delivered measurable gains in recall and F1. Finally, low inter-feature correlation allowed us to keep all predictors, simplifying the pipeline and enhancing interpretability. These EDA-driven choices underpin the robustness and performance of our final models.

#### Feature Engineering

-Map Education Level to years (e.g., ‘high school’ → 12).

-Create Had Previous Contact flag from Previous Contact Days.

-Convert loans and default to {1,0,–1}.

-Scaling: StandardScaler on Age & Campaign Calls.

-Encoding:Logistic Regression: One-Hot Encoding for nominal features.RF/XGBoost: Label Encoding for faster training.

#### Key takeaways

Ordinal Mapping

Education → Years: Supplies a single quantitative variable for LR’s linear boundary and for tree-based models to split on meaningful thresholds.

Previous Contact Flag: Collapses a noisy “999/0” numeric placeholder into a clear binary signal of past engagement, reducing noise and letting all models pick up “ever contacted” vs. “never” cleanly.

Ternary Conversion for Loans/Default

Mapping {“yes”→1, “no”→0, “unknown”→–1} preserves informational order (positive, negative, missing) in a compact form. LR treats it continuously, trees can split on “–1” vs. “≥0,” and XGBoost can even enforce monotonic relations if desired.

Scaling for Continuous Features

StandardScaler on Age & Campaign Calls: Centers and scales to unit variance, which is crucial for LR (to ensure coefficients are comparable) and beneficial for any gradient-based algorithm; not strictly required for tree models but doesn’t hurt.

Model-Specific Encoding

Logistic Regression → One-Hot: Avoids imposing an arbitrary ordering on nominal categories and keeps LR’s linear assumption valid.

Random Forest / XGBoost → Label-Encode: Yields a single integer per category, minimizing feature-space inflation and speeding up split-finding in large trees or ensembles.

Overall Benefits

Compact, Informative Features: You avoid high-dimensional one-hot bloat where it’s unnecessary, while still giving Logistic Regression the right inputs.

Reduced Noise: Binary/ternary flags and sensible imputation reduce the risk of skewing model thresholds or coefficients.

Consistent Pipelines: A unified mapping/encoding scheme makes training vs. inference deterministic and easier to debug.

Flexibility for Non-Linear Effects: Trees and boosting can still learn complex interactions on top of these engineered features without extra work.

#### Feature-processing summary

| Feature                  | Treatment                       | Rationale                            |
| ------------------------ | ------------------------------- | ------------------------------------ |
| Education Level          | Mapped to years (6–17)          | Quantitative input for LR & trees    |
| Previous Contact Days    | Flag = (days != 999 & days > 0) | Clear binary “ever contacted” signal |
| Housing/Personal Loan    | {ye s→ 1, no → 0, unknown → –1} | Preserve missing info ordinally      |
| Age, Campaign Calls      | StandardScaler                  | Normalize for LR & gradient methods  |
| Categorical (LR only)    | One-Hot Encoding                | Avoid arbitrary ordering             |
| Categorical (Trees only) | Label Encoding                  | Compact, faster split searches       |

### Pipeline Overview

| Step                                       | File                 | Description                                                                                                                                              |
| ------------------------------------------ | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Data Loading**                        | `src/load_data.py`   | Connects to `bmarket.db` and loads the _bank_marketing_ table into a Pandas DataFrame.                                                                   |
| **2. Preprocessing & Feature Engineering** | `src/preprocess.py`  | Cleans/imputes fields, maps education → years, creates contact flags, encodes loans/defaults, scales numerics, and applies one-hot or label encoding.    |
| **3. Model Training**                      | `src/train_model.py` | Splits data, applies SMOTE-Tomek balancing, reads hyperparameter grids from `config.json`, runs `GridSearchCV`, and fits both baseline and tuned models. |
| **4. Evaluation & Thresholding**           | `src/evaluate.py`    | Generates precision–recall curves, selects F1-optimal thresholds, prints classification reports and ROC-AUC scores.                                      |
| **5. Orchestration**                       | `src/main.py`        | Parses CLI args (`logreg`, `rf`, `xgb`), then loads → preprocesses → trains → evaluates in order, reporting completion.                                  |
| **6. Configuration**                       | `src/config.py`      | Defines hyperparameter grids for each model for `GridSearchCV`.                                                                                          |
| **7. Run Script**                          | `run.sh`             | User entrypoint: `./run.sh <model>` to run one model or `./run.sh all` to iterate through all three.                                                     |

To execute the pipeline:

1. Create a virtual environment with python version 3.11 inside the folder
2. Activate the virtual environment example: source venv/bin/activate
3. download libraries using the requirements file
4. Add in the data folder with bmarket.db into the folder
5. Can adjust and change the hyperparameter in the config.py
6. Use GitBash terminal and run example: ./run.sh all to see all the models, or can run each of the model example: ./run.sh logreg

ML model: Used SMOTETomek. SMOTE alone underperformed, SMOTETomek gave the best recall and class balance.

#### Choice of Models

Because our task is to predict a relatively rare event (term‐deposit subscription) from a mix of numerical and categorical features—and because interpretability, training speed, and handling of class imbalance all matter —Thus selected these mdoels:

1. **Logistic Regression**

   - **Baseline & Interpretability**: Provides a simple linear boundary and easily interpretable feature coefficients, which helps stakeholders understand which customer attributes (e.g., age, previous contact flag) drive subscription likelihood.
   - **Efficiency**: Trains quickly on one-hot encoded features, making it ideal for rapid prototyping and threshold tuning.
   - **Robustness**: With class-weight adjustment, it handles imbalance without synthetic sampling.

2. **Random Forest**

   - **Non-Linear Relationships**: Captures complex interactions (e.g., education × loan status) that a linear model can miss.
   - **Built-In Imbalance Handling**: The `class_weight='balanced_subsample'` option automatically reweights trees to focus more on minority subscribers.
   - **Feature Importance**: Ranks predictors by importance, guiding future feature engineering.

3. **XGBoost**
   - **Regularization & Speed**: L1/L2 penalties reduce overfitting on noisy campaign metrics; tree-based boosting converges quickly on large datasets.
   - **Fine-Grained Control**: Hyperparameters like learning rate and subsample let us balance precision vs. recall.
   - **State-of-the-Art Performance**: Often achieves the highest F1 and ROC-AUC by sequentially correcting earlier errors, which is crucial for capturing the minority “yes” class.

By combining interpretability (Logistic Regression), robustness to non-linearities (Random Forest), and high-performance boosting (XGBoost), we cover a spectrum of strengths and ensure reliable subscription predictions.

Each model ships with default hyperparameters (e.g. C=1.0 for Logistic Regression, n_estimators=100 for Random Forest). To tailor performance to our imbalanced term-deposit dataset, customized hyperparameter grids in config.py and invoked GridSearchCV with scoring='f1'. This tuning process systematically searches for the combination of settings that maximizes F1 score, yielding models better aligned with the business goal of balancing true-positive capture against false-positive waste.

#### Classification reports

Evaluating Original Model...
Original Model - Best Threshold: 0.6603
Original Model - Best F1 Score: 0.3554
Original Model Classification Report:
precision recall f1-score support

           0     0.9110    0.9669    0.9381      5886
           1     0.5149    0.2713    0.3554       763

    accuracy                         0.8871      6649

macro avg 0.7130 0.6191 0.6467 6649
weighted avg 0.8655 0.8871 0.8712 6649

Original Model ROC AUC Score: 0.7193655202450759

Evaluating Best Model from GridSearchCV...
Best Model - Best Threshold: 0.6525
Best Model - Best F1 Score: 0.3582

[GridSearchCV] Best parameters for 'logreg': {'C': 0.01, 'class_weight': None, 'penalty': 'l1', 'solver': 'saga'}
Best Model Classification Report:
precision recall f1-score support

           0     0.9109    0.9708    0.9399      5886
           1     0.5426    0.2674    0.3582       763

    accuracy                         0.8901      6649

macro avg 0.7267 0.6191 0.6490 6649
weighted avg 0.8686 0.8901 0.8731 6649

Best Model ROC AUC Score: 0.7188986995821438
Pipeline completed for model: logreg

Evaluating Original Model...
Original Model - Best Threshold: 0.2843
Original Model - Best F1 Score: 0.3184
Original Model Classification Report:
precision recall f1-score support

           0     0.9207    0.8165    0.8655      5886
           1     0.2442    0.4574    0.3184       763

    accuracy                         0.7753      6649

macro avg 0.5825 0.6370 0.5920 6649
weighted avg 0.8431 0.7753 0.8027 6649

Original Model ROC AUC Score: 0.6797448818953743

Evaluating Best Model from GridSearchCV...
Best Model - Best Threshold: 0.3684
Best Model - Best F1 Score: 0.3082

[GridSearchCV] Best parameters for 'rf': {'class_weight': 'balanced_subsample', 'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
Best Model Classification Report:
precision recall f1-score support

           0     0.9176    0.8247    0.8686      5886
           1     0.2406    0.4286    0.3082       763

    accuracy                         0.7792      6649

macro avg 0.5791 0.6266 0.5884 6649
weighted avg 0.8399 0.7792 0.8043 6649

Best Model ROC AUC Score: 0.6780428179089908
Pipeline completed for model: rf
Finished: rf

Evaluating Original Model...
Original Model - Best Threshold: 0.4278
Original Model - Best F1 Score: 0.3636
Original Model Classification Report:
precision recall f1-score support

           0     0.9159    0.9309    0.9233      5886
           1     0.3898    0.3408    0.3636       763

    accuracy                         0.8631      6649

macro avg 0.6529 0.6358 0.6435 6649
weighted avg 0.8555 0.8631 0.8591 6649

Original Model ROC AUC Score: 0.7014453070551041

Evaluating Best Model from GridSearchCV...
Best Model - Best Threshold: 0.5244
Best Model - Best F1 Score: 0.3792

[GridSearchCV] Best parameters for 'xgb': {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.8}
Best Model Classification Report:
precision recall f1-score support

           0     0.9146    0.9575    0.9356      5886
           1     0.4867    0.3106    0.3792       763

    accuracy                         0.8833      6649

macro avg 0.7006 0.6341 0.6574 6649
weighted avg 0.8655 0.8833 0.8717 6649

Best Model ROC AUC Score: 0.7117428609727238
Pipeline completed for model: xgb
Finished: xgb

**Key Takeaways:**

- **Recall vs. F1 trade-off:** Random Forest maximizes recall but sacrifices precision, while XGBoost strikes the best balance (highest F1).
- **Overall discrimination:** Logistic Regression edges out slightly in ROC-AUC, but XGBoost’s balanced F1 makes it the top performer for our imbalanced target.

#### Metrics used for Evaluation

We evaluated each model using four complementary metrics to capture both classification accuracy and ranking quality, then prioritized Recall and ROC-AUC to align with our business goals:

1. **Precision**

   - **Definition:** Of all the customers predicted to subscribe, the percentage who actually did.
   - **Why it matters:** High precision means fewer wasted calls on non-subscribers, reducing operational cost per call.

2. **Recall**

   - **Definition:** Of all true subscribers, the percentage our model correctly flags.
   - **Why it matters:** Each missed subscriber is a lost revenue opportunity. Maximizing recall ensures we capture as many “yes” leads as possible—even if it means a few extra false positives.

3. **F1 Score**

   - **Definition:** The harmonic mean of precision and recall.
   - **Why it matters:** Balances the trade-off between over-calling (false positives) and under-calling (false negatives). In our imbalanced scenario, F1 gives a single measure of overall effectiveness.

4. **ROC-AUC**
   - **Definition:** Area under the Receiver Operating Characteristic curve, summarizing how well the model ranks subscribers vs. non-subscribers across all thresholds.
   - **Why it matters:** A high AUC means marketing can confidently prioritize highest-scoring leads, even if we adjust our calling threshold over time.

---

##### Why F1 Score Is Our Top Priority

- **Balanced Call Efficiency & Coverage:**  
  In our imbalanced setting, neither extreme precision nor extreme recall alone captures the business need. F1 score balances both—ensuring we don’t flood the call centre with too many unlikely leads (precision) while still catching the bulk of true subscribers (recall).

- **Cost–Benefit Alignment:**  
  Each call incurs a cost. A model tuned for maximum recall might flag too many false positives, wasting resources, whereas a model tuned solely for precision might miss valuable subscribers. Optimizing F1 maintains an optimal trade-off, maximizing net revenue per call.

- **EDA-Driven Insights:**  
  Our exploratory analysis showed that subscribers are scattered across overlapping customer segments rather than concentrated in a single group. This means some “sure-bet” leads coexist with harder-to-spot prospects. F1 score rewards a model that performs well on both the obvious and subtle cases, rather than over-fitting to only one cohort.

- **Threshold Robustness:**  
  By targeting F1, we select a probability cutoff that jointly maximizes precision and recall. This leads to a stable operating point where small shifts in call-centre capacity or campaign budget do not disproportionately degrade performance.

- **Holistic Performance Metric:**  
  Unlike ROC-AUC—which focuses on ranking—and recall—which focuses on coverage—F1 gives a single, actionable number reflecting real-world trade-offs. It directly answers the question: “What proportion of our calls will reach actual subscribers, while still finding as many subscribers as possible?”

In summary, optimizing for F1 score ensures our model delivers both efficient and comprehensive subscriber outreach, aligning precisely with our dual goals of cost control and revenue growth.
