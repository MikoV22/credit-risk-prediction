# Credit Risk Prediction

A machine learning project that predicts whether a bank client will repay their loan, 
based on their personal and financial data.

## What this project does

This project builds and compares two machine learning models (Logistic Regression 
and Random Forest) to predict credit risk - whether someone is likely to repay 
or default on a loan. It includes data exploration, data cleaning and preparation, 
model training, hyperparameter tuning, and an additional SQL-based analysis.

This was built as a learning project to practice Python, machine learning, and SQL.

## Dataset

This project uses the [German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) 
dataset from the UCI Machine Learning Repository. It contains 1000 loan applicants, 
each described by 20 features (age, credit amount, employment status, savings, 
loan purpose, etc.) and a target variable showing whether they repaid the loan.

## Tech Stack

- **Python** - pandas, numpy, scikit-learn, matplotlib, seaborn
- **SQL** - SQLite, for data exploration and querying
- **Jupyter Notebook** - for analysis and experimentation

## Project Structure

```
credit-risk-prediction/
  data/
    raw/
      german_credit/
        german.data          (original dataset)
    processed/
      german_credit/
        credit_risk.db       (SQLite database)
  notebooks/
    01_eda_german.ipynb
    02_preprocessing_modeling_german.ipynb
    03_sql_analysis_german.ipynb
    04_final_pipeline_german.ipynb
  src/
    data_prep.py             (data loading and preprocessing)
    train.py                  (model training)
    evaluate.py                (model evaluation)
  requirements.txt
  README.md
```

## Methodology

1. **Exploratory Data Analysis (EDA)** - checked data shape, missing values, 
   duplicates, class balance, and distributions of key features (age, credit amount).

2. **Preprocessing** - encoded categorical variables (One-Hot Encoding), split data 
   into train/test sets (80/20, stratified), and scaled numerical features 
   (StandardScaler).

3. **Model Training** - trained two models:
   - Logistic Regression (baseline)
   - Random Forest, tuned using GridSearchCV (5-fold cross-validation)

4. **Evaluation** - compared models using Accuracy, Precision, Recall, F1-score, 
   and ROC-AUC, since the dataset has imbalanced classes (~70% repaid, ~30% defaulted).

5. **SQL Analysis** - loaded the data into a SQLite database and ran SQL queries 
   to cross-check the EDA findings and explore additional patterns.

## Results

| Metric | Logistic Regression | Random Forest (tuned) |
|---|---|---|
| Accuracy | 0.740 | 0.720 |
| Precision | 0.547 | 0.524 |
| Recall | 0.783 | 0.717 |
| F1-score | 0.644 | 0.606 |
| ROC-AUC | 0.807 | 0.801 |

Logistic Regression performed slightly better than Random Forest, even after 
hyperparameter tuning. This is likely because the dataset is small (800 training 
samples) and the relationships between features and the target are mostly linear.

## Key Findings

- Logistic Regression slightly outperformed Random Forest, even after hyperparameter 
  tuning - showing that a simpler model can sometimes be just as good as (or better 
  than) a more complex one.
- The dataset has imbalanced classes (~70% repaid, ~30% defaulted), so metrics 
  beyond accuracy (like ROC-AUC and recall) were essential to properly evaluate 
  the models.
- High recall (0.783) was prioritized over precision, since in credit risk, 
  missing a risky client is usually more costly than rejecting a good one.

## How to Run

1. Clone this repository
2. Create a virtual environment and activate it
3. Install dependencies: `pip install -r requirements.txt`
4. Open the notebooks in the `notebooks/` folder, in order (01 to 04)

## Possible Extensions

- Add XGBoost/LightGBM for comparison
- Extend the project with the larger, multi-table Home Credit Default Risk dataset
- Build a simple web app (e.g. Streamlit) to compare model predictions interactively

## Note

Code comments and notebook contents are in Polish, as this project was built as 
a personal learning exercise. This README provides a full summary in English.