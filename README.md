# Breast Cancer Risk Prediction with Fairness Analysis
### Overview
This project aims to build predictive models for breast cancer risk using the BCSC dataset, focusing not only on model accuracy but also on fairness across racial and ethnic groups. Breast cancer risk prediction can be life-saving, but models must be equitable to avoid perpetuating health disparities. This project includes:

Data preprocessing and feature engineering

Model training with Logistic Regression and Random Forest

Hyperparameter tuning and performance evaluation

Fairness auditing through subgroup stratification by race/ethnicity

Application of fairness-aware algorithms to reduce bias

Error analysis of false positives and false negatives by group


### Main Question:
How fair are existing breast cancer risk prediction models across racial and ethnic subgroups?

### Sub-Questions:

Do model accuracy metrics (like AUC) vary by race and ethnicity?

Are certain groups more likely to receive false negatives?

Can reweighting or fairness-aware algorithms reduce disparities?
This work includes:


### Dataset
The dataset is sourced from the Breast Cancer Surveillance Consortium (BCSC) and contains:

Patient demographic features including race_eth

Clinical features relevant to breast cancer risk (e.g., age at menarche, breast density)

Breast cancer history (binary target variable)

##### Note: Entries with unknown breast cancer history or race (race_eth = 9) are filtered out during modeling to ensure data quality.

### Project Structure
data/
Contains raw and cleaned datasets.

notebooks/
Jupyter notebooks for data exploration, preprocessing, model training, tuning, and fairness evaluation.

scripts/
Standalone scripts for training and evaluation pipelines.

results/
Model outputs, classification reports, fairness metrics, and plots.

### Methodology
Data Preprocessing
Filter out samples with unknown breast cancer history.

Select important features based on clinical relevance.

One-hot encode categorical variables, especially race/ethnicity.

Split data into training (80%) and test (20%) sets with stratification to preserve class balance.

Model Training
Train Logistic Regression and Random Forest classifiers.

Hyperparameter tuning with GridSearchCV to optimize F1 score for minority class (positive breast cancer history).

Evaluate using accuracy, precision, recall, F1 score, and ROC-AUC.

### Fairness Analysis
Stratify model performance metrics by race_eth to detect disparities.

Analyze false negatives and false positives by racial group to identify vulnerable populations.

Visualize subgroup metrics for clear comparison.

### Fairness Mitigation
Implement fairness-aware algorithms such as Exponentiated Gradient Reduction to explicitly enforce fairness constraints during training.

Adjust model predictions post hoc to balance error rates across groups.

Evaluate trade-offs between accuracy and fairness.

Usage
Setup
Clone the repository.

Install dependencies:
pip install -r requirements.txt
Place the bcsc_cleaned.csv dataset in the data/ folder.

Run Modeling Pipeline
Preprocess data and generate train-test splits.

Train baseline models:
python scripts/train_baseline_models.py
Perform hyperparameter tuning:
python scripts/hyperparameter_tuning.py
Evaluate model fairness:
python scripts/fairness_evaluation.py
Train fairness-aware model and apply mitigation:
python scripts/fairness_mitigation.py

### Results Summary
Baseline Random Forest achieves ROC-AUC around 0.81 with reasonable recall for positive cases.

Significant disparities found in recall and false negative rates across racial groups, with some minorities underdiagnosed.

Fairness-aware training reduces disparities, especially in false negative rates, with acceptable impact on overall accuracy.

Post-processing thresholds show promise for further refinement of equitable predictions.

### Future Work
Incorporate additional socio-economic features (e.g., income level) to broaden fairness assessment.

Experiment with other fairness algorithms and metrics such as demographic parity or equal opportunity.

Deploy models in simulated clinical workflows to assess real-world impact.

Expand dataset to include longitudinal patient outcomes for better risk calibration.

### References
Breast Cancer Surveillance Consortium (BCSC) Dataset

Fairlearn: A toolkit for assessing and improving fairness in AI systems

Relevant literature on health equity and predictive modeling

### Contact
For questions or collaboration, please reach out to:

Kehinde Soetan
Graduate Student, Medical Humanities
[kehindesoetan3@gmail.com]
[Linkedin: Kehinde Soetan/Keehinde678]

