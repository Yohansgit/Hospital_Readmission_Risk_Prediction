# Diabetic Patient Readmission Risk Prediction

## 📋 Project Overview
This project developed a machine learning solution to predict diabetic patient readmission risk based on healthcare input data. The implementation leverages **PySpark** for distributed data processing and **Power BI** for interactive visualization and business intelligence.

### 🎯 Business Problem
The high frequency of unplanned readmissions for diabetic patients within 30 days of discharge drives up healthcare costs, hinders operational efficiency, and serves as a key benchmark for assessing hospital care quality. In the US, preventable readmission costs **$26B+ annually** and our healthcare 30-day readmission rate is **18%**, above the national average.

Multiple models (logistic regression, tree-based, and SVM) were developed and evaluated on performance metrics (accuracy, precision, recall, F1 score, ROC AUC) and explainability (SHAP and permutation importance).

## 🎯 Project Scope
Predicting which diabetic patients are most likely to be readmitted within 30 days enables proactive care (e.g., follow-up calls, home care visits) and identifies effective treatment strategies.

### Key Objectives:
- **Pinpoint** key clinical and demographic factors driving high readmission rates
- **Evaluate** the impact of specific medications and treatment patterns on patient outcomes
- **Empower** hospital teams to prioritize high-risk patients for early intervention
- **Deploy** a robust, scalable machine learning pipeline that integrates with existing data infrastructure

## 📊 Dataset
The dataset comprises 10 years of clinical records for over 100,000 diabetic patients sourced from the Diabetes 130-US Hospitals dataset.

**Dataset Statistics:**
- **101,766** encounters with **50 features**
- Covers demographics, treatments, and lab results

### Data Features:
| Category | Features |
|----------|----------|
| **Patient Identifiers** | `encounter_id`, `patient_nbr` |
| **Patient Demographics** | `race`, `gender`, `age`, `weight`, `payer_code` |
| **Admission Details** | `admission_source_id`, `admission_type_id`, `discharge_disposition_id` |
| **Medical History** | `number_outpatient`, `number_inpatient`, `number_emergency` |
| **Admission Details** | `time_in_hospital`, `number_diagnoses`, `num_lab_procedures`, `num_procedures`, `medical_specialty`, `diag_1`, `diag_2`, `diag_3` |
| **Clinical Results** | `max_gluc_serum`, `A1cresult` |
| **Medication Details** | `diabetesMed`, `change`, `num_medications`, +23 medication features |
| **Target Variable** | `readmitted`: class variable (0 or 1) |

### Sample Data (First 5 rows)
| encounter_id | patient_nbr | race | gender | age | weight | admission_type_id | discharge_disposition_id | admission_source_id | time_in_hospital | ... | readmitted |
|--------------|-------------|------|--------|-----|--------|------------------|--------------------------|-------------------|------------------|-----|------------|
| 2278393 | 8222157 | Caucasian | Female | [0-10) | NULL | 6 | 25 | 1 | 1 | ... | >30 |
| 149194 | 55629189 | Caucasian | Female | [10-20) | NULL | 1 | 1 | 7 | 3 | ... | >30 |
| 64495 | 86047875 | AfricanAmerican | Female | [20-30) | NULL | 1 | 1 | 7 | 2 | ... | NO |
| 500396 | 82442376 | Caucasian | Male | [30-40) | NLL | 1 | 1 | 7 | 2 | ... | NO |
| 16697 | 42519267 | Caucasian | Male | [40-50) | NULL | 1 | 1 | 7 | 1 | ... | NO |

### Statistical Summary
| Metric |  time_in_hospital | num_lab_procedures | num_procedures | num_medications | number_outpatient | number_emergency | number_inpatient | number_diagnoses |
|--------|------------------|-------------------|---------------|----------------|------------------|-----------------|-----------------|-----------------|
| **Count** |  101,766 | 101,766 | 101,766 | 101,766 | 101,766 | 101,766 | 101,766 | 101,766 |
| **Mean** |  4.40 | 43.10 | 1.34 | 16.02 | 0.37 | 0.20 | 0.64 | 7.42 |
| **Std Dev** |  | 2.99 | 19.67 | 1.71 | 8.13 | 1.27 | 0.93 | 1.26 | 1.93 |
| **Min** |1 | 1 | 0 | 1 | 0 | 0 | 0 | 1 |
| **Max** | 14 | 132 | 6 | 81 | 42 | 76 | 21 | 16 |

## 🏗️ Model Pipeline
Data → PySpark ETL → Feature Engineering → Model Training → Explainability → Dashboard

### Pipeline Components:
1. **Data Processing & ETL** (`01_Data_Ingest_and_ETL_(PySpark).ipynb`)
   - PySpark for distributed ETL, cleaning, and data transformation
   - Class imbalance handled by class weighting

2. **Exploratory Data Analysis & Feature Engineering** (`02_EDA_and_Feature_Engineering_(PySpark).ipynb`)
   - One-hot encoding, scaling, and missing data imputation

3. **Training and Evaluation** (`03_Model_Training_and_Evaluation.ipynb`)
   - LightGBM_Tuned chosen for prediction based on performance and medical relevance

4. **Reproducibility**
   - Fully parameterized pipeline for retraining and version control

## 📈 The Outcome
Final model identifies the **top 10% highest risk patients** with strong predictive performance (`04_Insight_Generation_and_Visualization.ipynb`).

### SHAP Feature Importance
*What influences predictions most?*

| Feature | SHAP Value | What It Means |
|---------|------------|---------------|
| **metformin** | 0.81 | Medication changes signal diabetes management issues |
| **discharge_disposition_id** | 0.79 | Post-discharge destination impacts readmission |
| **time_in_hospital** | 0.57 | Longer stays reflect severe conditions |
| **age** | 0.51 | Older patients more likely to be readmitted |
| **total_visits** | 0.43 | Frequent hospital use related to chronic instability |

**Overall**: Readmission risk is shaped by a mix of clinical severity, care transitions, and patient complexity.

### Permutation Importance
*What the model depends on most?*

| Feature | Permutation Importance | Why It Matters |
|---------|------------------------|----------------|
| **time_in_hospital** | 0.043 | Removing it hurts performance most → key severity indicator |
| **total_visits** | 0.034 | High prior usage strongly predicts risk |
| **age** | 0.031 | Stable demographic predictor |
| **number_diagnoses** | 0.028 | More conditions = higher chance of readmission |
| **discharge_disposition_id** | 0.024 | Post-discharge destination strongly influences risk |

## 🔍 Insight Deep Dive
The model not only predicts readmission but also highlights **why** patients are likely to return. Each driver has a measurable impact on 30-day readmission risk.

### 🎯 Metformin: The Medication Marker
- **Patients on or adjusting metformin are 1.8x more likely to be readmitted**
- **Priority Action**: Implement targeted medication counseling

### 🏥 Discharge Disposition: The Path Home
- **Non-home discharges** (e.g., rehab, SNF) increase readmission risk by **2.3x**
- **Priority Action**: Enhanced post-discharge care coordination

### ⏰ Time in Hospital: The Clock of Severity
- Each additional day in hospital **beyond 5 days** increases readmission risk by **10–15%**
- Stays of **7–10 days** increase risk by **60%**
- **Priority Action**: Targeted discharge planning for long-stay patients

## 💼 Business Impact
- **Reduces readmissions** → lower penalties, better care quality
- **Supports clinicians** with real-time risk flags
- **Scalable architecture** ready for live EMR integration
- **Provides explainability** for trustworthy AI in healthcare

## 🛠️ Installation

### Prerequisites
- Python 3.11
- Java 17
- Spark 3.5
- Pip packages (`requirements.txt`)

### Steps:
1. Download Diabetes 130-US dataset
2. Import PySpark notebooks into Databricks
3. Run ETL → Feature Engineering → Modeling pipeline sequentially
4. Open Power BI file (`Hospital_Readmission_Dashboard.pbix`) to explore results

## 📊 Model Performance
| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 0.88 | Percentage of correct predictions |
| **Precision** | 0.84 | Ratio of true positives to total positive predictions |
| **Recall** | 0.84 | Ratio of true positives to total actual positives |
| **F1 Score** | 0.83 | Harmonic mean of Precision and Recall |
| **ROC AUC** | 0.64 | Area under the ROC curve |

## 📊 Visual Highlights
- **SHAP summary & bar plot** shows top drivers clearly
- **Scatter/impact plots** reveal how risk changes with variables like age or length of stay
- **Readmission distribution visuals** reveal patterns across groups

## 🎯 Project Motivation
This project was developed to:
- Build knowledge in machine learning in healthcare domain
- Gain hands-on experience with model building and interpretation
- Generate impactful insight and business value

## 📄 License
This project is licensed under the **MIT License**.

---


