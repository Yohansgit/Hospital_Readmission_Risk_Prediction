
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

*Last updated: November 2024*
