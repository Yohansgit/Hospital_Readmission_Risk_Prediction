# Model Training and Evaluation Script
import os
import sys
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, IntegerType, FloatType
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler

# Set environment variables
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-17.0.12"
os.environ["PATH"] = os.environ["JAVA_HOME"] + r"\bin;" + os.environ["PATH"]
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Initialize Spark
spark = SparkSession.builder \
    .appName("DiabeticReadmission") \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.hadoop.security.authentication", "simple") \
    .config("spark.hadoop.security.authorization", "false") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .getOrCreate()

print("Spark session ready!")

# Load data
data_path = r"C:\Projects\hospital_readmission_prediction\output\cleaned_diabetic_data\ml_ready_data.csv"
df_pandas = pd.read_csv(data_path)
df = spark.createDataFrame(df_pandas)
print(f"Data loaded: {df.count()} rows")

# Prepare features and target
if 'features' in df.columns and 'readmitted' in df.columns:
    from pyspark.ml.linalg import Vectors, VectorUDT
    from pyspark.sql.functions import udf

    def fast_vector_parse(vec_str):
        if not vec_str or vec_str == '[]':
            return None
        try:
            cleaned = vec_str[1:-1]
            values = np.fromstring(cleaned, sep=',', dtype=np.float64)
            return Vectors.dense(values.tolist())
        except:
            return None

    vector_udf = udf(fast_vector_parse, VectorUDT())
    df_ml = df.withColumn("features_vec", vector_udf("features")).filter("features_vec is not null")
    df = df_ml.select("features_vec", "readmitted").withColumnRenamed("features_vec", "features").cache()
else:
    raise ValueError("Required columns 'features' or 'readmitted' not found in df.")

# Split dataset
train_data, test_data = df.randomSplit([0.8, 0.2], seed=1)

# Helper function for training and evaluating ML models
def train_and_evaluate(model, train_df, test_df, model_name):
    fitted_model = model.fit(train_df)
    train_pred = fitted_model.transform(train_df)
    test_pred = fitted_model.transform(test_df)
    acc_evaluator = MulticlassClassificationEvaluator(labelCol="readmitted", predictionCol="prediction", metricName="accuracy")
    f1_evaluator = MulticlassClassificationEvaluator(labelCol="readmitted", predictionCol="prediction", metricName="f1")
    precision_evaluator = MulticlassClassificationEvaluator(labelCol="readmitted", predictionCol="prediction", metricName="weightedPrecision")
    recall_evaluator = MulticlassClassificationEvaluator(labelCol="readmitted", predictionCol="prediction", metricName="weightedRecall")
    roc_evaluator = BinaryClassificationEvaluator(labelCol="readmitted", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    metrics = {
        "Model_Name": model_name,
        "Train_Accuracy": acc_evaluator.evaluate(train_pred),
        "Test_Accuracy": acc_evaluator.evaluate(test_pred),
        "Train_f1": f1_evaluator.evaluate(train_pred),
        "Train_precision": precision_evaluator.evaluate(train_pred),
        "Train_recall": recall_evaluator.evaluate(train_pred),
        "Train_auc_roc": roc_evaluator.evaluate(train_pred),
        "Test_f1": f1_evaluator.evaluate(test_pred),
        "Test_precision": precision_evaluator.evaluate(test_pred),
        "Test_recall": recall_evaluator.evaluate(test_pred),
        "Test_auc_roc": roc_evaluator.evaluate(test_pred),
    }
    return metrics

# Example: Train Logistic Regression
lr_base = LogisticRegression(featuresCol="features", labelCol="readmitted", maxIter=50)
lr_metrics = train_and_evaluate(lr_base, train_data, test_data, "LogisticRegression")
print("Logistic Regression Metrics:")
print(lr_metrics)

# Save model metrics to CSV
scores_pd = pd.DataFrame([lr_metrics])
model_dir = r"C:\Projects\hospital_readmission_prediction\model"
os.makedirs(model_dir, exist_ok=True)
csv_path = os.path.join(model_dir, "model_performance.csv")
scores_pd.to_csv(csv_path, index=False)
print(f"Saved CSV: {csv_path}")

# Save trained model (example for Logistic Regression)
model_path = os.path.join(model_dir, "logistic_regression_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(lr_base.fit(train_data), f)
print(f"Saved model: {model_path}")