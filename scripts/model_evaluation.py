import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# Initialize Spark
spark = SparkSession.builder \
    .appName("ModelEvaluation") \
    .master("local[*]") \
    .getOrCreate()

# Load test data
test_data_path = r"C:\Projects\hospital_readmission_prediction\output\cleaned_diabetic_data\ml_ready_data.csv"
df_pandas = pd.read_csv(test_data_path)
df = spark.createDataFrame(df_pandas)

# Vector conversion if needed (assuming 'features' column is a string)
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
import numpy as np

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

# Split dataset
train_data, test_data = df.randomSplit([0.8, 0.2], seed=1)

# Define evaluation function
def evaluate_model(model, train_df, test_df, model_name):
    fitted_model = model.fit(train_df)
    train_pred = fitted_model.transform(train_df)
    test_pred = fitted_model.transform(test_df)
    acc_eval = MulticlassClassificationEvaluator(labelCol="readmitted", predictionCol="prediction", metricName="accuracy")
    f1_eval = MulticlassClassificationEvaluator(labelCol="readmitted", predictionCol="prediction", metricName="f1")
    roc_eval = BinaryClassificationEvaluator(labelCol="readmitted", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    metrics = {
        "Model_Name": model_name,
        "Train_Accuracy": acc_eval.evaluate(train_pred),
        "Test_Accuracy": acc_eval.evaluate(test_pred),
        "Train_f1": f1_eval.evaluate(train_pred),
        "Test_f1": f1_eval.evaluate(test_pred),
        "Train_auc_roc": roc_eval.evaluate(train_pred),
        "Test_auc_roc": roc_eval.evaluate(test_pred),
    }
    print(f"{model_name} metrics:", metrics)
    return metrics

# Example: Evaluate Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="readmitted", maxIter=50)
lr_metrics = evaluate_model(lr, train_data, test_data, "LogisticRegression")

# Example: Evaluate Decision Tree
dt = DecisionTreeClassifier(featuresCol="features", labelCol="readmitted", maxDepth=5)
dt_metrics = evaluate_model(dt, train_data, test_data, "DecisionTree")

# Example: Evaluate Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol="readmitted", numTrees=100, maxDepth=5)
rf_metrics = evaluate_model(rf, train_data, test_data, "RandomForest")

# Example: Evaluate GBT (XGBoost/LightGBM proxy)
gbt = GBTClassifier(featuresCol="features", labelCol="readmitted", maxDepth=5, maxIter=100)
gbt_metrics = evaluate_model(gbt, train_data, test_data, "GBTClassifier")

# Save metrics to CSV
metrics_df = pd.DataFrame([lr_metrics, dt_metrics, rf_metrics, gbt_metrics])
output_dir = r"C:\Projects\hospital_readmission_prediction\model"
os.makedirs(output_dir, exist_ok=True)
metrics_df.to_csv(os.path.join(output_dir, "model_evaluation_metrics.csv"), index=False)
print("Model evaluation metrics saved.")