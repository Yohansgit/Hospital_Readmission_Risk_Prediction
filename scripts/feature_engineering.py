# Feature Engineering Script
import os
import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler

# Initialize Spark
spark = SparkSession.builder \
    .appName("FeatureEngineering") \
    .master("local[*]") \
    .getOrCreate()

print("Spark session ready for feature engineering!")

# Load cleaned data
data_path = r"C:\Projects\hospital_readmission_prediction\output\cleaned_diabetic_data\ml_ready_data.csv"
df_pandas = pd.read_csv(data_path)
df = spark.createDataFrame(df_pandas)

# Example: String Indexing for categorical columns
categorical_cols = [col for col in df.columns if df.schema[col].dataType == 'string' and col != 'readmitted']
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_idx") for col in categorical_cols]

# Example: One-Hot Encoding
encoders = [OneHotEncoder(inputCol=f"{col}_idx", outputCol=f"{col}_vec") for col in categorical_cols]

# Assemble features
feature_cols = [f"{col}_vec" for col in categorical_cols] + [col for col in df.columns if df.schema[col].dataType != 'string' and col != 'readmitted']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Build pipeline
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler])

# Fit and transform
df_fe = pipeline.fit(df).transform(df)

# Save engineered features
output_path = r"C:\Projects\hospital_readmission_prediction\output\cleaned_diabetic_data\ml_ready_data_fe.csv"
df_fe.select("features", "readmitted").toPandas().to_csv(output_path, index=False)
print(f"Feature engineered data saved: {output_path}")
