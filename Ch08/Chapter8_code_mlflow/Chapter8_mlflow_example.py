import pyspark
from pyspark.sql import SparkSession
import mlflow
import mlflow.spark
import sys
import time
# from common import *
# from sparkml_udf_workaround import log_udf_model

spark = SparkSession.builder.appName("mlflow_example").getOrCreate()
# show_versions(spark)

filename = "/home/jovyan/work/bank-full.csv"
target_variable_name = "y"
from pyspark.sql import functions as F
df = spark.read.csv(filename, header=True, inferSchema=True, sep=';')
df = df.withColumn('label', F.when(F.col("y") == 'yes', 1).otherwise(0))
df = df.drop('y')
train, test = df.randomSplit([0.7, 0.3], seed=12345)

for k, v in df.dtypes:
    if v not in ['string']:
        print(k)

df = df.select(['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous', 'label'])


from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

#assemble individual columns to one column - 'features'
def assemble_vectors(df, features_list, target_variable_name):
    stages = []
    #assemble vectors
    assembler = VectorAssembler(inputCols=features_list, outputCol='features')
    stages = [assembler]
    #select all the columns + target + newly created 'features' column
    selectedCols = [target_variable_name, 'features']
    #use pipeline to process sequentially
    pipeline = Pipeline(stages=stages)
    #assembler model
    assembleModel = pipeline.fit(df)
    #apply assembler model on data
    df = assembleModel.transform(df).select(selectedCols)

    return df

#exclude target variable and select all other feature vectors
features_list = df.columns
#features_list = char_vars #this option is used only for ChiSqselector
features_list.remove('label')

# apply the function on our dataframe
assembled_train_df = assemble_vectors(train, features_list, 'label')
assembled_test_df = assemble_vectors(test, features_list, 'label')

print(sys.argv[1])
print(sys.argv[2])
print(sys.argv[3])

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


maxBinsVal = float(sys.argv[1]) if len(sys.argv) > 3 else 20
maxDepthVal = float(sys.argv[2]) if len(sys.argv) > 3 else 3

model_name='model'+'_'+str(int(time.time()))
with mlflow.start_run():
    stages_tree=[]
    classifier = RandomForestClassifier(labelCol = 'label',featuresCol = 'features',maxBins=maxBinsVal, maxDepth=maxDepthVal)
    stages_tree += [classifier]
    pipeline_tree=Pipeline(stages=stages_tree)
    print('Running RFModel')
    RFmodel = pipeline_tree.fit(assembled_train_df)
    print('Completed training RFModel')
    predictions = RFmodel.transform(assembled_test_df)
    evaluator = BinaryClassificationEvaluator()
    print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))

    mlflow.log_param("maxBins", maxBinsVal)
    mlflow.log_param("maxDepth", maxDepthVal)
    mlflow.log_metric("ROC", evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"}))
    print(mlflow.get_artifact_uri())
    mlflow.spark.log_model(RFmodel,"spark-model")
    result = mlflow.register_model(
    "runs:/d16076a3ec534311817565e6527539c0/artifacts/sklearn-model",
    "sk-learn-random-forest-reg")


# Chapter9_mlflow_example.py

# spark-submit --master local[*] /home/jovyan/work/Chapter9_mlflow_example.py --maxBinsVal 32 --maxDepthVal 5

#
# mlflow server --default-artifact-root /home/jovyan/work/mlflow/artifacts/ --host 0.0.0.0 &
#
# spark-submit --master local[*] /home/jovyan/work/Chapter9_mlflow_example.py 32 5 model_run  --spark_autolog True
#
# sudo -U postgres -i
#
#
# docker run -it -u root -p 5000:5000 -v /Users/ramcharankakarla/demo_data/:/home/jovyan/work/ jupyter/pyspark-notebook:latest bash
# pip install mlflow
#
# sudo apt-get update
#
# sudo apt-get install postgresql postgresql-contrib postgresql-server-dev-all
# sudo apt install gcc
# pip install psycopg2
#
# pg_ctlcluster 12 main start
#
# sudo -u postgres -i
# createuser --interactive -P
# createdb -O mlflow_user mlflow_db
# exit


#
#
# mlflow server --backend-store-uri postgresql://mlflow_user:mlflow@0.0.0.0/mlflow_db --default-artifact-root file:/home/jovyan/artifact_root \
# --host 0.0.0.0 \
# --port 5000 &
#
# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root /home/jovyan/work/mlruns/0 \
# --host 0.0.0.0 \
# --port 5000 &
#
#
#
# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root /home/jovyan/work --host 0.0.0.0 --port 5000 &
# MLFLOW_TRACKING_URI="http://0.0.0.0:5000" spark-submit --master local[*] /home/jovyan/work/Chapter9_mlflow_example.py 16 5  --spark_autolog True
