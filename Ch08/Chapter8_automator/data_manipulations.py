"""
    data_manipulations.py

    The code here is used to perform data manipulation. The following modules are available

    1. Missing value calculation
    2. Identify the data type for each variables
    3. Convert Categorical to Numerical using Label encoders
    4. Impute Numerical columns with a specific value. The default is set to 0.
    5. Rename columns
    6. Join X and Y vector using a monotonically increasing row id
    7. Train, Valid and Test data creator
    8. Assembling vectors
    9. Scale Input variables
"""


from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.ml.feature import IndexToString
from pyspark.sql.functions import col
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler

#    1. Missing value calculation

def missing_value_calculation(X, miss_per=0.75):

    missing = X.select([F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c) for c in X.columns])
    missing_len = X.count()
    final_missing = missing.toPandas().transpose()
    final_missing.reset_index(inplace=True)
    final_missing.rename(columns={0:'missing_count'},inplace=True)
    final_missing['missing_percentage'] = final_missing['missing_count']/missing_len
    vars_selected = final_missing['index'][final_missing['missing_percentage'] <= miss_per]
    return vars_selected

#    2. Identify the data type for each variables

def identify_variable_type(X):

    l = X.dtypes
    char_vars = []
    num_vars = []
    for i in l:
        if i[1] in ('string'):
            char_vars.append(i[0])
        else:
            num_vars.append(i[0])
    return char_vars, num_vars

#    3. Convert Categorical to Numerical using Label encoders

def categorical_to_index(X, char_vars):
    chars = X.select(char_vars)
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index",handleInvalid="keep") for column in chars.columns]
    pipeline = Pipeline(stages=indexers)
    char_labels = pipeline.fit(chars)
    X = char_labels.transform(X)
    return X, char_labels

#    4. Impute Numerical columns with a specific value. The default is set to 0.

def numerical_imputation(X,num_vars, impute_with=0):
    X = X.fillna(impute_with,subset=num_vars)
    return X

#    5. Rename columns

def rename_columns(X, char_vars):
    mapping = dict(zip([i+ '_index' for i in char_vars], char_vars))
    X = X.select([col(c).alias(mapping.get(c, c)) for c in X.columns])
    return X

#    6. Join X and Y vector using a monotonically increasing row id

def join_features_and_target(X, Y):

    X = X.withColumn('id', F.monotonically_increasing_id())
    Y = Y.withColumn('id', F.monotonically_increasing_id())
    joinedDF = X.join(Y,'id','inner')
    joinedDF = joinedDF.drop('id')
    return joinedDF

#    7. Train, Valid and Test data creator

def train_valid_test_split(df, train_size=0.4, valid_size=0.3,seed=12345):

    train, valid, test = df.randomSplit([train_size, valid_size,1-train_size-valid_size], seed=12345)
    return train,valid,test

#    8. Assembling vectors

def assembled_vectors(train,list_of_features_to_scale,target_column_name):

    stages = []
    assembler = VectorAssembler(inputCols=list_of_features_to_scale, outputCol='features')
    stages=[assembler]
    selectedCols = [target_column_name,'features'] + list_of_features_to_scale

    pipeline = Pipeline(stages=stages)
    assembleModel = pipeline.fit(train)

    train = assembleModel.transform(train).select(selectedCols)
    return train

#    9. Scale Input variables

def scaled_dataframes(train,valid,test,list_of_features_to_scale,target_column_name):

    stages = []
    assembler = VectorAssembler(inputCols=list_of_features_to_scale, outputCol='assembled_features')
    scaler = StandardScaler(inputCol=assembler.getOutputCol(), outputCol='features')
    stages=[assembler,scaler]
    selectedCols = [target_column_name,'features'] + list_of_features_to_scale

    pipeline = Pipeline(stages=stages)
    pipelineModel = pipeline.fit(train)

    train = pipelineModel.transform(train).select(selectedCols)
    valid = pipelineModel.transform(valid).select(selectedCols)
    test = pipelineModel.transform(test).select(selectedCols)

    return train, valid, test, pipelineModel
