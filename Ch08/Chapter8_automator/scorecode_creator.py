"""
    scorecode_creator.py

Create pseudo score code for production deployment using this module. It links to all your model objects created during the modeling process and links them in one place. If you plan to use this file, then change the "score_table" variable to point to your input data.
"""

#Import the scoring features
import string

import_packages = """
#This is a pseudo score code for production deployment. It links to all your model objects created during the modeling process. If you plan to use this file, then change the "score_table" variable to point to your input data. Double check the "home_path" and "hdfs_path", if you altered the location of model objects.
import os
os.chdir('/home/jovyan/work/spark-warehouse/auto_model_builder')
from pyspark import SparkContext,HiveContext,Row,SparkConf
from pyspark.sql import *
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors,VectorUDT
from pyspark.sql.functions import *
from pyspark.mllib.stat import *
from pyspark.ml.feature import *
from pyspark.ml.feature import IndexToString,StringIndexer,VectorIndexer
from sklearn.metrics import roc_curve,auc
import numpy as np
import pandas as pd
import subprocess
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline,PipelineModel
from pyspark.sql import functions as func
from datetime import *
from pyspark.sql import SparkSession,SQLContext
from pyspark.sql.types import *
from dateutil.relativedelta import relativedelta
from data_manipulations import *
from model_builder import *
import datetime
from datetime import date
import string
import os
import sys
import time
import numpy
spark = SparkSession.builder.appName("MLA_Automated_Scorecode").enableHiveSupport().getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
sc = spark.sparkContext
"""

parameters = string.Template("""
user_id = '${user_id}'
mdl_output_id = '${mdl_output_id}'
mdl_ltrl = '${mdl_ltrl}'
#Since the hdfs and home path below are pointing to your user_id by default, to use this file for scoring, you need to upload the model objects in hdfs_path and home_path to the appropriate score location path (Could be advanl or any other folder path). You would need the following files to perform scoring.
#hdfs_path  - all the files in the path specified below
#home_path - 'model_scoring_info.z'
hdfs_path = '/user/${user_id}' + '/' + 'mla_${mdl_ltrl}' #update score location hdfs_path
home_path = '/home/${user_id}' + '/' + 'mla_${mdl_ltrl}' #update score location home_path
""")

import_variables = """
from sklearn.externals import joblib
from pyspark.ml import Pipeline,PipelineModel
final_vars,id_vars,vars_selected,char_vars,num_vars,impute_with,selected_model,dev_table_name = joblib.load(home_path + '/model_scoring_info.z')
char_labels = PipelineModel.load(hdfs_path + '/char_label_model.h5')
pipelineModel = PipelineModel.load(hdfs_path + '/pipelineModel.h5')
"""

load_models = """
KerasModel = ''
loader_model_list = [LogisticRegressionModel, RandomForestClassificationModel, GBTClassificationModel, DecisionTreeClassificationModel, MultilayerPerceptronClassificationModel, KerasModel]
models_to_run = ['logistic', 'randomForest','gradientBoosting','decisionTree','neuralNetwork','keras']
load_model = loader_model_list[models_to_run.index(selected_model)]
model = load_model.load(hdfs_path + '/' + selected_model + '_model.h5')
"""

score_function = """
score_table = spark.sql("select " + ", ".join(final_vars) + " from " + dev_table_name) #update this query appropriately
def score_new_df(scoredf, model):
    newX = scoredf.select(final_vars)
    newX = newX.select(list(vars_selected))
    newX = char_labels.transform(newX)
    newX = numerical_imputation(newX,num_vars, impute_with)
    newX = newX.select([c for c in newX.columns if c not in char_vars])
    newX = rename_columns(newX, char_vars)
    finalscoreDF = pipelineModel.transform(newX)
    finalscoreDF.cache()
    finalpredictedDF = model.transform(finalscoreDF)
    finalpredictedDF.cache()
    return finalpredictedDF
ScoredDF = score_new_df(score_table, model)
"""

def selected_model_scorecode(user_id, mdl_output_id, mdl_ltrl, parameters):

    parameters = parameters.substitute(locals())
    scorefile = open('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/score_code_selected_model.py', 'w')
    scorefile.write(import_packages)
    scorefile.write(parameters)
    scorefile.write(import_variables)
    scorefile.write(load_models)
    scorefile.write(score_function)
    scorefile.close()
    print('Score code generation complete')


# # Generate individual score codes

# In[24]:

def individual_model_scorecode(user_id, mdl_output_id, mdl_ltrl, parameters):

    loader_model_list = ['LogisticRegressionModel', 'RandomForestClassificationModel', 'GBTClassificationModel', 'DecisionTreeClassificationModel', 'MultilayerPerceptronClassificationModel', 'KerasModel']
    models_to_run = ['logistic', 'randomForest','gradientBoosting','decisionTree','neuralNetwork','keras']

    parameters = parameters.substitute(locals())
    for i in models_to_run:
        try:
            load_model = loader_model_list[models_to_run.index(i)]

            write_model_parameter = string.Template("""
model = ${load_model}.load(hdfs_path + '/' + ${i} + '_model.h5')
            """).substitute(locals())

            scorefile = open('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/' + str(i[0].upper()) + str(i[1:]) + '/score_code_' + i + '_model.py', 'w')
            scorefile.write(import_packages)
            scorefile.write(parameters)
            scorefile.write(import_variables)
            scorefile.write(write_model_parameter)
            scorefile.write(score_function)
            scorefile.close()
        except:
            pass

    print('Individual Score code generation complete')
