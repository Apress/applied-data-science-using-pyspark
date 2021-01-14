"""
    build_and_execute_pipe.py

Core Functionality:
1. Performs feature selection
2. Develops machine learning models (5 different algorithms).
3. Validates the models on hold out datasets
4. Picks the best algorithm to deploy based on user selected statistics (ROC, KS, Accuracy)
5. Produces pseudo score code for production deployment
"""


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
import datetime
from datetime import date
import string
import os
import sys
import time
import numpy

spark = SparkSession.builder.appName("Automated_model_building").enableHiveSupport().getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
sc = spark.sparkContext

import_data = False
stop_run = False
message = ''
filename = ''


user_id = 'jovyan'
# email_id = sys.argv[2]
mdl_output_id = 'test_run01' #An unique ID to represent the model
mdl_ltrl = 'chapter8_testrun' #An unique literal or tag to represent the model

input_dev_file='churn_modelling.csv'
input_oot1_file=''
input_oot2_file=''

dev_table_name = ''
oot1_table_name = ''
oot2_table_name = ''

delimiter_type = ','

include_vars = '' # user specified variables to be used
include_prefix = '' # user specified prefixes to be included for modeling
include_suffix = '' # user specified prefixes to be included for modeling
exclude_vars = 'rownumber,customerid,surname' # user specified variables to be excluded for modeling
exclude_prefix = '' # user specified prefixes to be excluded for modeling
exclude_suffix = '' # user specified suffixes to be excluded for modeling

target_column_name = 'exited'

run_logistic_model = 1
run_randomforest_model = 1
run_boosting_model = 1
run_neural_model = 1


miss_per = 0.75
impute_with = 0.0
train_size=0.7
valid_size=0.2 #Train -Test difference is used for test data
seed=2308

model_selection_criteria = 'ks' #possible_values ['ks','roc','accuracy']
dataset_to_use = 'train' #possible_values ['train','valid','test','oot1','oot2']

data_folder_path = '/home/jovyan/work/'
hdfs_folder_path = '/home/jovyan/work/spark-warehouse/'


####################################################################
######No input changes required below this for default run##########
####################################################################



if input_oot1_file=='':
    input_oot1_file=input_dev_file
if input_oot2_file=='':
    input_oot2_file=input_dev_file
# assign input files if the user uploaded files instead of tables.
if dev_table_name.strip() == '':
    dev_input_file = input_dev_file
    if dev_input_file.strip() == '':
        print('Please provide a development table or development file to process the application')
        stop_run = True
        message = 'Development Table or file is not provided. Please provide a development table or file name to process'

    import_data = True
    file_type = dev_table_name.split('.')[-1]
    out,err=subprocess.Popen(['cp',data_folder_path+dev_input_file,hdfs_folder_path],stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()

if oot1_table_name.strip() == '':
    oot1_input_file = input_oot1_file
    out,err=subprocess.Popen(['cp',data_folder_path+oot1_input_file,hdfs_folder_path],stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()

if oot2_table_name.strip() == '':
    oot2_input_file = input_oot2_file
    out,err=subprocess.Popen(['cp',data_folder_path+oot2_input_file,hdfs_folder_path],stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()

ignore_data_type = ['timestamp', 'date']
ignore_vars_based_on_datatype = []

# extract the input variables in the file or table
if not stop_run:
    if import_data:
        df = spark.read.option("delimiter",delimiter_type).option("header", "true").option("inferSchema", "true").csv(hdfs_folder_path + dev_input_file)
        df = pd.DataFrame(zip(*df.dtypes),['col_name', 'data_type']).T
    else:
        df = spark.sql('describe ' + dev_table_name)
        df = df.toPandas()

    input_vars = list(str(x.lower()) for x in df['col_name'])
    print(input_vars)
    for i in ignore_data_type:
        ignore_vars_based_on_datatype += list(str(x) for x in df[df['data_type'] == i]['col_name'])

    if len(ignore_vars_based_on_datatype) > 0:
        input_vars = list(set(input_vars) - set(ignore_vars_based_on_datatype))

    input_vars.remove(target_column_name)


    ## variables to include
    import re
    prefix_include_vars = []
    suffix_include_vars = []

    if include_vars.strip() != '':
        include_vars = re.findall(r'\w+', include_vars.lower())

    if include_prefix.strip() != '':
        prefix_to_include = re.findall(r'\w+', include_prefix.lower())

        for i in prefix_to_include:
            temp = [x for x in input_vars if x.startswith(str(i))]
            prefix_include_vars.append(temp)

        prefix_include_vars = [item for sublist in prefix_include_vars for item in sublist]

    if include_suffix.strip() != '':
        suffix_to_include = re.findall(r'\w+', include_suffix.lower())

        for i in suffix_to_include:
            temp = [x for x in input_vars if x.startswith(str(i))]
            suffix_include_vars.append(temp)

        suffix_include_vars = [item for sublist in suffix_include_vars for item in sublist]

    include_list = list(set(include_vars) | set(prefix_include_vars) | set(suffix_include_vars))

    ## Variables to exclude
    prefix_exclude_vars = []
    suffix_exclude_vars = []

    if exclude_vars.strip() != '':
        exclude_vars = re.findall(r'\w+', exclude_vars.lower())

    if exclude_prefix.strip() != '':
        prefix_to_exclude = re.findall(r'\w+', exclude_prefix.lower())

        for i in prefix_to_exclude:
            temp = [x for x in input_vars if x.startswith(str(i))]
            prefix_exclude_vars.append(temp)

        prefix_exclude_vars = [item for sublist in prefix_exclude_vars for item in sublist]

    if exclude_suffix.strip() != '':
        suffix_to_exclude = re.findall(r'\w+', exclude_suffix.lower())

        for i in suffix_to_exclude:
            temp = [x for x in input_vars if x.startswith(str(i))]
            suffix_exclude_vars.append(temp)

        suffix_exclude_vars = [item for sublist in suffix_exclude_vars for item in sublist]

    exclude_list = list(set(exclude_vars) | set(prefix_exclude_vars) | set(suffix_exclude_vars))




    if len(include_list) > 0:
        input_vars = list(set(input_vars) & set(include_list))

    if len(exclude_list) > 0:
        input_vars = list(set(input_vars) - set(exclude_list))




if not stop_run:

    final_vars = input_vars  # final list of variables to be pulled
    from datetime import datetime
    insertion_date = datetime.now().strftime("%Y-%m-%d")

    import re
    from pyspark.sql.functions import col



    # import data for the modeling
    if import_data:
        train_table = spark.read.option("delimiter",delimiter_type).option("header", "true").option("inferSchema", "true").csv(hdfs_folder_path + dev_input_file)
        oot1_table = spark.read.option("delimiter",delimiter_type).option("header", "true").option("inferSchema", "true").csv(hdfs_folder_path + oot1_input_file)
        oot2_table = spark.read.option("delimiter",delimiter_type).option("header", "true").option("inferSchema", "true").csv(hdfs_folder_path + oot2_input_file)
    else:
        train_table = spark.sql("select " + ", ".join(final_vars + [target_column_name]) + " from " + dev_table_name)
        oot1_table = spark.sql("select " + ", ".join(final_vars + [target_column_name]) + " from " + oot1_table_name)
        oot2_table = spark.sql("select " + ", ".join(final_vars + [target_column_name]) + " from " + oot2_table_name)

    train_table = train_table.where(train_table[target_column_name].isNotNull())
    oot1_table = oot1_table.where(oot1_table[target_column_name].isNotNull())
    oot2_table = oot2_table.where(oot2_table[target_column_name].isNotNull())
    print (final_vars)

    oot1_table=oot1_table.toDF(*[c.lower() for c in oot1_table.columns])
    oot2_table=oot2_table.toDF(*[c.lower() for c in oot2_table.columns])
    print(oot1_table.columns)
    print(oot2_table.columns)
    X_train = train_table.select(*final_vars)
    X_train.cache()

    # apply data manipulations on the data - missing value check, label encoding, imputation

    from data_manipulations import *

    vars_selected_train = missing_value_calculation(X_train, miss_per) # missing value check



    vars_selected = filter(None,list(set(list(vars_selected_train))))
    print('vars selected')
    X = X_train.select(*vars_selected)
    print(X.columns)
    vars_selectedn=X.columns
    X = X.cache()

    Y = train_table.select(target_column_name)
    Y = Y.cache()




    char_vars, num_vars = identify_variable_type(X)
    X, char_labels = categorical_to_index(X, char_vars) #label encoding
    X = numerical_imputation(X,num_vars, impute_with) # imputation
    X = X.select([c for c in X.columns if c not in char_vars])
    X = rename_columns(X, char_vars)
    joinedDF = join_features_and_target(X, Y)

    joinedDF = joinedDF.cache()
    print('Features and targets are joined')

    train, valid, test = train_valid_test_split(joinedDF, train_size, valid_size, seed)
    train = train.cache()
    valid = valid.cache()
    test = test.cache()
    print('Train, valid and test dataset created')




    x = train.columns
    x.remove(target_column_name)
    feature_count = len(x)
    print(feature_count)

    if feature_count > 30:
        print('# No of features - ' + str(feature_count) + '.,  Performing feature reduction before running the model.')

    # directory to produce the outputs of the automation
    import os

    try:
        if not os.path.exists('/home/' + user_id + '/' + 'mla_' + mdl_ltrl):
            os.mkdir('/home/' + user_id + '/' + 'mla_' + mdl_ltrl)
    except:
        user_id = 'jovyan'
        if not os.path.exists('/home/' + user_id + '/' + 'mla_' + mdl_ltrl):
            os.mkdir('/home/' + user_id + '/' + 'mla_' + mdl_ltrl)

    subprocess.call(['chmod','777','-R','/home/' + user_id + '/' + 'mla_' + mdl_ltrl])

    x = train.columns
    x.remove(target_column_name)
    sel_train = assembled_vectors(train,x, target_column_name)
    sel_train.cache()

    # # Variable Reduction for more than 30 variables in the feature set using Random Forest

    from pyspark.ml.classification import  RandomForestClassifier
    from feature_selection import *

    rf = RandomForestClassifier(featuresCol="features",labelCol = target_column_name)
    mod = rf.fit(sel_train)
    varlist = ExtractFeatureImp(mod.featureImportances, sel_train, "features")
    selected_vars = [str(x) for x in varlist['name'][0:30]]
    train = train.select([target_column_name] + selected_vars)
    train.cache()

    save_feature_importance(user_id, mdl_ltrl, varlist) #Create feature importance plot and excel data

    x = train.columns
    x.remove(target_column_name)
    feature_count = len(x)
    print(feature_count)

    train, valid, test, pipelineModel = scaled_dataframes(train,valid,test,x,target_column_name)

    train = train.cache()
    valid = valid.cache()
    test = test.cache()
    print('Train, valid and test are scaled')
    print (train.columns)

    # import packages to perform model building, validation and plots

    import time
    from validation_and_plots import *

    #apply the transformation done on train dataset to OOT 1 and OOT 2 using the score_new_df function
    def score_new_df(scoredf):
        newX = scoredf.select(*final_vars)
        #idX = scoredf.select(id_vars)
        print(newX.columns)
        newX = newX.select(*vars_selectedn)
        print(newX.columns)
        newX = char_labels.transform(newX)
        newX = numerical_imputation(newX,num_vars, impute_with)
        newX = newX.select([c for c in newX.columns if c not in char_vars])
        newX = rename_columns(newX, char_vars)

        finalscoreDF = pipelineModel.transform(newX)
        finalscoreDF.cache()
        return finalscoreDF

    #apply the transformation done on train dataset to OOT 1 and OOT 2 using the score_new_df function

    x = 'features'
    y = target_column_name

    oot1_targetY = oot1_table.select(target_column_name)
    print(oot1_table.columns)
    oot1_intDF = score_new_df(oot1_table)
    oot1_finalDF = join_features_and_target(oot1_intDF, oot1_targetY)
    oot1_finalDF.cache()
    print(oot1_finalDF.dtypes)

    oot2_targetY = oot2_table.select(target_column_name)
    oot2_intDF = score_new_df(oot2_table)
    oot2_finalDF = join_features_and_target(oot2_intDF, oot2_targetY)
    oot2_finalDF.cache()
    print(oot2_finalDF.dtypes)

    # run individual models

    from model_builder import *
    from metrics_calculator import *

    loader_model_list = []
    dataset_list = ['train','valid','test','oot1','oot2']
    datasets = [train,valid,test,oot1_finalDF, oot2_finalDF]
    print(train.count())
    print(test.count())
    print(valid.count())
    print(oot1_finalDF.count())
    print(oot2_finalDF.count())
    models_to_run = []

    if run_logistic_model:
        lrModel = logistic_model(train, x, y) #build model
        lrModel.write().overwrite().save('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/logistic_model.h5') #save model object
        print("Logistic model developed")
        model_type = 'Logistic'
        l = []

        try:
            os.mkdir('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/' + str(model_type))
        except:
            pass

        for i in datasets:
            l += model_validation(user_id, mdl_ltrl, i, y, lrModel, model_type, dataset_list[datasets.index(i)]) #validate model

        draw_ks_plot(user_id, mdl_ltrl, model_type) #ks charts
        joblib.dump(l,'/home/' + user_id + '/' + 'mla_' + mdl_ltrl  + '/logistic_metrics.z') #save model metrics
        models_to_run.append('logistic')
        loader_model_list.append(LogisticRegressionModel)

    if run_randomforest_model:
        rfModel = randomForest_model(train, x, y) #build model
        rfModel.write().overwrite().save('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/randomForest_model.h5') #save model object
        print("Random Forest model developed")
        model_type = 'RandomForest'
        l = []

        try:
            os.mkdir('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/' + str(model_type))
        except:
            pass

        for i in datasets:
            l += model_validation(user_id, mdl_ltrl, i, y, rfModel, model_type, dataset_list[datasets.index(i)]) #validate model

        draw_ks_plot(user_id, mdl_ltrl, model_type) #ks charts
        joblib.dump(l,'/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/randomForest_metrics.z') #save model metrics
        models_to_run.append('randomForest')
        loader_model_list.append(RandomForestClassificationModel)

    if run_boosting_model:
        gbModel = gradientBoosting_model(train, x, y) #build model
        gbModel.write().overwrite().save('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/gradientBoosting_model.h5') #save model object
        print("Gradient Boosting model developed")
        model_type = 'GradientBoosting'
        l = []

        try:
            os.mkdir('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/' + str(model_type))
        except:
            pass

        for i in datasets:
            l += model_validation(user_id, mdl_ltrl, i, y, gbModel, model_type, dataset_list[datasets.index(i)]) #validate model

        draw_ks_plot(user_id, mdl_ltrl, model_type) #ks charts
        joblib.dump(l,'/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/gradientBoosting_metrics.z') #save model metrics
        models_to_run.append('gradientBoosting')
        loader_model_list.append(GBTClassificationModel)



    if run_neural_model:
        mlpModel = neuralNetwork_model(train, x, y, feature_count) #build model
        mlpModel.write().overwrite().save('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/neuralNetwork_model.h5') #save model object
        print("Neural Network model developed")
        model_type = 'NeuralNetwork'
        l = []

        try:
            os.mkdir('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/' + str(model_type))
        except:
            pass

        for i in datasets:
            l += model_validation(user_id, mdl_ltrl, i, y, mlpModel, model_type, dataset_list[datasets.index(i)]) #validate model

        draw_ks_plot(user_id, mdl_ltrl, model_type) #ks charts
        joblib.dump(l,'/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/neuralNetwork_metrics.z') #save model metrics
        models_to_run.append('neuralNetwork')
        loader_model_list.append(MultilayerPerceptronClassificationModel)


    # model building complete. Let us validate the metrics for the models created


    # model validation part starts now.
    from model_selection import *
    output_results = select_model(user_id, mdl_ltrl, model_selection_criteria, dataset_to_use) #select Champion, Challenger based on the metrics provided by user

    #print(type(output_results), output_results)

    selected_model = output_results['model_type'][0] #Champion model based on selected metric

    load_model = loader_model_list[models_to_run.index(selected_model)] #load the model object for Champion model
    model = load_model.load('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/' + selected_model + '_model.h5')

    print('Model selected for scoring - ' + selected_model)


    # Produce pseudo score for production deployment
    # save objects produced in the steps above for future scoring
    import joblib

    char_labels.write().overwrite().save('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/char_label_model.h5')
    pipelineModel.write().overwrite().save('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/pipelineModel.h5')

    save_list = [final_vars,vars_selected,char_vars,num_vars,impute_with,selected_model,dev_table_name]
    joblib.dump(save_list,'/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/model_scoring_info.z')


    # # Create score code


    from scorecode_creator import *
    selected_model_scorecode(user_id, mdl_output_id, mdl_ltrl, parameters)
    individual_model_scorecode(user_id, mdl_output_id, mdl_ltrl, parameters)

    message = message + 'Model building activity complete and the results are attached with this email. Have Fun'

    from zipper_function import *
    try:
        filename = zipper('/home/' + user_id + '/' + 'mla_' + mdl_ltrl)
    except:
        filename = ''

# clean up files loaded in the local path
if import_data:
    file_list = [dev_input_file, oot1_input_file, oot2_input_file]

    for i in list(set(file_list)):
        try:
            os.remove(data_folder_path + str(i))
        except:
            pass

# clean up files loaded in the hdfs path
if import_data:
    file_list = [dev_input_file, oot1_input_file, oot2_input_file]

    for i in list(set(file_list)):
        try:
            out,err=subprocess.Popen([ 'rm','-r','-f',hdfs_folder_path+str(i)],stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()
        except:
            pass
