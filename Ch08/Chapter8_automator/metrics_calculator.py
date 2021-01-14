"""
    metrics_calculator.py

    Calculate the metrics like ROC, Accuracy and KS using the code below

"""

from pyspark.sql.types import DoubleType
from pyspark.sql import *
from pyspark.sql.functions import desc
from pyspark.sql.functions import udf
from pyspark.sql import functions as F
import sys
import time
# import __builtin__ as builtin
import builtins
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import QuantileDiscretizer
import numpy
import numpy as np
from pyspark import SparkContext,HiveContext,Row,SparkConf

spark = SparkSession.builder.appName("MLA_metrics_calculator").enableHiveSupport().getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
sc = spark.sparkContext

def highlight_max(data, color='yellow'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),index=data.index, columns=data.columns)

def calculate_metrics(predictions,y,data_type):
    start_time4 = time.time()

    # Calculate ROC
    evaluator = BinaryClassificationEvaluator(labelCol=y,rawPredictionCol='probability')
    auroc = evaluator.evaluate(predictions,{evaluator.metricName: "areaUnderROC"})
    print('AUC calculated',auroc)

    selectedCols = predictions.select(F.col("probability"), F.col('prediction'), F.col(y)).rdd.map(lambda row: (float(row['probability'][1]), float(row['prediction']), float(row[y]))).collect()
    y_score, y_pred, y_true = zip(*selectedCols)

    # Calculate Accuracy
    accuracydf=predictions.withColumn('acc',F.when(predictions.prediction==predictions[y],1).otherwise(0))
    accuracydf.createOrReplaceTempView("accuracyTable")
    RFaccuracy=spark.sql("select sum(acc)/count(1) as accuracy from accuracyTable").collect()[0][0]
    print('Accuracy calculated',RFaccuracy)

#     # Build KS Table
    split1_udf = udf(lambda value: value[1].item(), DoubleType())

    if data_type in ['train','valid','test','oot1','oot2']:
        decileDF = predictions.select(y, split1_udf('probability').alias('probability'))
    else:
        decileDF = predictions.select(y, 'probability')

    decileDF=decileDF.withColumn('non_target',1-decileDF[y])

    window = Window.orderBy(desc("probability"))
    decileDF = decileDF.withColumn("rownum", F.row_number().over(window))
    decileDF.cache()
    decileDF=decileDF.withColumn("rownum",decileDF["rownum"].cast("double"))

    window2 = Window.orderBy("rownum")
    RFbucketedData=decileDF.withColumn("deciles", F.ntile(10).over(window2))
    RFbucketedData = RFbucketedData.withColumn('deciles',RFbucketedData['deciles'].cast("int"))
    RFbucketedData.cache()
    #a = RFbucketedData.count()
    #print(RFbucketedData.show())

    ## to pandas from here
    print('KS calculation starting')
    target_cnt=RFbucketedData.groupBy('deciles').agg(F.sum(y).alias('target')).toPandas()
    non_target_cnt=RFbucketedData.groupBy('deciles').agg(F.sum("non_target").alias('non_target')).toPandas()
    overall_cnt=RFbucketedData.groupBy('deciles').count().alias('Total').toPandas()
    overall_cnt = overall_cnt.merge(target_cnt,on='deciles',how='inner').merge(non_target_cnt,on='deciles',how='inner')
    overall_cnt=overall_cnt.sort_values(by='deciles',ascending=True)
    overall_cnt['Pct_target']=(overall_cnt['target']/overall_cnt['count'])*100
    overall_cnt['cum_target'] = overall_cnt.target.cumsum()
    overall_cnt['cum_non_target'] = overall_cnt.non_target.cumsum()
    overall_cnt['%Dist_Target'] = (overall_cnt['cum_target'] / overall_cnt.target.sum())*100
    overall_cnt['%Dist_non_Target'] = (overall_cnt['cum_non_target'] / overall_cnt.non_target.sum())*100
    overall_cnt['spread'] = builtins.abs(overall_cnt['%Dist_Target']-overall_cnt['%Dist_non_Target'])
    decile_table=overall_cnt.round(2)
    print("KS_Value =", builtins.round(overall_cnt.spread.max(),2))
    #print "Test Error =", builtin.round((1.0 - RFaccuracy),3)
    #print "Accuracy =", builtin.round(RFaccuracy,3)
    #print "AUC=", builtin.round(auroc,3)
    decileDF.unpersist()
    RFbucketedData.unpersist()
    print("Metrics calculation process Completed in : "+ " %s seconds" % (time.time() - start_time4))
    return auroc,RFaccuracy,builtins.round(overall_cnt.spread.max(),2), y_score, y_pred, y_true, overall_cnt
#    return auroc,RFaccuracy, None
