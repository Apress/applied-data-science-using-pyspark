from flask import Flask, request, redirect, url_for, flash, jsonify, make_response
import numpy as np
import pickle
import json
import os, sys
# Path for spark source folder
os.environ['SPARK_HOME'] = '/usr/local/spark'
# Append pyspark  to Python Path
sys.path.append('/usr/local/spark/python')

from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from helper import *

conf = SparkConf().setAppName("real_time_scoring_api")
conf.set('spark.sql.warehouse.dir', 'file:///usr/local/spark/spark-warehouse/')
conf.set("spark.driver.allowMultipleContexts", "true")
spark = SparkSession.builder.master('local').config(conf=conf).getOrCreate()
sc = spark.sparkContext

app = Flask(__name__)

@app.route('/api/', methods=['POST'])
def makecalc():

    json_data = request.get_json()
    #read the real time input to pyspark df
    score_data = spark.read.json(sc.parallelize(json_data))
    #score df
    final_scores_df = score_new_df(score_data)
    #convert predictions to Pandas dataframe
    pred = final_scores_df.toPandas()
    final_pred = pred.to_dict(orient='rows')[0]
    return jsonify(final_pred)

if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=5000)
