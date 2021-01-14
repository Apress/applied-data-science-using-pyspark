# import necessary packages
from pyspark.sql import SparkSession
from helper import *

# new data to score
path_to_output_scores = '/localuser'
filename = path_to_output_scores + "/score_data.csv"
spark = SparkSession.builder.getOrCreate()
score_data = spark.read.csv(filename, header=True, inferSchema=True, sep=';')

#score the data
final_scores_df = score_new_df(score_data)
#final_scores_df.show()
final_scores_df.repartition(1).write.format('csv').mode("overwrite").options(sep='|', header='true').save(path_to_output_scores + "/predictions.csv")
