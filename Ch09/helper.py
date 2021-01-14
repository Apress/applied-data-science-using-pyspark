#helper functions
from pyspark.sql import functions as F
import pickle
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, DoubleType

# read model objects saved from the training process
path_to_read_objects = '/home/work/Desktop/score_code_objects'

#pyspark objects
char_labels = PipelineModel.load(path_to_read_objects + '/char_label_model.h5')
assembleModel = PipelineModel.load(path_to_read_objects + '/assembleModel.h5')
clf_model = RandomForestClassificationModel.load(path_to_read_objects + '/clf_model.h5')
#python objects
with open(path_to_read_objects + '/file.pkl', 'rb') as handle:
    features_list, char_vars, num_vars = pickle.load(handle)
    
#make necessary transformations
def rename_columns(df, char_vars):
    mapping = dict(zip([i + '_index' for i in char_vars], char_vars))
    df = df.select([F.col(c).alias(mapping.get(c, c)) for c in df.columns])
    return df

# score the new data
def score_new_df(scoredf):
    X = scoredf.select(features_list)
    X = char_labels.transform(X)
    X = X.select([c for c in X.columns if c not in char_vars])
    X = rename_columns(X, char_vars)
    final_X = assembleModel.transform(X)
    final_X.cache()
    pred = clf_model.transform(final_X)
    pred.cache()
    split_udf = udf(lambda value: value[1].item(), DoubleType())
    pred = pred.select('prediction', split_udf('probability').alias('probability'))
    return pred