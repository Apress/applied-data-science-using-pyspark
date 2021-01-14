"""
    model_builder.py

    Build the models depending upon the user input. Currently this module supports

    1. Logistic Regression
    2. Random Forest
    3. Gradient Boosting
    4. Decision Tree
    5. Neural Network

"""

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LogisticRegressionModel
# from sklearn.externals import joblib
import joblib

def logistic_model(train, x, y):
    lr = LogisticRegression(featuresCol = x, labelCol = y, maxIter = 10)
    lrModel = lr.fit(train)
    return lrModel

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import RandomForestClassificationModel

def randomForest_model(train, x, y):
    rf = RandomForestClassifier(featuresCol = x, labelCol = y, numTrees=10)
    rfModel = rf.fit(train)
    return rfModel

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import GBTClassificationModel

def gradientBoosting_model(train, x, y):
    gb = GBTClassifier(featuresCol = x, labelCol = y, maxIter=10)
    gbModel = gb.fit(train)
    return gbModel

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import DecisionTreeClassificationModel

def decisionTree_model(train, x, y):

    dt = DecisionTreeClassifier(featuresCol = x, labelCol = y, maxDepth=5)
    dtModel = dt.fit(train)
    return dtModel

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import MultilayerPerceptronClassificationModel

def neuralNetwork_model(train, x, y, feature_count):
    layers = [feature_count, feature_count*3, feature_count*2, 2]
    mlp = MultilayerPerceptronClassifier(featuresCol = x, labelCol = y, maxIter=100, layers=layers, blockSize=512,seed=12345)
    mlpModel = mlp.fit(train)
    return mlpModel
