import numpy as np
import matplotlib.pyplot as plt

from pyspark.sql import Row 
from pyspark.sql import SparkSession

spark = SparkSession\
        .builder\
        .appName("ALSExample")\
        .getOrCreate()

# Load the rdd object for training of the model
lines = spark.read.text("data/data_train_processed.csv").rdd # "data/data_train_woHeader.csv" "data/try_woHeader.csv")
parts = lines.map(lambda row: row.value.split(","))
ratingsRDD = parts.map(lambda p: Row(movieId=int(p[0]), userId=int(p[1]), 
                                     rating=float(p[2])))
ratings = spark.createDataFrame(ratingsRDD)


from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from plots import plot_train_test_K_CV, plot_train_test_lambda_CV
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
features_K = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60])
#lambda_ = np.array([0.5, 0.1, 0.05, 0.01, 0.005, 0.001])
   
rmse_all = []
for i,el in enumerate(features_K):
    als_cv = ALS(rank=el, maxIter=10, regParam =0.01, userCol="userId", itemCol="movieId", ratingCol="rating")
    paramGrid = ParamGridBuilder().build()
    crossval = CrossValidator(
            estimator=als_cv,
            estimatorParamMaps=paramGrid,
            evaluator=evaluator,
            numFolds=4)
                       
    model = crossval.fit(ratings)
    
    predictions_cv_all = model.transform(ratings)
    rmse_all_tmp = evaluator.evaluate(predictions_cv_all)
    rmse_all.append(rmse_all_tmp)
    
plot_train_test_K_CV(rmse_all, features_K)
#plot_train_test_lambda_CV(rmse_all, lambda_)

