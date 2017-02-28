import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import Row 
from pyspark.sql import SparkSession

spark = SparkSession\
        .builder\
        .appName("ALSExample")\
        .getOrCreate()

lines = spark.read.text("data/data_train_processed.csv").rdd # "data/data_train_woHeader.csv" "data/try_woHeader.csv")
parts = lines.map(lambda row: row.value.split(","))
ratingsRDD = parts.map(lambda p: Row(movieId=int(p[0]), userId=int(p[1]), 
                                     rating=float(p[2])))
ratings = spark.createDataFrame(ratingsRDD)


from pyspark.ml.recommendation import ALS
from plots import plot_train_test_lambda, plot_train_test_K
from pyspark.ml.evaluation import RegressionEvaluator


#features_K = np.array([20, 25, 30, 35, 40, 45, 50])
lambda_ = np.array([0.5, 0.1, 0.05, 0.01, 0.005, 0.001])
(training, test) = ratings.randomSplit([0.8, 0.2])

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
rmse_tr = []
rmse_te = []
for i,el in enumerate(lambda_):
    als = ALS(rank=20, maxIter=4, regParam=el, userCol="userId", itemCol="movieId", ratingCol="rating")  
    model = als.fit(training)
    
    predictions_tr = model.transform(training)
    rmse_tr_tmp = evaluator.evaluate(predictions_tr)
    rmse_tr.append(rmse_tr_tmp)
    
    predictions_te = model.transform(test)
    rmse_te_tmp = evaluator.evaluate(predictions_te)
    rmse_te.append(rmse_te_tmp)

#plot_train_test_K(rmse_tr, rmse_te, features_K)
plot_train_test_lambda(rmse_tr, rmse_te, lambda_)