# Load training csv file data_train.csv
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

# Load submissionSample for prediction by ALS
lines = spark.read.text("data/sampleSubmission_processed.csv").rdd 
parts = lines.map(lambda row: row.value.split(","))
ratingsRDD = parts.map(lambda p: Row(movieId=int(p[0]), userId=int(p[1]), 
                                     rating=float(p[2])))
submission_ratings = spark.createDataFrame(ratingsRDD)

# Load RDD objects for prediction
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")

(training, test) = ratings.randomSplit([0.8, 0.2])

# construction of the model starts here
from pyspark.ml.recommendation import ALS
# Build the recommendation model using ALS on the training data
als = ALS(rank=20, maxIter=10, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")  
model = als.fit(training)

# transform the model and calculate error on test
predictions_te = model.transform(test)
rmse_te = evaluator.evaluate(predictions_te)
print("Root-mean-square test error = " + str(rmse_te))

# transform the model and calculate error on train
predictions_tr = model.transform(training)
rmse_tr = evaluator.evaluate(predictions_tr)
print("Root-mean-square train error = " + str(rmse_tr))

# prepare file for submission
# save DataFrame as csv
predictions_tr.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("data/sampleSubmission_new") # predictions.write.csv('data/pred_als_spark.csv')
# rename file just after previous operation, and place it in upper directory "data"
# and run the script "run2.py"