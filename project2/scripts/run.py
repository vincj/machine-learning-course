from pyspark.sql import SparkSession
# required modules for loading the RDD from external files
from helpers_als import prepare_csv_for_als
from pyspark.sql import Row 
# required modules for 
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
# prepare csv file for subsequent operations 
prepare_csv_for_als("data/sampleSubmission_processed.csv")
# don't forget to remove header Id,Prediction 

spark = SparkSession\
        .builder\
        .appName("ALSExample")\
        .getOrCreate()

# Load the rdd object for training of the model
lines = spark.read.text("data/data_train_processed.csv").rdd
parts = lines.map(lambda row: row.value.split(","))
ratingsRDD = parts.map(lambda p: Row(movieId=int(p[0]), userId=int(p[1]), 
                                     rating=float(p[2])))
ratings = spark.createDataFrame(ratingsRDD)

# Load the rdd object for predictions
lines = spark.read.text("data/sampleSubmission_processed.csv").rdd 
parts = lines.map(lambda row: row.value.split(","))
ratingsRDD = parts.map(lambda p: Row(movieId=int(p[0]), userId=int(p[1]), 
                                     rating=float(p[2])))
submission_ratings = spark.createDataFrame(ratingsRDD)

# construction of the model starts here
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
# Build the recommendation model using ALS on the training data
als_cv = ALS(rank=50, maxIter=10, regParam = 0.01, userCol="userId", itemCol="movieId", ratingCol="rating")  

paramGrid = ParamGridBuilder().build()
# initialize parameters for cross validation using CrossValidator
crossval = CrossValidator(
    estimator=als_cv,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=4)

model = crossval.fit(ratings)
pred_cv = model.transform(submission_ratings)

# prepare file for submission
# save DataFrame as csv
pred_cv.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("data/sampleSubmission_new") # predictions.write.csv('data/pred_als_spark.csv')
# rename file just after previous operation
