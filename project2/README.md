

# Project 2 


Machine Learning Course, Fall 2016





# Environment


Requires Spark package. Download and add paths to bash. 





# run.py and run2.py


This is the algorithm of ALS with cross validation. Proceed as follow to run it:


Place your submission "sampleSubmission.csv" in folder data. Create a copy of "sampleSubmission.csv" and rename it "sampleSubmission_processed.csv". Remove the header of "sampleSubmission_processed.csv" by hand.


Because we use the Spark package, the script should be run with the command "spark-submit". The running should last less than 10 minutes.





A new folder "sampleSubmission_new" is created, with a csv file inside whose name begins with "part-r-00000". Copy this csv file in the upper directory and rename it "tmp.csv". Run the script "run2.py".





It is with this script that we obtained our best score. Yet we had not seeded the random split between train and test. Therefore could not find back our best kaggle score.





# als_noCV.py


ALS method using Spark package. This script can be run similarly as run.py for submission to kaggle: by renaming the csv file produced, moving it to upper directory and running "run2.py".





# als_CV_plot.py and als_noCV_plot.py 


In each script, either with or without cross validation, we plot the error for different values of K or lambda. On has to uncomment or comment the appropriate array of values to plot either K or lambda errors.





# als_old.py


This script is a first version of ALS. Very slow because coded by hand.





# baseline.py


Predictions made from global, user or item means.





# sgd.py


Calculate matrix factorization using Stochastic Gradient Descent. One can comment the appropriate part of the code if only interested in plotting the result of cross-validation, or only interested in the code for submission.





# similarity.py 


Importantly inspired from the example on our reference number 1.





# svd.py


Inspired as well from reference 1.





# plots.py, helpers.py, helpers_als.py


These are all "helper" scripts.




