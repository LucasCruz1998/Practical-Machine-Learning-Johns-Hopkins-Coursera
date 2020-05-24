#R PROJECT 

#This script was executed under R version 3.6.1 using Mac OS Mojave version 10.14.6

#SUMMARY 
# 1 - Reproductibility of the algorithm
# 2 - Configuration and Data Download 
# 3 - Reading Data 
# 4 - Cleaning Data
# 5 - Partionning Training Set 
# 6 - Data Modelling
# 7 - Using the Random Forest Model to the Test Data Set

## 1 - Reproductibility of the algorithm 

#To make this algorithm work, you need to check for certain packages and setting a seed equals to mine. NB : To install a package in R, run the command "install.packages("nameofpackage)"

library(rattle)
library(caret)
library(rpart)
library(rpart.plot)
library(corrplot)
library(randomForest)

set.seed(23343)

## 2 - Configuration and Data Download

#I set the working directory. In my case I choose to create a folder named "Final Project" in the default R folder being R Project

#Data Sources : 
#- Training Set : https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv 
# - Testing Set : https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

#I select the link for my 2 data sets. 

TrainDataLink <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"

TestDataLink <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"


#Thanks "http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har" for sharing the data with Coursera for the Pratical Machine Learning Final Project !


## 3 - Reading Data 

TrainDataRaw <- read.csv(TrainDataLink)
TestDataRaw <- read.csv(TestDataLink)
dim(TrainDataRaw);dim(TestDataRaw)

#The Training Data set contains 19622 observations and 160 variables. The Testing Data set contains 20 observations and 160 variables. The "classe" variable is what we need to predict in this particular case

## 4 - Cleaning Data 

#Here what we need to do is to remove the observations with missing values, meaningless variables, ...

#First Cleaning : We remove the Near Zero Variance variables from both data set

NearZeroVariance <- nearZeroVar(TrainDataRaw, saveMetrics = TRUE)
head(NearZeroVariance,20)

TrainSet1 <- TrainDataRaw[, !NearZeroVariance$nzv]
TestSet1 <- TestDataRaw[, !NearZeroVariance$nzv]
dim(TrainSet1); dim(TestSet1)


#First Cleaning Conclusion : We have now a training data set of 19622 observations and 100 variables and a testing data set with 20 observations and 100 variables

#Second Cleaning : We remove the columns that do not contribute much to the accelerometer measurements

regex <- grepl("^X|timestamp|user_name", names(TrainSet1))
TrainingSet <- TrainSet1[, !regex]
TestingSet <- TestSet1[, !regex]
dim(TrainingSet);dim(TestingSet)

#Second Cleaning Conclusion : We now have a training set with 19622 observations and 95 variables and a testing set with 20 observations and 54 variables

#Third Cleaning : We remove columns that have NA values in them. 

Condition <- (colSums(is.na(TrainingSet))== 0)
TrainingSet <- TrainingSet[, Condition]
TestingSet <- TestingSet[, Condition]
dim(TrainingSet);dim(TestingSet)


#Third Cleaning Conclusion : We now have a training set with 19622 observations and 54 variables and a testing set with 20 observations and 54 variables

#Now we can display the Correlation Matrix of Columns for the Training Data Set

correlationMatrix <- cor(TrainingSet[,-length(names(TrainingSet))])
corrplot(correlationMatrix,method="color")

## 5 - Partitionning Training Set

#Please set the same seed because of reproductibility issues menntionned in 1 - Reproductibility 

set.seed(23343)

#We set 70% of the values in the pure training data set and 30% in the test validation data set. The validation data set will be helpful to perform cross validation in future steps.

inTrain <- createDataPartition(TrainingSet$classe, p=0.70, list=FALSE)
Training <- TrainingSet[inTrain,]
Validation<- TrainingSet[-inTrain,]

#The Dataset now consists of 54 variables with the observations divided as follow : Training Data Set : 13737 observations; Validation Data : 5885 observations; Testing Data : 20 observations


## 6 - Data Modelling

# A - Decision Tree

#We fit a predictive model for activity recognition using the "Decision Tree" algorithm in R and we can now display the Decision Tree. 

modelTree <- rpart(classe ~., data = Training, method = "class")
prp(modelTree)

#We should now estimate the performance of our model algorithm on the validation data set

predictTree <- predict(modelTree, Validation, type = "class")
confusionMatrix(Validation$classe, predictTree)

accuracy <- postResample(predictTree, Validation$classe)
out_of_sample_error <- 1 - as.numeric(confusionMatrix(Validation$classe, predictTree)$overall[1])

print(accuracy)
print(out_of_sample_error)

#We get an Estimated Accuracy of 82.01% for the Decision Tree Model and the Estimated Out-of-Sample-Error is 17.99%

# B - Random Forest

#We fit a predictive model for activity recognition using the "Random Forest" algorithm in R because it is useful as it selects important variables and is robust to correlated covariates for example in general. We will use a 5-fold cross validation when using the algorithm. 

ModelRandomForest <- train(classe ~., method ="rf", data = Training, trControl= trainControl(method="cv", 5), ntree = 250)
ModelRandomForest

#NB : This part of the algotrithm may be really slow to display so please be patient. 

#We should now estimate the performance of our model algorithm on the validation data set

PredictRandomForest <- predict(ModelRandomForest, Validation)
confusionMatrix(Validation$classe, PredictRandomForest)

accuracy <- postResample(PredictRandomForest, Validation$classe)
out_of_sample_error <- 1 - as.numeric(confusionMatrix(Validation$classe,PredictRandomForest)$overall[1])

print(accuracy)
print(out_of_sample_error)

#We get an Estimated Accuracy of 99.76% for the Decision Tree Model and the Estimated Out-of-Sample-Error is 0.24%

# C - Conclusion 

#We get way better results with Random Forests, but it was to be expected !

##7 - Using the Random Forest model to the Test Data Set

predict(ModelRandomForest, TestingSet[,-length(names(TestingSet))])













 





