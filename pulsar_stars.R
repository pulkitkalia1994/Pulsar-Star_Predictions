stars<-read.csv("C:\\R-Programming\\pulsar star prediction\\pulsar_stars.csv")

library(Boruta)
library(rpart)
library(rpart.plot)
library(caret)
library(caTools)
library("xgboost")
library(e1071)
library("Ckmeans.1d.dp")
boruta.train <- Boruta(target_class~., data = stars, doTrace = 2)
print(boruta.train)
#all variables are important here 
plot(boruta.train, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i)
boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
       at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)



train_index <- sample(1:nrow(stars), nrow(stars)*0.75)
# Full data set
data_variables <- as.matrix(stars[,-1])
data_label <- stars[,"target_class"]
data_matrix <- xgb.DMatrix(data = as.matrix(stars), label = data_label)
# split train data and make xgb.DMatrix
train_data   <- data_variables[train_index,]
train_label  <- data_label[train_index]
train_matrix <- xgb.DMatrix(data = train_data, label = train_label)
# split test data and make xgb.DMatrix
test_data  <- data_variables[-train_index,]
test_label <- data_label[-train_index]
test_matrix <- xgb.DMatrix(data = test_data, label = test_label)

numberOfClasses <- length(unique(stars$target_class))
xgb_params <- list("objective" = "multi:softmax",
                   "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses)
nround    <- 5 # number of XGBoost rounds
cv.nfold  <- 5

# Fit cv.nfold * cv.nround XGB models and save OOF predictions

bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = nround)

predictions<-predict(bst_model,newdata=test_matrix)
print(table(predictions,test_label))


