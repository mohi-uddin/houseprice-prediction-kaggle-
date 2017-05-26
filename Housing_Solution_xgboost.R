rm(list=ls())
path = "D:/mohi/kaggle work/kaggle/house price regression model"
setwd(path)

train = read.csv("train.csv")
test = read.csv("test.csv")

dim(train)
dim(test)

str(train)
str(test)

train = read.csv("train.csv", stringsAsFactors = FALSE) 
test = read.csv("test.csv", stringsAsFactors = FALSE) 

#load pakages
library(caret)
library(plyr)
library(xgboost)
library(Metrics)

# Remove the target variable not found in test set
SalePrice = train$SalePrice 
train$SalePrice = NULL

# Combine data sets
full_data = rbind(train,test)

# Convert character columns to factor, filling NA values with "missing"
for (col in colnames(full_data)){
  if (typeof(full_data[,col]) == "character"){
    new_col = full_data[,col]
    new_col[is.na(new_col)] = "missing"
    full_data[col] = as.factor(new_col)
  }
}

# Separate out our train and test sets
train = full_data[1:nrow(train),]
train$SalePrice = SalePrice  
test = full_data[(nrow(train)+1):nrow(full_data),]

summary(train)
summary(test)

# Fill remaining NA values with -1
train[is.na(train)] = -1
test[is.na(test)] = -1


#correlation detection
for (col in colnames(train)){
  if(is.numeric(train[,col])){
    if( abs(cor(train[,col],train$SalePrice)) > 0.5){
      print(col)
      print( cor(train[,col],train$SalePrice) )
    }
  }
}
#less correlated
for (col in colnames(train)){
  if(is.numeric(train[,col])){
    if( abs(cor(train[,col],train$SalePrice)) < 0.1){
      print(col)
      print( cor(train[,col],train$SalePrice) )
    }
  }
}
#correlated data
cors = cor(train[ , sapply(train, is.numeric)])
high_cor = which(abs(cors) > 0.6 & (abs(cors) < 1))
rows = rownames(cors)[((high_cor-1) %/% 38)+1]
cols = colnames(cors)[ifelse(high_cor %% 38 == 0, 38, high_cor %% 38)]
vals = cors[high_cor]

cor_data = data.frame(cols=cols, rows=rows, correlation=vals)
cor_data

# Add variable that combines above grade living area with basement sq footage
train$total_sq_footage = train$GrLivArea + train$TotalBsmtSF
test$total_sq_footage = test$GrLivArea + test$TotalBsmtSF

# Add variable that combines above ground and basement full and half baths
train$total_baths = train$BsmtFullBath + train$FullBath + (0.5 * (train$BsmtHalfBath + train$HalfBath))
test$total_baths = test$BsmtFullBath + test$FullBath + (0.5 * (test$BsmtHalfBath + test$HalfBath))

# Remove Id since it should have no value in prediction
train$Id = NULL    
test$Id = NULL

# Create custom summary function in proper format for caret
custom_summary = function(data, lev = NULL, model = NULL){
  out = rmsle(data[, "obs"], data[, "pred"])
  names(out) = c("rmsle")
  out
}

# Create control object
control = trainControl(method = "cv",  # Use cross validation
                       number = 5,     # 5-folds
                       summaryFunction = custom_summary                      
)

# Create grid of tuning parameters
grid = expand.grid(nrounds=c(100, 200, 400, 800), # Test 4 values for boosting rounds
                   max_depth= c(4, 6),           # Test 2 values for tree depth
                   eta=c(0.1, 0.05, 0.025),      # Test 3 values for learning rate
                   gamma= c(0.1), 
                   colsample_bytree = c(1), 
                   min_child_weight = c(1))

set.seed(12)

xgb_tree_model =  train(SalePrice~.,      # Predict SalePrice using all features
                        data=train,
                        method="xgbTree",
                        trControl=control, 
                        tuneGrid = grid,
                        metric="rmsle",     # Use custom performance metric
                        maximize = FALSE)   # Minimize the metric

xgb_tree_model$results

xgb_tree_model$bestTune

varImp(xgb_tree_model)

test_predictions = predict(xgb_tree_model, newdata=test)

submission = read.csv("sample_submission.csv")
submission$SalePrice = test_predictions
write.csv(submission, "home_prices_xgb_sub1.csv", row.names=FALSE)