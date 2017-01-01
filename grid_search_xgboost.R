setwd("path/to/your/data")

library(xgboost)
library(Matrix)

# verilerin r'a aktarilmasi
train_df <- read.csv("train.csv", T, ",")
test_df <- read.csv("test.csv", T, ",")


target <- train_df$loss
train_df$loss <- NULL
data <- rbind(train_df, test_df)
data$id <- NULL 

# sparse matrix olusturma
data_sparse <- sparse.model.matrix(~.-1, data = as.data.frame(data))
cat("Data size: ", data_sparse@Dim[1], " x ", data_sparse@Dim[2], "  \n", sep = "")


#sparse matrix'i xgboost matrix formatına getirme
dtrain <- xgb.DMatrix(data = data_sparse[1:nrow(train_df), ], label = target)

rm(data, test_df, train_df, data_sparse, target)

# grid search yapacagimiz parametreleri ve degerlerini liste haline getiriyoruz
searchGridSubCol <- expand.grid(subsample = c(0.7, 0.6), 
                                colsample_bytree = c(0.8, 0.7),
                                max_depth = c(7,8,9),
                                min_child = c(2,3,4)
                  )
                                

# grid search yapacak fonksiyonu calistiriyoruz
rmseErrorsHyperparameters <- apply(searchGridSubCol, 1, function(parameterList){
  
  # belirledigimiz parametleri model icinde kullanmak icin degiskenlere atiyoruz
  currentSubsampleRate <- parameterList[["subsample"]]
  currentColsampleRate <- parameterList[["colsample_bytree"]]
  currentDepth <- parameterList[["max_depth"]]
  currentMinChild <- parameterList[["min_child"]]
  
  
  
  xgboostModelCV <- xgb.cv(data =  dtrain, 
                           nrounds = 200, 
                           nfold = 5, 
                           showsd = TRUE, 
                           metrics = "rmse", 
                           verbose = TRUE, 
                           "eval_metric" = "rmse",
                           "objective" = "reg:linear", 
                           "max.depth" = currentDepth, 
                           "eta" = 0.05,                               
                           "subsample" = currentSubsampleRate, 
                           "colsample_bytree" = currentColsampleRate,
                            print_every_n = 10, 
                           "min_child_weight" = currentMinChild )
  
  xvalidationScores <- as.data.frame(xgboostModelCV$evaluation_log)
  #Save rmse of the last iteration
  rmse <- tail(xvalidationScores$test_rmse_mean, 1)
  trmse <- tail(xvalidationScores$train_rmse_mean,1)
output <- return(c(rmse, trmse, currentSubsampleRate, currentColsampleRate, currentDepth))

})

# fonksiyon sonucu output adinda bir degiskene rmse, test rmse ve 
#girdigimiz parametrelerin degerini atiyoruz
# output degiskenini data frame haline getirip sonuclari inceleyebiliriz

output <- as.data.frame(t(rmseErrorsHyperparameters))
varnames <- c("TestRMSE", "TrainRMSE", "SubSampRate", "ColSampRate", "Depth")
names(output) <- varnames
output

