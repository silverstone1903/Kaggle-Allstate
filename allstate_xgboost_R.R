setwd("/path/to/your/dataset")

library(xgboost)
library(Matrix)

train_df <- read.csv("train.csv", T, ",")
test_df <- read.csv("test.csv", T, ",")


#xgboost on isleme 
target <- train_df$loss
train_df$loss <- NULL
data <- rbind(train_df, test_df)
data$id <- NULL 

#sparse matrix olusturma
data_sparse <- sparse.model.matrix(~.-1, data = as.data.frame(data))
cat("Data size: ", data_sparse@Dim[1], " x ", data_sparse@Dim[2], "  \n", sep = "")

#xgboost icin design matrix olusturma
dtrain <- xgb.DMatrix(data = data_sparse[1:nrow(train_df), ], label = target) 
dtest <- xgb.DMatrix(data = data_sparse[(nrow(train_df)+1):nrow(data), ]) 


#xgboost ile egitim
set.seed(1903)

# model parametrelerini belirliyoruz
# detaylar icin -> https://xgboost.readthedocs.io/en/latest/how_to/param_tuning.html
# cross validation ile modeli olusturuyoruz

system.time( #ne kadar surdugunu gormek icin system time fonksiyonu ile calistiriyorum
temp_model <- xgb.cv(data = dtrain,
                     nfold = 10,
                     nrounds = 100,
                     max_depth = 7,
                     eta = 0.05,
                     subsample = 0.7,
                     colsample_bytree = 0.8,
                     gamma = 2,
                     metrics = "rmse",
                     maximize = FALSE,
                     early_stopping_rounds = 10,
                     min_child_weight = 4,
                     objective = "reg:linear",
                     print_every_n = 50,
                     verbose = TRUE)
)

#en iyi iterasyonu seciyoruz
best_it <- temp_model$best_iteration

set.seed(1903)
#cv ile buldugumuz en iyi iterasyon kadar egitim modelini calistiriyoruz
temp_model <- xgb.train(data = dtrain,
                        nrounds = best_it,
                        max_depth = 7,
                        eta = 0.05,
                        subsample = 0.7,
                        colsample_bytree = 0.8,
                        eval_metric = "rmse",
                        gamma = 2,
                        min_child_weight = 4,
                        maximize = FALSE,
                        objective = "reg:linear",
                        print_every_n = 10,
                        verbose = TRUE,
                        watchlist = list(train = dtrain))

#tahmin edilen degerleri gormek icin
pred <- predict(temp_model, dtest)
head(pred)

# oznitelik onemine bakmak icin
importance <- xgb.importance(feature_names = data_sparse@Dimnames[[2]], model = temp_model)


submission <- read.csv("/your/submission/file", header = TRUE, ";")
submission$loss <- pred
write.csv(submission, "my_submission.csv", row.names = FALSE)


# modeli kaydetmek istersek
xgb.save(temp_model, "xgboost.model")
#kaydettigimiz modeli yuklemek icin
model <- xgb.load("xgboost.model")



