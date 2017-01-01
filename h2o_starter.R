setwd("path/to/your/data")
library(h2o)

# tum islemci cekirdeklerini kullanacak sekilde h2o'yu baslatiyoruz
h2o.init(nthreads = -1)

# egitim ve test verisinin okunmasi
train <- h2o.importFile("train.csv", destination_frame = "train.hex")
test <- h2o.importFile("test.csv", destination_frame = "test.hex")

# bagimli degisken ve bagimsiz degiskenler icin atama yapiyoruz 
target <- "loss"
dim(train)
a <- colnames(train)
x_indep <- a[2:131]

## egitim verisinin %20'si ile "validation" seti olusturuyoruz
splits <- h2o.splitFrame(
  data = train, 
  ratios = c(0.8),   
  destination_frames = c("train.hex", "valid.hex"), seed = 1903)

train <- splits[[1]]
valid <- splits[[2]]



# ilk olarak random forest ile regresyon yapiyoruz
# parametreler icin ?h2o.randomForest 
system.time(
rf <- h2o.randomForest(y = target, 
                       x = x_indep, 
                       training_frame = train,
                       validation_frame = valid,
                       ntrees = 3000, 
                       mtries = 5, 
                       max_depth = 9, 
                       seed = 1903, 
                       sample_rate = 0.7, 
                       nfolds = 4, 
                       stopping_metric = "RMSE",
                       col_sample_rate_per_tree = 0.8,
                       stopping_rounds = 25))

# modelin performansina bakiyoruz
h2o.performance(rf)
# yarismada bizden istenen sonuc mae oldugu icin mae'ye bakiyoruz
h2o.mae(h2o.performance(rf, valid = T))

# test verisi icin modelin tahminleri hesaplaniyor
predict.rforest <- h2o.predict(rf, test)
head(predict.rforest)

# gonderim icin id ve tahminlerden olusan bir csv dosyasi olusturuyoruz
submissionFrame <- h2o.cbind(test$id, predict.rforest)
colnames(submissionFrame) <- c("id", "loss")
head(submissionFrame)
h2o.exportFile(submissionFrame, path="h2o_allstate_rf.csv")

# oznitelik onemine bakmak icin rfvimp'e atama yapiyoruz
rfvimp <- h2o.varimp(rf)
head(rfvimp)


# ysa ile regresyon tahmini

system.time(
  dl <- h2o.deeplearning(y = target,
                         x = x_indep,
                        training_frame = train,
                        validation_frame = valid,
                        rate = 0.05,
                        distribution = "gaussian",
                        stopping_rounds = 10,
                        stopping_metric = "RMSE",
                        epoch = 30,
                        hidden = c(40,50,60),
                        activation = "Rectifier",
                        seed = 1903
  ))

# modelin performansina bakiyoruz
h2o.mse(h2o.performance(dl, valid = TRUE))
h2o.performance(dl)

# test verisi icin modelin tahminleri hesaplaniyor
predict.dl <- (h2o.predict(dl, test))
head(predict.dl)

# gonderim icin id ve tahminlerden olusan bir csv dosyasi olusturuyoruz
submissionFrame <- h2o.cbind(test$id, predict.dl)
colnames(submissionFrame) <- c("id", "loss")
head(submissionFrame)
h2o.exportFile(submissionFrame, path= "h2o_allstate_dl.csv")


# gbm 
system.time(
gbm <- h2o.gbm(y = target,
               x = x_indep,
            distribution = "gaussian",
            training_frame = train,
            validation_frame = valid,
            stopping_metric = "RMSE",
            ntrees = 100,
            max_depth = 7,
            learn_rate = 0.05,
            stopping_rounds = 10,
            stopping_tolerance = 1e-4,
            sample_rate = 0.7,   
            col_sample_rate = 0.7,                                                   
            seed = 1903,    
            score_tree_interval = 50,
            nfolds = 10))


h2o.mae(h2o.performance(gbm, valid = TRUE))
h2o.performance(gbm)

pred  <- h2o.predict(gbm, test)
head(pred)

# gonderim icin id ve tahminlerden olusan bir csv dosyasi olusturuyoruz
submissionFrame <- h2o.cbind(test$id, pred)
colnames(submissionFrame) <- c("id","loss")
head(submissionFrame)
h2o.exportFile(submissionFrame,path="h2o_allstate_gbm.csv")
