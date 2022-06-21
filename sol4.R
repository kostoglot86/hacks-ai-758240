library(tidyverse)
library(lightgbm)
library(lubridate)
library(caret)

# Путь к папке с данными
path_to_data = ''

# Константы и параметры
exclude =  c(1,37)
param_lgb = list(objective = "binary",
                max_bin = 256,
                learning_rate = 0.005,
                num_leaves = 7,
                bagging_fraction = 0.7,
                feature_fraction = 0.5,
                min_data = 20,
                bagging_freq = 1,
                metric = "binary_logloss")

# Чтение данных
train = read.csv(paste0(path_to_data, "/train.csv"), header = T, fileEncoding = 'UTF-8')
test = read.csv(paste0(path_to_data, "/test_dataset_test.csv", header = T, fileEncoding = 'UTF-8')
target = train[c(34:39)]
tr_te = rbind(train[c(1:33)],test)

# Новые признаки
tr_te$Время.засыпания = period_to_seconds(hms(tr_te$Время.засыпания))
tr_te$Время.пробуждения = period_to_seconds(hms(tr_te$Время.пробуждения))
tr_te$delta_sleep = ifelse(tr_te$Время.пробуждения - tr_te$Время.засыпания < 0, 
                           24*3600 + tr_te$Время.пробуждения - tr_te$Время.засыпания, tr_te$Время.пробуждения - tr_te$Время.засыпания)
for (i in c(as.numeric(which(sapply(tr_te, "class") == 'character')[-c(1)])))
  tr_te[,i] = as.numeric(as.factor(tr_te[,i]))
tr_te$bad_time = ifelse(is.na(tr_te$Возраст.алког) == T, 0,tr_te$Возраст.алког) +
  ifelse(is.na(tr_te$Возраст.курения) == T, 0,tr_te$Возраст.курения)
tr_te$cig = tr_te$Возраст.курения * tr_te$Сигарет.в.день

# Отделяем train и test
train = tr_te[c(1:dim(train)[1]),]
te = tr_te[c((dim(train)[1] + 1):(dim(tr_te)[1])),]

# Формируем фолды для кросс-валидации
set.seed(13)
train$fold <- createFolds(target$Артериальная.гипертензия, 1:nrow(train), k=5,list = FALSE)
fold.ids <- unique(train$fold)
custom.folds <- vector("list", length(fold.ids))
i <- 1
for( id in fold.ids){
  custom.folds[[i]] <- which(train$fold %in% id )
  i <- i+1
}

# Для каждого таргета определяем оптимальное число итераций lgb с параметрами выше для минимизации logloss на CV
dtrain <- lgb.Dataset(as.matrix(train[-c(exclude)]),label = target$Артериальная.гипертензия)
model_lgb1 = lgb.cv(data=dtrain, params = param_lgb, nrounds=10000, folds = custom.folds,
                    eval_freq = 100, early_stopping_rounds = 100)

dtrain <- lgb.Dataset(as.matrix(train[-c(exclude)]),label = target$ОНМК)
model_lgb2 = lgb.cv(data=dtrain, params = param_lgb, nrounds=10000, folds = custom.folds,
                    eval_freq = 100, early_stopping_rounds = 100)

dtrain <- lgb.Dataset(as.matrix(train[-c(exclude)]),label = target$Стенокардия..ИБС..инфаркт.миокарда)
model_lgb3 = lgb.cv(data=dtrain, params = param_lgb, nrounds=10000, folds = custom.folds,
                    eval_freq = 100, early_stopping_rounds = 100)

dtrain <- lgb.Dataset(as.matrix(train[-c(exclude)]),label = target$Сердечная.недостаточность)
model_lgb4 = lgb.cv(data=dtrain, params = param_lgb, nrounds=10000, folds = custom.folds,
                    eval_freq = 100, early_stopping_rounds = 100)

dtrain <- lgb.Dataset(as.matrix(train[-c(exclude)]),label = target$Прочие.заболевания.сердца)
model_lgb5 = lgb.cv(data=dtrain, params = param_lgb, nrounds=10000, folds = custom.folds,
                    eval_freq = 100, early_stopping_rounds = 100)

# Зафиксировали оптимальное кол-во итераций для каждого таргета
NR1 = model_lgb1$best_iter
NR2 = model_lgb2$best_iter
NR3 = model_lgb3$best_iter
NR4 = model_lgb4$best_iter
NR5 = model_lgb5$best_iter

# Для каждого таргета подбираем оптимальный трешхолд (шаг = 0.0001) для максимизации recall macro
res1 = c()  
for (j in c(1:5)) { 
  dtrain <- lgb.Dataset(as.matrix(train[train$fold != j,][-c(exclude)]),
                        label = target[train$fold != j,]$Артериальная.гипертензия)
  dtest <- lgb.Dataset(as.matrix(train[train$fold == j,][-c(exclude)]),
                       label = target[train$fold == j,]$Артериальная.гипертензия)
  valids = list(test = dtest)
  model_lgb1 = lgb.train(data=dtrain, valids = valids, params = param_lgb, nrounds=NR1, bagging_seed = 13,
                         feature_fraction_seed=42, eval_freq = 100)
  
  pr = predict(model_lgb1, as.matrix(train[train$fold == j,][-c(exclude)]))
  res = c()
  for (k in c(1:10000)) {
    pr_disc = ifelse(pr > k/10000, 1, 0)
    rec1 = sum(pr_disc == 1 & 
                 target[train$fold == j,]$Артериальная.гипертензия == 1)/sum(target[train$fold == j,]$Артериальная.гипертензия == 1)
    rec0 = sum(pr_disc == 0 & 
                 target[train$fold == j,]$Артериальная.гипертензия == 0)/sum(target[train$fold == j,]$Артериальная.гипертензия == 0)
    metric = 0.5*rec1 + 0.5*rec0
    res = rbind(res, data.frame(k = k/10000, metric = metric))
  }
  res1 = rbind(res1, data.frame(fold = j, res))
}
res1 = res1 %>% group_by(k) %>% summarize(metric = mean(metric))


res2 = c()  
for (j in c(1:5)) { 
  dtrain <- lgb.Dataset(as.matrix(train[train$fold != j,][-c(exclude)]),
                        label = target[train$fold != j,]$ОНМК)
  dtest <- lgb.Dataset(as.matrix(train[train$fold == j,][-c(exclude)]),
                       label = target[train$fold == j,]$ОНМК)
  valids = list(test = dtest)
  model_lgb1 = lgb.train(data=dtrain, valids = valids, params = param_lgb, nrounds=NR2, bagging_seed = 13,
                         feature_fraction_seed=42, eval_freq = 100)
  
  pr = predict(model_lgb1, as.matrix(train[train$fold == j,][-c(exclude)]))
  res = c()
  for (k in c(1:10000)) {
    pr_disc = ifelse(pr > k/10000, 1, 0)
    rec1 = sum(pr_disc == 1 & 
                 target[train$fold == j,]$ОНМК == 1)/sum(target[train$fold == j,]$ОНМК == 1)
    rec0 = sum(pr_disc == 0 & 
                 target[train$fold == j,]$ОНМК == 0)/sum(target[train$fold == j,]$ОНМК == 0)
    metric = 0.5*rec1 + 0.5*rec0
    res = rbind(res, data.frame(k = k/10000, metric = metric))
  }
  res2 = rbind(res2, data.frame(fold = j, res))
}
res2 = res2 %>% group_by(k) %>% summarize(metric = mean(metric))

res3 = c()  
for (j in c(1:5)) { 
  dtrain <- lgb.Dataset(as.matrix(train[train$fold != j,][-c(exclude)]),
                        label = target[train$fold != j,]$Стенокардия..ИБС..инфаркт.миокарда)
  dtest <- lgb.Dataset(as.matrix(train[train$fold == j,][-c(exclude)]),
                       label = target[train$fold == j,]$Стенокардия..ИБС..инфаркт.миокарда)
  valids = list(test = dtest)
  model_lgb1 = lgb.train(data=dtrain, valids = valids, params = param_lgb, nrounds=NR3, bagging_seed = 13,
                         feature_fraction_seed=42, eval_freq = 100)
  
  pr = predict(model_lgb1, as.matrix(train[train$fold == j,][-c(exclude)]))
  res = c()
  for (k in c(1:10000)) {
    pr_disc = ifelse(pr > k/10000, 1, 0)
    rec1 = sum(pr_disc == 1 & 
                 target[train$fold == j,]$Стенокардия..ИБС..инфаркт.миокарда == 1)/sum(target[train$fold == j,]$Стенокардия..ИБС..инфаркт.миокарда == 1)
    rec0 = sum(pr_disc == 0 & 
                 target[train$fold == j,]$Стенокардия..ИБС..инфаркт.миокарда == 0)/sum(target[train$fold == j,]$Стенокардия..ИБС..инфаркт.миокарда == 0)
    metric = 0.5*rec1 + 0.5*rec0
    res = rbind(res, data.frame(k = k/10000, metric = metric))
  }
  res3 = rbind(res3, data.frame(fold = j, res))
}
res3 = res3 %>% group_by(k) %>% summarize(metric = mean(metric))

res4 = c()  
for (j in c(1:5)) { 
  dtrain <- lgb.Dataset(as.matrix(train[train$fold != j,][-c(exclude)]),
                        label = target[train$fold != j,]$Сердечная.недостаточность)
  dtest <- lgb.Dataset(as.matrix(train[train$fold == j,][-c(exclude)]),
                       label = target[train$fold == j,]$Сердечная.недостаточность)
  valids = list(test = dtest)
  model_lgb1 = lgb.train(data=dtrain, valids = valids, params = param_lgb, nrounds=NR4, bagging_seed = 13,
                         feature_fraction_seed=42, eval_freq = 100)
  
  pr = predict(model_lgb1, as.matrix(train[train$fold == j,][-c(exclude)]))
  res = c()
  for (k in c(1:10000)) {
    pr_disc = ifelse(pr > k/10000, 1, 0)
    rec1 = sum(pr_disc == 1 & 
                 target[train$fold == j,]$Сердечная.недостаточность == 1)/sum(target[train$fold == j,]$Сердечная.недостаточность == 1)
    rec0 = sum(pr_disc == 0 & 
                 target[train$fold == j,]$Сердечная.недостаточность == 0)/sum(target[train$fold == j,]$Сердечная.недостаточность == 0)
    metric = 0.5*rec1 + 0.5*rec0
    res = rbind(res, data.frame(k = k/10000, metric = metric))
  }
  res4 = rbind(res4, data.frame(fold = j, res))
}
res4 = res4 %>% group_by(k) %>% summarize(metric = mean(metric))

res5 = c()  
for (j in c(1:5)) { 
  dtrain <- lgb.Dataset(as.matrix(train[train$fold != j,][-c(exclude)]),
                        label = target[train$fold != j,]$Прочие.заболевания.сердца)
  dtest <- lgb.Dataset(as.matrix(train[train$fold == j,][-c(exclude)]),
                       label = target[train$fold == j,]$Прочие.заболевания.сердца)
  valids = list(test = dtest)
  model_lgb1 = lgb.train(data=dtrain, valids = valids, params = param_lgb, nrounds=NR5, bagging_seed = 13,
                         feature_fraction_seed=42, eval_freq = 100)
  
  pr = predict(model_lgb1, as.matrix(train[train$fold == j,][-c(exclude)]))
  res = c()
  for (k in c(1:10000)) {
    pr_disc = ifelse(pr > k/10000, 1, 0)
    rec1 = sum(pr_disc == 1 & 
                 target[train$fold == j,]$Прочие.заболевания.сердца == 1)/sum(target[train$fold == j,]$Прочие.заболевания.сердца == 1)
    rec0 = sum(pr_disc == 0 & 
                 target[train$fold == j,]$Прочие.заболевания.сердца == 0)/sum(target[train$fold == j,]$Прочие.заболевания.сердца == 0)
    metric = 0.5*rec1 + 0.5*rec0
    res = rbind(res, data.frame(k = k/10000, metric = metric))
  }
  res5 = rbind(res5, data.frame(fold = j, res))
}
res5 = res5 %>% group_by(k) %>% summarize(metric = mean(metric))

thresh = rbind(res1[which.max(res1$metric),],
               res2[which.max(res2$metric),],
               res3[which.max(res3$metric),],
               res4[which.max(res4$metric),],
               res5[which.max(res5$metric),])

# оптимальный трешхолд и значение метрики
print(thresh)
# кросс-вал скор целевой метрики
print(mean(thresh$metric))

# Для каждого таргета прогоняем 10 итераций моделек и усредняем предикты
pr_final1 = 0
pr_final2 = 0
pr_final3 = 0
pr_final4 = 0
pr_final5 = 0
ITERS = 10
for (i in c(1:ITERS))
{
  message(i)
  dtrain <- lgb.Dataset(as.matrix(train[-c(exclude)]),label = target$Артериальная.гипертензия)
  model_lgb1 = lgb.train(data=dtrain, params = param_lgb, nrounds=NR1, bagging_seed = 13 + i, feature_fraction_seed = 42+i)
  pr1 = predict(model_lgb1, as.matrix(te[-c(exclude)]))
  dtrain <- lgb.Dataset(as.matrix(train[-c(exclude)]),label = target$ОНМК)
  model_lgb2 = lgb.train(data=dtrain, params = param_lgb, nrounds=NR2, bagging_seed = 13 + i, feature_fraction_seed = 42+i)
  pr2 = predict(model_lgb2, as.matrix(te[-c(exclude)]))
  dtrain <- lgb.Dataset(as.matrix(train[-c(exclude)]),label = target$Стенокардия..ИБС..инфаркт.миокарда)
  model_lgb3 = lgb.train(data=dtrain, params = param_lgb, nrounds=NR3, bagging_seed = 13 + i, feature_fraction_seed = 42+i)
  pr3 = predict(model_lgb3, as.matrix(te[-c(exclude)]))
  dtrain <- lgb.Dataset(as.matrix(train[-c(exclude)]),label = target$Сердечная.недостаточность)
  model_lgb4 = lgb.train(data=dtrain, params = param_lgb, nrounds=NR4, bagging_seed = 13 + i, feature_fraction_seed = 42+i)
  pr4 = predict(model_lgb4, as.matrix(te[-c(exclude)]))
  dtrain <- lgb.Dataset(as.matrix(train[-c(exclude)]),label = target$Прочие.заболевания.сердца)
  model_lgb5 = lgb.train(data=dtrain, params = param_lgb, nrounds=NR5, bagging_seed = 13 + i, feature_fraction_seed = 42+i)
  pr5 = predict(model_lgb5, as.matrix(te[-c(exclude)]))
  pr_final1 = pr_final1 + pr1
  pr_final2 = pr_final2 + pr2
  pr_final3 = pr_final3 + pr3
  pr_final4 = pr_final4 + pr4
  pr_final5 = pr_final5 + pr5
}

pr_all1 = pr_final1/ITERS
pr_all2 = pr_final2/ITERS
pr_all3 = pr_final3/ITERS
pr_all4 = pr_final4/ITERS
pr_all5 = pr_final5/ITERS

# Формируем из предиктов бинарный ответ согласно получившимся ранее оптимальным трешхолдам
sub4 = data.frame(ID = test$ID, 
                  `Артериальная.гипертензия` = ifelse(pr_all1 > thresh$k[1], 1, 0),
                  `ОНМК` = ifelse(pr_all2 > thresh$k[2], 1, 0),
                  `Стенокардия..ИБС..инфаркт.миокарда` = ifelse(pr_all3 > thresh$k[3], 1, 0),
                  `Сердечная.недостаточность` = ifelse(pr_all4 > thresh$k[4], 1, 0),
                  `Прочие.заболевания.сердца` = ifelse(pr_all5 > thresh$k[5], 1, 0))

# Пишем файлик решения
write.csv(sub4, paste0(path_to_data, "/sub4.csv", row.names = F, quote = F)