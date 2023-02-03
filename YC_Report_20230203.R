## ---- warning = FALSE, message = FALSE-------------------------------------------------------------------------------
if(!require(librarian)) install.packages("librarian", repos = "http://cran.us.r-project.org")
library(librarian)

librarian::shelf("DataExplorer")
librarian::shelf("tidyverse")
librarian::shelf("caret")
librarian::shelf("data.table")
librarian::shelf("PerformanceAnalytics")
librarian::shelf("corrplot")
librarian::shelf("janitor")


## --------------------------------------------------------------------------------------------------------------------
edx <- fread ("diabetes.csv")


## --------------------------------------------------------------------------------------------------------------------
str(edx)
names(edx)
dim(edx)
summary(edx)


## --------------------------------------------------------------------------------------------------------------------
for (i in c("Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",         "BMI", "DiabetesPedigreeFunction","Age")){
  p <- edx %>% 
    count(.data[[i]]) %>% ggplot(aes(n))+
  geom_histogram(bins = 20 , color = "black", fill = "gray")+
  scale_x_log10()+
  ggtitle(paste0("Distribution of ", i)) +
    theme_classic()
  print(p)
}


## --------------------------------------------------------------------------------------------------------------------
ggplot(stack(edx[, 1:8]), aes(x = ind, y = values, fill = ind)) + 
  geom_boxplot() +
  labs(title = "Boxplots of the predictor variables") + 
  labs(x = "", y = "Values") +
  theme(axis.text.x = element_text(
    angle = 45,
    hjust = 1,
    vjust = 1
  )) 


## ---- fig.height=7---------------------------------------------------------------------------------------------------
chart.Correlation(edx[, 1:8], histogram=TRUE, pch=19)


## --------------------------------------------------------------------------------------------------------------------
plot_missing(edx)


## --------------------------------------------------------------------------------------------------------------------
# A method from caret
set.seed(123)

inTrain = createDataPartition(y = edx$Outcome, p = .80, list = FALSE)
train_set = edx[inTrain,]
test_set = edx[-inTrain,] 


## --------------------------------------------------------------------------------------------------------------------
RMSE <- function(true_value, predicted_value){
  sqrt(mean((true_value - predicted_value)^2, na.rm = TRUE))
}


## --------------------------------------------------------------------------------------------------------------------
trControl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 10)


## ---- fig.height=7---------------------------------------------------------------------------------------------------
set.seed(123)
fit_gbm <- train(Outcome ~ ., data = train_set, 
                 method = "gbm", 
                 trControl = trControl,
                 verbose = FALSE) 
fit_gbm
summary(fit_gbm)
plot(fit_gbm)

# predict and RMSE
predictions <- predict(fit_gbm, test_set)
RMSE_gbm <- RMSE(predictions, test_set$Outcome)

# A result table is created to collect the RMSE results from different modelling
rmse_results <- data.frame(Method = "gbm", RMSE = RMSE_gbm) %>%  print()


## --------------------------------------------------------------------------------------------------------------------
set.seed(123)
fit_cart <- train(Outcome~., data = train_set, method = "rpart", trControl = trControl)
fit_cart
plot(fit_cart)

# predict and RMSE
predictions <- predict(fit_cart, test_set)
RMSE_cart <- RMSE(predictions, test_set$Outcome)

# The result table is expanded to collect the RMSE results from different modelling
rmse_results <- bind_rows(rmse_results, tibble(Method="cart",  RMSE = RMSE_cart)) %>%  
  arrange(RMSE) %>% 
  print()


## --------------------------------------------------------------------------------------------------------------------
set.seed(123)
fit_knn <- train(Outcome~., data = train_set, method = "knn", trControl = trControl)
fit_knn
plot(fit_knn)

# predict and RMSE
predictions <- predict(fit_knn, test_set)
RMSE_knn <- RMSE(predictions, test_set$Outcome)

# The result table is expanded to collect the RMSE results from different modelling
rmse_results <- bind_rows(rmse_results, tibble(Method="knn",  RMSE = RMSE_knn)) %>%  
  arrange(RMSE) %>% 
  print()


## --------------------------------------------------------------------------------------------------------------------
set.seed(123)
fit_svm <- train(Outcome~., data = train_set, method = "svmRadial", trControl = trControl)
fit_svm
plot(fit_svm)

# predict and RMSE
predictions <- predict(fit_svm, test_set)
RMSE_svm <- RMSE(predictions, test_set$Outcome)

# The result table is expanded to collect the RMSE results from different modelling
rmse_results <- bind_rows(rmse_results, tibble(Method="svm",  RMSE = RMSE_svm)) %>%  
  arrange(RMSE) %>% 
  print()


## --------------------------------------------------------------------------------------------------------------------
set.seed(123)
fit_rf <- train(Outcome~., data = train_set, method = "ranger", trControl = trControl)
fit_rf
plot(fit_rf)

# predict and RMSE
predictions <- predict(fit_rf, test_set)
RMSE_rf <- RMSE(predictions, test_set$Outcome)

# The result table is expanded to collect the RMSE results from different modelling
rmse_results <- bind_rows(rmse_results, tibble(Method="rf",  RMSE = RMSE_rf)) %>%  
  arrange(RMSE) 

rmse_results


## --------------------------------------------------------------------------------------------------------------------

results <- resamples(list(GBM=fit_gbm, SVM=fit_svm, CART=fit_cart, KNN=fit_knn, RF=fit_rf))

summary(results)

bwplot(results)

dotplot(results)


## --------------------------------------------------------------------------------------------------------------------
pkgs <- c("RCurl","jsonlite")
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
}
install.packages("h2o", type="source", repos=(c("http://h2o-release.s3.amazonaws.com/h2o/latest_stable_R")))


## --------------------------------------------------------------------------------------------------------------------
library(h2o)

# Start the H2O cluster (locally)
h2o.init()

# import datasets
train_set2 <- as.h2o(train_set)
test_set2 <- as.h2o(test_set)

# Identify predictors and response
y <- "Outcome"
x <- setdiff(names(train_set2), y)

# Run AutoML for mutliple base models
aml <- h2o.automl(x = x, y = y,
                  training_frame = train_set2,
                  max_models = 5,
                  seed = 1)

# View the AutoML Leaderboard
lb <- aml@leaderboard
print(lb, n = nrow(lb))  # Print all rows instead of default (6 rows)

# The leader model is stored here
fit_aml <- aml@leader
fit_aml



## --------------------------------------------------------------------------------------------------------------------
# retrieve the model performance
perf <- h2o.performance(fit_aml, test_set2)
perf
RMSE_aml <- perf@metrics$RMSE

# The result table is expanded to collect the RMSE results from different modelling
rmse_results <- bind_rows(rmse_results, tibble(Method="H2O’s AutoML",  RMSE = RMSE_aml)) %>%  
  arrange(RMSE)


## --------------------------------------------------------------------------------------------------------------------
DiffValue = c(NA, diff(rmse_results$RMSE))
rmse_results %>% mutate(Difference = DiffValue) %>%
  replace (is.na(.), "") %>% 
   print()

