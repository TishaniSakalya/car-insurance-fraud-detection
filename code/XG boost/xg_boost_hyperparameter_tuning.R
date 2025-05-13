# Load necessary libraries
##install.packages('ParBayesianOptimization')
library(xgboost)
library(ParBayesianOptimization)
library(caret)

train_data_encoded <- read.csv("C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Advanced Analysis project/train_smote_undersampling.csv")

test_data_encoded  <- read.csv("C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Advanced Analysis project/test_set.csv")
str(test_data_encoded)
table(as.factor(test_data_encoded$is_claim))
# Convert data to matrix format for XGBoost
train_matrix <- xgb.DMatrix(data = as.matrix(train_data_encoded[,-which(names(train_data_encoded)=="is_claim")]), 
                            label = as.numeric(train_data_encoded$is_claim))

test_matrix <- xgb.DMatrix(data = as.matrix(test_data_encoded[,-which(names(test_data_encoded)=="is_claim")]), 
                           label = as.numeric(test_data_encoded$is_claim))

 
# Define Bayesian Optimization Function
set.seed(42)
bayes_optimization <- function(eta, max_depth, gamma, subsample, colsample_bytree) {
  param <- list(
    objective = "binary:logistic",
    eval_metric = "error",
    eta = eta,
    max_depth = as.integer(max_depth),
    gamma = gamma,
    subsample = subsample,
    colsample_bytree = colsample_bytree
  )
  
  # Perform Cross-Validation to Evaluate Performance
  cv_result <- xgb.cv(params = param,
                      data = train_matrix,
                      nrounds = 100,
                      nfold = 5,
                      verbose = FALSE)
  
  # Return best log-loss score (negative for maximization)
  return(list(Score = -min(cv_result$evaluation_log$test_error_mean), Pred = 0))
}

# Perform Bayesian Optimization
opt_results <- bayesOpt(
  FUN = bayes_optimization,
  bounds = list(eta = c(0.01, 0.3), max_depth = c(3, 10), gamma = c(0, 5), 
                subsample = c(0.5, 1), colsample_bytree = c(0.5, 1)),
  initPoints = 10,
  iters.n = 30
)

# Extract Best Parameters
best_params <- getBestPars(opt_results)
print(best_params)
# Train Final XGBoost Model Using Optimized Parameters
final_model <- xgboost(data = train_matrix, 
                       objective = "binary:logistic",
                       eval_metric = "error",
                       eta = best_params$eta,
                       max_depth = as.integer(best_params$max_depth),
                       gamma = best_params$gamma,
                       subsample = best_params$subsample,
                       colsample_bytree = best_params$colsample_bytree,
                       nrounds = 100)

# ---- Training Set Accuracy ----
train_pred <- predict(final_model, train_matrix)
train_pred_labels <- ifelse(train_pred > 0.5, 1, 0)
train_accuracy <- sum(train_pred_labels == train_data_encoded$is_claim) / nrow(train_data_encoded)
print(paste("Training Accuracy:", round(train_accuracy * 100, 2), "%"))

# ---- Test Set Accuracy ----
test_pred <- predict(final_model, test_matrix)
test_pred_labels <- ifelse(test_pred > 0.5, 1, 0)
test_accuracy <- sum(test_pred_labels == test_data_encoded$is_claim) / nrow(test_data_encoded)
print(paste("Test Accuracy:", round(test_accuracy * 100, 2), "%"))


# ---- Training Set Performance ----
train_conf_matrix <- confusionMatrix(as.factor(train_pred_labels), as.factor(train_data_encoded$is_claim))


# ---- Test Set Performance ----
test_conf_matrix <- confusionMatrix(as.factor(test_pred_labels), as.factor(test_data_encoded$is_claim))


# ---- Extract Precision, Recall, and F1-Score ----
train_precision <- train_conf_matrix$byClass["Pos Pred Value"]
train_recall <- train_conf_matrix$byClass["Sensitivity"]
train_f1 <- 2 * ((train_precision * train_recall) / (train_precision + train_recall))

test_precision <- test_conf_matrix$byClass["Pos Pred Value"]
test_recall <- test_conf_matrix$byClass["Sensitivity"]
test_f1 <- 2 * ((test_precision * test_recall) / (test_precision + test_recall))

# Print Performance Metrics
cat("\nTraining Metrics:\n")
cat("Precision:", round(train_precision, 4), "\n")
cat("Recall:", round(train_recall, 4), "\n")
cat("F1-Score:", round(train_f1, 4), "\n")

cat("\nTest Metrics:\n")
cat("Precision:", round(test_precision, 4), "\n")
cat("Recall:", round(test_recall, 4), "\n")
cat("F1-Score:", round(test_f1, 4), "\n")

# ---- Feature Importance ----
importance_matrix <- xgb.importance(feature_names = colnames(train_data_encoded[,-which(names(train_data_encoded)=="is_claim")]), model = final_model)
print("Feature Importance:")
print(importance_matrix)

# Create a new Category column with general group names
importance_matrix$Category <- importance_matrix$Feature

# Group 'modelM1', 'modelM2', ... into 'model'
importance_matrix$Category <- sub("^model.*", "model", importance_matrix$Category)

# Group 'area_clusterHigh', 'area_clusterLow', 'area_clusterMedium' into 'area_cluster'
importance_matrix$Category <- sub("^area_cluster.*", "area_cluster", importance_matrix$Category)


library(dplyr)

# Sum feature importance scores for each category
importance_by_category <- importance_matrix %>%
  group_by(Category) %>%
  summarise(Importance = sum(Gain)) %>%  # Summing the 'Gain' importance scores
  arrange(desc(Importance))

# Print the summarized importance
print(importance_by_category)



library(ggplot2)

ggplot(importance_by_category, aes(x = reorder(Category, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance by Category",
       x = "Feature Category",
       y = "Importance") +
  theme_minimal()




# Load necessary library
##install.packages("pROC")
library(pROC)

# Compute ROC curve for training set
train_roc <- roc(train_data_encoded$is_claim, train_pred)
plot(train_roc, col = "blue", main = "ROC Curve - Training vs Test")
auc_train <- auc(train_roc)
legend("bottomright", legend = paste("Train AUC:", round(auc_train, 4)), col = "blue", lwd = 2)

# Compute ROC curve for test set
test_roc <- roc(test_data_encoded$is_claim, test_pred)
plot(test_roc, col = "red", add = TRUE)
auc_test <- auc(test_roc)
legend("bottomright", legend = c(paste("Train AUC:", round(auc_train, 4)), 
                                 paste("Test AUC:", round(auc_test, 4))), 
       col = c("blue", "red"), lwd = 2)



# Variable importance plot
varImpPlot(rf_model)

library(pdp)
library(ggplot2)


# Compute partial dependence for age_of_policyholder (Probability of is_claim = 1)
pdp_age <- partial(final_model, pred.var = "age_of_policyholder", train = train_data, prob = TRUE)
ggplot(pdp_age, aes(x = age_of_policyholder, y = yhat)) +
  geom_line() +
  ggtitle("Partial Dependence: Age of Policyholder") +
  xlab("Age of Policyholder") +
  ylab("Predicted Probability of Claim") +
  theme_minimal()

# Compute partial dependence for age_of_car (Probability of is_claim = 1)
pdp_car <- partial(final_model, pred.var = "age_of_car", train = train_data, prob = TRUE)
ggplot(pdp_car, aes(x = age_of_car, y = yhat)) +
  geom_line() +
  ggtitle("Partial Dependence: Age of Car") +
  xlab("Age of Car") +
  ylab("Predicted Probability of Claim") +
  theme_minimal()

# Compute partial dependence for policy_tenure (Probability of is_claim = 1)
pdp_tenure <- partial(final_model, pred.var = "policy_tenure", train = train_data, prob = TRUE)
ggplot(pdp_tenure, aes(x = policy_tenure, y = yhat)) +
  geom_line() +
  ggtitle("Partial Dependence: Policy Tenure") +
  xlab("Policy Tenure") +
  ylab("Predicted Probability of Claim") +
  theme_minimal()
