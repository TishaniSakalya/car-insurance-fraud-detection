# Load necessary libraries
library(randomForest)
library(caret)  # For evaluation metrics

# Load your pre-split training and testing datasets
train_data <- read.csv("C:/Users/Hiruni/OneDrive/Desktop/train_smote_without_OHE.csv")
str(train_data)
test_data  <- read.csv("C:/Users/Hiruni/OneDrive/Desktop/test_set.csv")

# Convert target variable 'is_claim' to factor (for classification) 
train_data$is_claim <- as.factor(train_data$is_claim)
train_data$model <- as.factor(train_data$model)
train_data$area_cluster<- as.factor(train_data$area_cluster)
test_data$is_claim  <- as.factor(test_data$is_claim)
test_data$model  <- as.factor(test_data$model)
test_data$area_cluster  <- as.factor(test_data$area_cluster)

# Train the Random Forest model
set.seed(42)  # For reproducibility
rf_model <- randomForest(is_claim ~ ., 
                         data = train_data, 
                         ntree = 100,     # Number of trees
                         mtry = sqrt(ncol(train_data) - 1),  # Number of features per split
                         importance = TRUE)  # Track variable importance

# Model summary
print(rf_model)

# Make predictions on the training set
y_train_pred <- predict(rf_model, train_data)

# Compute confusion matrix for training set
train_conf_matrix <- confusionMatrix(y_train_pred, train_data$is_claim)
train_accuracy <- train_conf_matrix$overall["Accuracy"]
train_precision <- train_conf_matrix$byClass["Precision"]
train_recall <- train_conf_matrix$byClass["Recall"]
train_f1 <- train_conf_matrix$byClass["F1"]

# Print training performance metrics
cat("\nTraining Set Performance:\n")
cat("Accuracy:", round(train_accuracy, 4), "\n")
cat("Precision:", round(train_precision, 4), "\n")
cat("Recall:", round(train_recall, 4), "\n")
cat("F1-Score:", round(train_f1, 4), "\n")


# Make predictions on the test set
y_test_pred <- predict(rf_model, test_data)

# Compute confusion matrix for test set
test_conf_matrix <- confusionMatrix(y_test_pred, test_data$is_claim)
test_accuracy <- test_conf_matrix$overall["Accuracy"]
test_precision <- test_conf_matrix$byClass["Precision"]
test_recall <- test_conf_matrix$byClass["Recall"]
test_f1 <- test_conf_matrix$byClass["F1"]

# Print test performance metrics
cat("\nTest Set Performance:\n")
cat("Accuracy:", round(test_accuracy, 4), "\n")
cat("Precision:", round(test_precision, 4), "\n")
cat("Recall:", round(test_recall, 4), "\n")
cat("F1-Score:", round(test_f1, 4), "\n")

# Variable importance plot
varImpPlot(rf_model)



