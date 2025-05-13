# Load necessary libraries
library(randomForest)
library(caret)  # For evaluation metrics

# Load your pre-split training and testing datasets
train_data <- read.csv("C:/Users/Hiruni/OneDrive/Desktop/train_set.csv")
test_data  <- read.csv("C:/Users/Hiruni/OneDrive/Desktop/test_set.csv")

# Convert target variable 'is_claim' to factor (for classification)
train_data$is_claim <- as.factor(train_data$is_claim)
test_data$is_claim  <- as.factor(test_data$is_claim)

# Check class distribution in train and test datasets
print(table(train_data$is_claim))
print(table(test_data$is_claim))

# Train the Random Forest model with class weights (adjust as necessary)
set.seed(42)  # For reproducibility
rf_model <- randomForest(is_claim ~ ., 
                         data = train_data,
                         classwt = c("0" = 0.063, "1" = 0.937),# Adjust based on the class imbalance
                         ntree = 100,     # Number of trees
                         mtry = sqrt(ncol(train_data) - 1),  # Number of features per split
                         importance = TRUE)  # Track variable importance

# Model summary (Random Forest details)
print(rf_model)

# Make predictions on the test set
y_pred <- predict(rf_model, test_data)

# Evaluate model performance on the test set
test_conf_matrix <- confusionMatrix(y_pred, test_data$is_claim)


# Model performance on the training set
y_train_pred <- predict(rf_model, train_data)
train_conf_matrix <- confusionMatrix(y_train_pred, train_data$is_claim)


# Print accuracy for training set
train_accuracy <- train_conf_matrix$overall["Accuracy"]
print(paste("Training Accuracy: ", round(train_accuracy, 4)))

# Print accuracy for test set
test_accuracy <- test_conf_matrix$overall["Accuracy"]
print(paste("Test Accuracy: ", round(test_accuracy, 4)))

# Print Precision, Recall, F1-Score, etc., for both training and test sets
print("Training Set Performance Metrics:")
print(train_conf_matrix$byClass[c("Precision", "Recall", "F1")])

print("Test Set Performance Metrics:")
print(test_conf_matrix$byClass[c("Precision", "Recall", "F1")])

# Variable importance plot
varImpPlot(rf_model, main = "Variable Importance Plot")
