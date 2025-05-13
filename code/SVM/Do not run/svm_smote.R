
# ---- Load Required Packages ----
library(caret)
library(e1071)  # Required for SVM
library(ggplot2)
library(pROC)
library(corrplot)
library(vip)

# ---- Read Training Data ----
train_new <- read.csv("C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Advanced Analysis project/train_smote_without_OHE.csv")

# Convert relevant columns to factors
train_new$is_claim <- as.factor(train_new$is_claim) 
train_new$model <- as.factor(train_new$model) 
train_new$area_cluster <- as.factor(train_new$area_cluster) 

set.seed(123)

# Fit Support Vector Machine Model
svm_model <- train(is_claim ~ ., data = train_new, method = "svmRadial", 
                   trControl = trainControl(method = "cv", number = 5),  # 5-fold cross-validation
                   preProcess = c("center", "scale"))

# ---- Accuracy Check for Training Set ----
train_preds <- predict(svm_model, newdata = train_new)  # Predict labels

# Compute confusion matrix
train_conf_matrix <- confusionMatrix(train_preds, train_new$is_claim)

# Extract metrics for training set
train_results <- data.frame(
  Set = "Training",
  Accuracy = train_conf_matrix$overall["Accuracy"],
  Precision = train_conf_matrix$byClass["Precision"],
  Recall = train_conf_matrix$byClass["Recall"],
  F1_Score = train_conf_matrix$byClass["F1"]
)

# ---- Accuracy Check for Test Set ----
test_set <- read.csv("C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Advanced Analysis project/test_set.csv")

# Convert relevant columns to factors
test_set$model <- as.factor(test_set$model) 
test_set$area_cluster <- as.factor(test_set$area_cluster) 

# Predict labels on test set
test_preds <- predict(svm_model, newdata = test_set)

test_set$is_claim <- factor(test_set$is_claim, levels = c(0, 1))
test_set$predicted_is_claim <- test_preds

# Save results to a CSV file
write.csv(test_set, "C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Advanced Analysis project/predicted_test_results_svm.csv", row.names = FALSE)

# Compute confusion matrix
test_conf_matrix <- confusionMatrix(test_set$predicted_is_claim, test_set$is_claim)

# Extract metrics for test set
test_results <- data.frame(
  Set = "Test",
  Accuracy = test_conf_matrix$overall["Accuracy"],
  Precision = test_conf_matrix$byClass["Precision"],
  Recall = test_conf_matrix$byClass["Recall"],
  F1_Score = test_conf_matrix$byClass["F1"]
)

# ---- Combine and Print Results with Proper Row Names ----
final_results <- rbind(train_results, test_results)
rownames(final_results) <- NULL  # Remove default row names
print(final_results)

# ---- Confusion Matrix Heatmap ----
conf_matrix_table <- as.data.frame(test_conf_matrix$table)
colnames(conf_matrix_table) <- c("Actual", "Predicted", "Freq")  # Ensure correct column names

ggplot(conf_matrix_table, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Confusion Matrix Heatmap", x = "Predicted", y = "Actual") +
  theme_minimal()

# ---- Feature Importance using Surrogate Model ----
library(caret)
library(ggplot2)

set.seed(123)

# Train a surrogate decision tree model
importance_model <- train(is_claim ~ ., data = train_new, method = "rpart",
                          trControl = trainControl(method = "cv", number = 5))

# Extract importance
feature_importance <- varImp(importance_model)$importance

# Remove category suffixes for categorical variables
feature_importance$Feature <- gsub("\\..*", "", rownames(feature_importance))  

# Aggregate importance by feature
feature_importance <- aggregate(Overall ~ Feature, data = feature_importance, FUN = mean)

# Select the top 6 features
feature_importance <- feature_importance[order(-feature_importance$Overall), ][1:6, ]

# Plot Feature Importance
ggplot(feature_importance, aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 6 Feature Importance (Surrogate Tree)", x = "Features", y = "Importance") +
  theme_minimal()
