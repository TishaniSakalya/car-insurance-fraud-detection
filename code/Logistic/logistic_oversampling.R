# ---- Load Required Packages ----
library(caret)

# ---- Read Training Data ----
train_new <- read.csv("C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Advanced Analysis project/train_oversampling_without_OHE.csv")

# Convert relevant columns to factors
train_new$is_claim <- as.factor(train_new$is_claim) 
train_new$model <- as.factor(train_new$model) 
train_new$area_cluster <- as.factor(train_new$area_cluster) 

set.seed(123)

# Fit Logistic Regression Model
logistic_model <- glm(is_claim ~ ., data = train_new, family = binomial)

# ---- Accuracy Check for Training Set ----
train_probs <- predict(logistic_model, newdata = train_new, type = "response") # Predict probabilities
train_preds <- ifelse(train_probs > 0.5, 1, 0)  # Convert to binary (Threshold = 0.5)
train_preds <- factor(train_preds, levels = c(0, 1))  # Convert to factor

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

# Predict probabilities on test set
test_probs <- predict(logistic_model, test_set, type = "response")
test_preds <- ifelse(test_probs > 0.5, 1, 0)  # Convert to binary
test_preds <- factor(test_preds, levels = c(0, 1))  # Convert to factor

# Ensure factor levels match
test_set$is_claim <- factor(test_set$is_claim, levels = c(0, 1))
test_set$predicted_is_claim <- test_preds


# Save results to a CSV file
write.csv(test_set, "C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Advanced Analysis project/predicted_test_results_oversampling.csv", row.names = FALSE)


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




