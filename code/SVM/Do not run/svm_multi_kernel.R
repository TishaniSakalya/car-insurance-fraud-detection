# ---- Load Required Packages ----
library(caret)
library(e1071)  # For SVM functions
library(ggplot2)
library(pROC)

# ---- Read Training Data ----
train_new <- read.csv("C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Advanced Analysis project/train_undersampling_without_OHE.csv")

# Convert relevant columns to factors
train_new$is_claim <- as.factor(train_new$is_claim) 
train_new$model <- as.factor(train_new$model) 
train_new$area_cluster <- as.factor(train_new$area_cluster) 

# Define kernel mapping for caret
kernel_mapping <- list(
  radial = "svmRadial",
  polynomial = "svmPoly",
  linear = "svmLinear"
)

# Create an empty list to store results
model_results <- list()

set.seed(123)

# ---- Train SVM Models with Different Kernels ----
for (kernel in names(kernel_mapping)) {
  
  cat("\nTraining SVM with", kernel, "kernel...\n")
  
  # Train SVM Model
  svm_model <- train(is_claim ~ ., data = train_new, method = kernel_mapping[[kernel]],
                     trControl = trainControl(method = "cv", number = 5),
                     preProcess = c("center", "scale"),
                     tuneLength = 10)  # Auto-tune hyperparameters
  
  # ---- Accuracy Check for Training Set ----
  train_preds <- predict(svm_model, newdata = train_new)
  train_conf_matrix <- confusionMatrix(train_preds, train_new$is_claim)
  
  # ---- Read Test Set ----
  test_set <- read.csv("C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Advanced Analysis project/test_set.csv")
  
  # Convert relevant columns to factors
  test_set$model <- as.factor(test_set$model) 
  test_set$area_cluster <- as.factor(test_set$area_cluster) 
  
  # Predict labels on test set
  test_preds <- predict(svm_model, newdata = test_set)
  
  # Ensure factor levels match
  test_set$is_claim <- factor(test_set$is_claim, levels = levels(train_new$is_claim))
  test_set$predicted_is_claim <- test_preds
  
  # Compute confusion matrix for test set
  test_conf_matrix <- confusionMatrix(test_set$predicted_is_claim, test_set$is_claim)
  
  # Store results
  model_results[[kernel]] <- data.frame(
    Kernel = kernel,
    Train_Accuracy = train_conf_matrix$overall["Accuracy"],
    Test_Accuracy = test_conf_matrix$overall["Accuracy"],
    Precision = test_conf_matrix$byClass["Precision"],
    Recall = test_conf_matrix$byClass["Recall"],
    F1_Score = test_conf_matrix$byClass["F1"]
  )
}

# ---- Handle Additional Kernels (RBF and Polynomial) ----
# RBF Kernel (Already included as svmRadial in the previous block)
# Polynomial Kernel
cat("\nTraining SVM with Polynomial Kernel...\n")
svm_poly_model <- train(is_claim ~ ., data = train_new, method = "svmPoly",
                        trControl = trainControl(method = "cv", number = 5),
                        preProcess = c("center", "scale"),
                        tuneLength = 10)  # Auto-tune hyperparameters

# ---- Accuracy Check for Polynomial Kernel ----
train_preds <- predict(svm_poly_model, newdata = train_new)
train_conf_matrix <- confusionMatrix(train_preds, train_new$is_claim)

# ---- Read Test Set ----
test_set <- read.csv("C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Advanced Analysis project/test_set.csv")
test_set$model <- as.factor(test_set$model) 
test_set$area_cluster <- as.factor(test_set$area_cluster) 

# Predict labels on test set
test_preds <- predict(svm_poly_model, newdata = test_set)

# Ensure factor levels match
test_set$is_claim <- factor(test_set$is_claim, levels = levels(train_new$is_claim))
test_set$predicted_is_claim <- test_preds

# Compute confusion matrix for test set
test_conf_matrix <- confusionMatrix(test_set$predicted_is_claim, test_set$is_claim)

# Store polynomial kernel results
model_results[["polynomial"]] <- data.frame(
  Kernel = "polynomial",
  Train_Accuracy = train_conf_matrix$overall["Accuracy"],
  Test_Accuracy = test_conf_matrix$overall["Accuracy"],
  Precision = test_conf_matrix$byClass["Precision"],
  Recall = test_conf_matrix$byClass["Recall"],
  F1_Score = test_conf_matrix$byClass["F1"]
)

# ---- Combine and Print Results ----
final_results <- do.call(rbind, model_results)
rownames(final_results) <- NULL
print(final_results)

# ---- Plot Results ----
ggplot(final_results, aes(x = Kernel, y = Test_Accuracy, fill = Kernel)) +
  geom_bar(stat = "identity") +
  labs(title = "SVM Kernel Comparison", x = "Kernel", y = "Test Accuracy") +
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

