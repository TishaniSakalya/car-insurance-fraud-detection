# ---- Load Required Packages ----
library(caret)
library(rpart)  # Required for Classification Tree
library(ggplot2)
library(pROC)
library(corrplot)
library(vip)
library(rpart.plot)  # For visualizing the decision tree

# ---- Read Training and Test Data ----
train_new <- read.csv("C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Advanced Analysis project/train_smote_undersampling.csv")
test_set <- read.csv("C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Advanced Analysis project/test_set.csv")

# Convert relevant columns to factors
train_new$is_claim <- as.factor(train_new$is_claim) 
train_new$model <- as.factor(train_new$model) 
train_new$area_cluster <- as.factor(train_new$area_cluster) 

test_set$is_claim <- as.factor(test_set$is_claim)
test_set$model <- as.factor(test_set$model)
test_set$area_cluster <- as.factor(test_set$area_cluster)

# Ensure factor levels are consistent
test_set$area_cluster <- factor(test_set$area_cluster, levels = levels(train_new$area_cluster))
test_set$model <- factor(test_set$model, levels = levels(train_new$model))

set.seed(123)

# ---- Fit Classification Tree Model ----
# ---- Fit a deeper classification tree model with increased depth ----
original_tree_model <- train(is_claim ~ ., data = train_new, method = "rpart", 
                             trControl = trainControl(method = "cv", number = 5),
                             preProcess = c("center", "scale"),
                             tuneGrid = data.frame(cp = 0.01))  


# ---- Predictions & Confusion Matrix ----
original_train_preds <- predict(original_tree_model, newdata = train_new)
original_test_preds <- predict(original_tree_model, newdata = test_set)

# Set the positive class to 1
original_train_conf_matrix <- confusionMatrix(original_train_preds, train_new$is_claim)
original_test_conf_matrix <- confusionMatrix(original_test_preds, test_set$is_claim)

# ---- Prune the Tree ----
# ---- Fit a deeper classification tree model (no pruning) ----
tree_model <- rpart(is_claim ~ ., data = train_new, method = "class", 
                    control = rpart.control(maxdepth = 6, cp = 0.001))  # Deeper tree, small cp for minimal pruning

# ---- Prune the Tree ----
pruned_tree <- prune(tree_model, cp = tree_model$cptable[which.min(tree_model$cptable[, "xerror"]), "CP"])

pruned_train_preds <- predict(pruned_tree, newdata = train_new, type = "class")
pruned_test_preds <- predict(pruned_tree, newdata = test_set, type = "class")

pruned_train_conf_matrix <- confusionMatrix(pruned_train_preds, train_new$is_claim)
pruned_test_conf_matrix <- confusionMatrix(pruned_test_preds, test_set$is_claim)

# ---- Compare Accuracy for Training and Test Set ----
all_results <- data.frame(
  Tree_Type = c("Original Tree", "Original Tree", "Pruned Tree", "Pruned Tree"),
  Set = c("Train", "Test", "Train", "Test"),
  Accuracy = c(original_train_conf_matrix$overall["Accuracy"], original_test_conf_matrix$overall["Accuracy"],
               pruned_train_conf_matrix$overall["Accuracy"], pruned_test_conf_matrix$overall["Accuracy"]),
  Precision = c(original_train_conf_matrix$byClass["Precision"], original_test_conf_matrix$byClass["Precision"],
                pruned_train_conf_matrix$byClass["Precision"], pruned_test_conf_matrix$byClass["Precision"]),
  Recall = c(original_train_conf_matrix$byClass["Recall"], original_test_conf_matrix$byClass["Recall"],
             pruned_train_conf_matrix$byClass["Recall"], pruned_test_conf_matrix$byClass["Recall"]),
  F1_Score = c(original_train_conf_matrix$byClass["F1"], original_test_conf_matrix$byClass["F1"],
               pruned_train_conf_matrix$byClass["F1"], pruned_test_conf_matrix$byClass["F1"])
)

print(all_results)

# ---- Print Confusion Matrices ----
cat("\nConfusion Matrix for Original Tree - Training Set:\n")
print(original_train_conf_matrix)

cat("\nConfusion Matrix for Original Tree - Test Set:\n")
print(original_test_conf_matrix)

cat("\nConfusion Matrix for Pruned Tree - Training Set:\n")
print(pruned_train_conf_matrix)

cat("\nConfusion Matrix for Pruned Tree - Test Set:\n")
print(pruned_test_conf_matrix)


# ---- Feature Importance ----
vip(original_tree_model$finalModel)
vip(pruned_tree)

# ---- Visualize Decision Trees ----
#rpart.plot(original_tree_model$finalModel, main = "Original Decision Tree", extra = 106, cex = 0.8)
rpart.plot(pruned_tree, main = "Pruned Decision Tree", extra = 106, cex = 0.8)
