### Load Required Libraries
install.packages(c("tidyverse", "cluster", "factoextra", "ggplot2", "caret", "randomForest"))
library(tidyverse)
library(cluster)
library(factoextra)
library(ggplot2)
library(caret)
library(randomForest)

# Load Encoded Dataset
data <- read.csv("C:/Users/User/Desktop/Colombo uni/3year - 2nd sem/ST 3082/2nd_project/train_encoded.csv")  # Ensure dataset is correctly loaded

### 1️⃣ Sample Data to Reduce Computation Load
set.seed(123)
sample_size <- min(10000, nrow(data))  # Limit to 10,000 observations to manage memory usage
sample_indices <- sample(1:nrow(data), size = sample_size)
data_sampled <- data[sample_indices, ]

### 2️⃣ Data Preprocessing for Clustering
# Identify categorical feature columns
area_cluster_cols <- grep("^area_cluster", names(data_sampled), value = TRUE)
model_cols <- grep("^model", names(data_sampled), value = TRUE)

# Select relevant numeric features for risk assessment
risk_features <- data_sampled %>% select(age_of_policyholder, policy_tenure, age_of_car, is_claim, all_of(area_cluster_cols), all_of(model_cols))

# Convert columns to numeric and handle missing values
risk_features <- risk_features %>% mutate(across(everything(), as.numeric))
risk_features <- risk_features %>% mutate(across(everything(), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Normalize data
risk_features_scaled <- scale(risk_features)

### 3️⃣ Apply k-Means Clustering to Segment Policyholders
set.seed(123)
kmeans_risk <- kmeans(risk_features_scaled, centers = 3, nstart = 10, iter.max = 100)

# Assign risk levels to dataset
data_sampled$Risk_Level <- as.factor(kmeans_risk$cluster)

### 4️⃣ Train Random Forest Model for Risk Prediction
rf_features <- c("age_of_policyholder", "policy_tenure", "age_of_car", area_cluster_cols, model_cols)
rf_model_risk <- randomForest(as.formula(paste("Risk_Level ~", paste(rf_features, collapse = " + "))),
                              data = data_sampled, ntree = 100, mtry = 3, importance = TRUE)

# View Model Summary
print(rf_model_risk)

# Predict risk levels for new policyholders
data_sampled$Predicted_Risk <- predict(rf_model_risk, data_sampled)

### 5️⃣ Visualizing Risk Segmentation
# Distribution of Risk Levels by Policy Tenure
ggplot(data_sampled, aes(x = policy_tenure, fill = Risk_Level)) +
  geom_histogram(bins = 30, alpha = 0.7) +
  labs(title = "Policy Tenure Distribution by Risk Level", x = "Policy Tenure (Years)", y = "Count") +
  theme_minimal()

# Bar Plot of Claim Rate Across Risk Levels
ggplot(data_sampled, aes(x = Risk_Level, fill = as.factor(is_claim))) +
  geom_bar(position = "fill") +
  labs(title = "Claim Rate by Risk Level", x = "Risk Level", y = "Proportion of Claims", fill = "Claim Status") +
  theme_minimal()

# Scatter Plot to Visualize Clusters
ggplot(data_sampled, aes(x = age_of_policyholder, y = policy_tenure, color = Risk_Level)) +
  geom_point(alpha = 0.6) +
  labs(title = "Clustered Policyholders by Age and Policy Tenure", x = "Age of Policyholder", y = "Policy Tenure (Years)") +
  theme_minimal()

# 2D Cluster Plot Using PCA Projection
fviz_cluster(kmeans_risk, data = risk_features_scaled, geom = "point")
