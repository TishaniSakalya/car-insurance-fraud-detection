### Load Required Libraries
install.packages(c("tidyverse", "cluster", "factoextra", "ggplot2", "caret", "randomForest", "fastcluster", "dbscan", "NbClust", "writexl"))
library(tidyverse)
library(cluster)
library(factoextra)
library(ggplot2)
library(caret)
library(randomForest)
library(fastcluster)  # Optimized hierarchical clustering
library(dbscan)       # For density-based clustering
library(NbClust)      # For optimal cluster determination
library(writexl)      # Export results to Excel

### Load Dataset
data <- read.csv("C:/Users/User/Desktop/Colombo uni/3year - 2nd sem/ST 3082/2nd_project/train_encoded.csv")

### Optimized Sampling (For Performance)
set.seed(123)
sample_size <- min(5000, nrow(data))  # Reduce sample size for speed
sample_indices <- sample(1:nrow(data), size = sample_size)
data_sampled <- data[sample_indices, ]

### Feature Selection & Preprocessing
selected_features <- c("age_of_policyholder", "policy_tenure", "age_of_car", "is_claim")
risk_features <- data_sampled %>% select(all_of(selected_features))

# Convert to Numeric and Handle Missing Values
risk_features <- risk_features %>% mutate(across(everything(), as.numeric))
risk_features <- risk_features %>% mutate(across(everything(), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Normalize Data
risk_features_scaled <- scale(risk_features)

### Dimensionality Reduction (PCA)
pca_result <- prcomp(risk_features_scaled, scale. = TRUE)
risk_features_pca <- pca_result$x[, 1:2]  # Use the first two principal components

### Determine Optimal Number of Clusters Using Multiple Methods
fviz_nbclust(risk_features_pca, kmeans, method = "wss") + 
  labs(title = "Elbow Method for Optimal Number of Clusters")

fviz_nbclust(risk_features_pca, kmeans, method = "silhouette") + 
  labs(title = "Silhouette Method for Optimal Cluster Selection")

gap_stat <- clusGap(risk_features_pca, FUN = kmeans, nstart = 25, K.max = 6, B = 50)
fviz_gap_stat(gap_stat)  # Gap statistic for cluster quality evaluation

### Improved Clustering Approaches

# 1️⃣ K-Means Clustering (Baseline)
set.seed(123)
num_clusters <- 3  # Adjust based on Elbow & Silhouette methods
kmeans_result <- kmeans(risk_features_pca, centers = num_clusters, nstart = 25)
data_sampled$Risk_Level_KMeans <- as.factor(kmeans_result$cluster)

# 2️⃣ Hierarchical Clustering (Alternative)
hc_result <- hclust(dist(risk_features_scaled), method = "ward.D2")
data_sampled$Risk_Level_HC <- as.factor(cutree(hc_result, k = num_clusters))

# 3️⃣ DBSCAN (Density-Based Clustering for Anomaly Detection)
dbscan_result <- dbscan(risk_features_scaled, eps = 0.5, minPts = 5)
data_sampled$Risk_Level_DBSCAN <- as.factor(dbscan_result$cluster)

### Cluster Evaluation
silhouette_kmeans <- silhouette(kmeans_result$cluster, dist(risk_features_scaled))
silhouette_hc <- silhouette(cutree(hc_result, k = num_clusters), dist(risk_features_scaled))

cat("\nSilhouette Score for K-Means:", mean(silhouette_kmeans[, 3]))
cat("\nSilhouette Score for Hierarchical Clustering:", mean(silhouette_hc[, 3]))

### Summary Table for Clusters
cluster_summary <- data_sampled %>%
  group_by(Risk_Level_KMeans) %>%
  summarise(
    Count = n(),  
    Avg_Age = mean(age_of_policyholder),
    Avg_Policy_Tenure = mean(policy_tenure),
    Avg_Car_Age = mean(age_of_car),
    Claim_Rate = mean(is_claim),  
    SD_Age = sd(age_of_policyholder),
    SD_Policy_Tenure = sd(policy_tenure),
    SD_Car_Age = sd(age_of_car)
  )

print(cluster_summary)

# Export K-means summary table to Excel
write_xlsx(cluster_summary, path = "C:/Users/User/Desktop/Colombo uni/3year - 2nd sem/ST 3082/2nd_project/kmeans_cluster_summary.xlsx")

### Visualizing Risk Segmentation

# 1️⃣ K-Means Cluster Visualization (PCA-Based)
fviz_cluster(kmeans_result, data = risk_features_pca, geom = "point") +
  labs(title = "Risk Segmentation Using K-Means Clustering")

# 2️⃣ Hierarchical Clustering Dendrogram
fviz_dend(hc_result, k = num_clusters, rect = TRUE, show_labels = FALSE) +
  labs(title = "Hierarchical Clustering Dendrogram for Risk Segmentation")

# 3️⃣ Policy Tenure Distribution by Risk Level
ggplot(data_sampled, aes(x = policy_tenure, fill = Risk_Level_KMeans)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  labs(title = "Policy Tenure Distribution by Risk Level", x = "Policy Tenure (Years)", y = "Count") +
  theme_minimal()

# 4️⃣ Claim Rate by Risk Level
ggplot(data_sampled, aes(x = Risk_Level_KMeans, fill = as.factor(is_claim))) +
  geom_bar(position = "fill") +
  labs(title = "Claim Rate by Risk Level", x = "Risk Level", y = "Proportion of Claims", fill = "Claim Status") +
  theme_minimal()

# 5️⃣ Scatter Plot (Age vs. Policy Tenure, Colored by Risk Level)
ggplot(data_sampled, aes(x = age_of_policyholder, y = policy_tenure, color = Risk_Level_KMeans)) +
  geom_point(alpha = 0.6) +
  labs(title = "Clustered Policyholders by Age and Policy Tenure", x = "Age of Policyholder", y = "Policy Tenure (Years)") +
  theme_minimal()
