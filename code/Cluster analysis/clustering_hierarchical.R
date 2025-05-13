
# Install required packages
install.packages(c("tidyverse", "cluster", "factoextra", "ggplot2", "caret", "randomForest", "fastcluster", "clustMixType", "daisy"))

update.packages(ask = FALSE)  # Update all installed packages
install.packages(c("tidyverse", "ggplot2", "tidyr", "purrr", "dplyr", "lubridate"), dependencies = TRUE)

library(tidyverse)
library(cluster)
library(factoextra)
library(ggplot2)
library(caret)
library(randomForest)
library(fastcluster)
library(clustMixType)  # For K-Prototypes clustering
library(daisy)  # For Gower Distance

# Load Encoded Dataset
data <- read.csv("C:/Users/User/Desktop/Colombo uni/3year - 2nd sem/ST 3082/2nd_project/train_set.csv")

### Optimized Sampling
set.seed(123)
sample_size <- min(5000, nrow(data))  # Reduce sample size for efficiency
sample_indices <- sample(1:nrow(data), size = sample_size)
data_sampled <- data[sample_indices, ]

cat_vars <- names(data_sampled)[sapply(data_sampled, function(x) is.character(x) | is.factor(x))]
print(cat_vars)

### Feature Selection & Preprocessing
# Identify categorical and numerical features
cat_vars <- c("area_cluster", "model")  
num_vars <- setdiff(names(data_sampled), cat_vars)

# Convert categorical variables to factors
data_sampled[cat_vars] <- lapply(data_sampled[cat_vars], as.factor)

# Convert and Handle Missing Values
data_sampled[num_vars] <- data_sampled[num_vars] %>% mutate(across(everything(), as.numeric))
data_sampled[num_vars] <- data_sampled[num_vars] %>% mutate(across(everything(), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Compute Gower Distance (handles both categorical and numerical data)
diss_matrix <- daisy(data_sampled, metric = "gower")

### Finding Optimal Number of Clusters
#sil_width <- c()
#for (k in 2:6) {  # Check clusters from 2 to 6
#  pam_fit <- pam(diss_matrix, k = k)
#  sil_width <- c(sil_width, pam_fit$silinfo$avg.width)
#}
#optimal_clusters <- which.max(sil_width) + 1  # Find best K
optimal_clusters <-3

### Apply PAM Clustering
pam_result <- pam(diss_matrix, k = optimal_clusters)
data_sampled$Cluster <- as.factor(pam_result$clustering)

### Visualizing Clusters

# PCA-based Cluster Plot
fviz_cluster(list(data = as.matrix(diss_matrix), cluster = data_sampled$Cluster), geom = "point") +
  labs(title = "Cluster Visualization Using Gower Distance")

# Histogram of Policy Tenure across Clusters
ggplot(data_sampled, aes(x = policy_tenure, fill = Cluster)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  labs(title = "Policy Tenure Distribution by Cluster", x = "Policy Tenure (Years)", y = "Count") +
  theme_minimal()

# Claim Rate by Cluster
ggplot(data_sampled, aes(x = Cluster, fill = as.factor(is_claim))) +
  geom_bar(position = "fill") +
  labs(title = "Claim Rate by Cluster", x = "Cluster", y = "Proportion of Claims", fill = "Claim Status") +
  theme_minimal()

# Scatter Plot (Age vs. Policy Tenure, Colored by Cluster)
ggplot(data_sampled, aes(x = age_of_policyholder, y = policy_tenure, color = Cluster)) +
  geom_point(alpha = 0.6) +
  labs(title = "Clustered Policyholders by Age and Policy Tenure", x = "Age of Policyholder", y = "Policy Tenure (Years)") +
  theme_minimal()
