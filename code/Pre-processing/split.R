# Assume your dataset is named 'df'
df=read.csv("C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Advanced Analysis project/full_transformed.csv")
set.seed(42)  # For reproducibility

# Get row indices for training set (80% of data)
train_indices <- sample(nrow(df), size = 0.8 * nrow(df))

# Split data
train_data <- df[train_indices, ]
test_data <- df[-train_indices, ]

# Count 0s and 1s in Training Set
train_counts <- table(train_data$is_claim)
print("Training Set Distribution:")
print(train_counts)

# Count 0s and 1s in Test Set
test_counts <- table(test_data$is_claim)
print("Test Set Distribution:")
print(test_counts)


# Save as CSV files
write.csv(train_data, "C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Advanced Analysis project/train_set.csv", row.names = FALSE)
write.csv(test_data, "C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Advanced Analysis project/test_set.csv", row.names = FALSE)

dim(train_data)
dim(test_data)

str(train_data)
str(test_data)
