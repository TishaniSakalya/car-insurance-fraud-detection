# ---- Load Libraries ----
library(ROSE)  # For synthetic resampling
library(ggplot2)
library(caret)

# ---- Load and Inspect Data ----
train_data_encoded <- read.csv("C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Advanced Analysis project/train_encoded.csv")

str(train_data_encoded)
num_observations <- nrow(train_data_encoded)
print(num_observations)

train_data_encoded$is_claim <- as.factor(train_data_encoded$is_claim)

# Print class distribution before resampling
claim_counts <- table(train_data_encoded$is_claim)
print(claim_counts)

# ---- Apply ROSE Resampling ----
set.seed(123)
rose_data <- ROSE(is_claim ~ ., data = train_data_encoded, seed = 123)$data  # ROSE generates synthetic samples

# Print new class distribution
print(table(rose_data$is_claim))

# Save the ROSE-resampled dataset
write.csv(rose_data, "C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Advanced Analysis project/train_rose.csv", row.names = FALSE)
print("ROSE-resampled dataset saved as 'train_rose.csv'.")

num_observe <- nrow(rose_data)
print(num_observe)
str(rose_data)

# ---- Bar Chart for ROSE Data ----
rose_data$is_claim <- as.factor(rose_data$is_claim)

# Calculate counts and percentages
count_data <- as.data.frame(table(rose_data$is_claim))
count_data$percentage <- count_data$Freq / sum(count_data$Freq) * 100

# Create the bar chart
ggplot(count_data, aes(x = Var1, y = Freq, fill = Var1)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), vjust = -0.5) +  
  labs(title = "Bar Chart of is_claim Variable (ROSE Method)", x = "is_claim", y = "Count") +
  scale_fill_manual(values = c("lightblue", "lightgreen")) +
  theme_minimal()
