# ---- Load Libraries ----
library(ROSE)  # For random oversampling
library(ggplot2)
library(caret)

# ---- Load and Inspect Data ----
train_data_encoded <- read.csv("C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Advanced Analysis project/train_encoded.csv")

str(train_data_encoded)
num_observations <- nrow(train_data_encoded)
print(num_observations)

train_data_encoded$is_claim <- as.factor(train_data_encoded$is_claim)

# Print class distribution before oversampling
claim_counts <- table(train_data_encoded$is_claim)
print(claim_counts)

# ---- Apply Random Oversampling ----
set.seed(123)
oversampled_data <- ovun.sample(is_claim ~ ., data = train_data_encoded, method = "over", 
                                N = 2 * table(train_data_encoded$is_claim)[1])$data  # Corrected data reference

# Print new class distribution
print(table(oversampled_data$is_claim))  # Fixed access to class distribution

# Save the oversampled dataset
write.csv(oversampled_data, "C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Advanced Analysis project/train_oversampled.csv", row.names = FALSE)
print("Oversampled dataset saved as 'train_oversampled.csv'.")

num_observe <- nrow(oversampled_data)
print(num_observe)
str(oversampled_data)

# ---- Bar Chart for Random Oversampling Data ----
oversampled_data$is_claim <- as.factor(oversampled_data$is_claim)

# Calculate counts and percentages
count_data <- as.data.frame(table(oversampled_data$is_claim))
count_data$percentage <- count_data$Freq / sum(count_data$Freq) * 100

# Create the bar chart
ggplot(count_data, aes(x = Var1, y = Freq, fill = Var1)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), vjust = -0.5) +  
  labs(title = "Bar Chart of is_claim Variable (Random Oversampling)", x = "is_claim", y = "Count") +
  scale_fill_manual(values = c("lightblue", "lightgreen")) +
  theme_minimal()
