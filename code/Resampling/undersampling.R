# ---- Load Libraries ----
library(ROSE)  # For random undersampling
library(ggplot2)
library(caret)

# ---- Load and Inspect Data ----
train_data_encoded <- read.csv("C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Advanced Analysis project/train_encoded.csv")

str(train_data_encoded)
num_observations <- nrow(train_data_encoded)
print(num_observations)

train_data_encoded$is_claim <- as.factor(train_data_encoded$is_claim)

# Print class distribution before undersampling
claim_counts <- table(train_data_encoded$is_claim)
print(claim_counts)

# ---- Apply Random Undersampling ----
set.seed(123)
minority_class_size <- min(table(train_data_encoded$is_claim))  # Get minority class count

undersampled_data <- ovun.sample(is_claim ~ ., data = train_data_encoded, method = "under", 
                                 N = 2 * minority_class_size)$data  # Match both classes to the minority count

# Print new class distribution
print(table(undersampled_data$is_claim))  # Fixed access to class distribution

# Save the undersampled dataset
write.csv(undersampled_data, "C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Advanced Analysis project/train_undersampled.csv", row.names = FALSE)
print("Undersampled dataset saved as 'train_undersampled.csv'.")

num_observe <- nrow(undersampled_data)
print(num_observe)
str(undersampled_data)

# ---- Bar Chart for Random Undersampling Data ----
undersampled_data$is_claim <- as.factor(undersampled_data$is_claim)

# Calculate counts and percentages
count_data <- as.data.frame(table(undersampled_data$is_claim))
count_data$percentage <- count_data$Freq / sum(count_data$Freq) * 100

# Create the bar chart
ggplot(count_data, aes(x = Var1, y = Freq, fill = Var1)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), vjust = -0.5) +  
  labs(title = "Bar Chart of is_claim Variable (Random Undersampling)", x = "is_claim", y = "Count") +
  scale_fill_manual(values = c("lightblue", "lightgreen")) +
  theme_minimal()
