train_data_encoded <- read.csv("C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Advanced Analysis project/train_encoded.csv")

str(train_data_encoded)
# Get the number of observations in the dataset
num_observations <- nrow(train_data_encoded)

# Print the number of observations
print(num_observations)

train_data_encoded$is_claim <- as.factor(train_data_encoded$is_claim)

# Get the count of 0's and 1's in the is_claim variable
claim_counts <- table(train_data_encoded$is_claim)
print(claim_counts)


# Install and load the SMOTE family package
library(smotefamily)

set.seed(123)

# Apply SMOTE to balance the dataset
train_set_balanced <- SMOTE(train_data_encoded[,-which(names(train_data_encoded)=="is_claim")], 
                            train_data_encoded$is_claim, K = 5, dup_size = 13)


# Check new class distribution
print(table(train_set_balanced$data$class))

# Create a new dataset with the balanced data
balanced_data <- train_set_balanced$data
colnames(balanced_data)[colnames(balanced_data) == "class"] <- "is_claim"

# Save the balanced dataset as a new CSV file
write.csv(balanced_data, "C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Advanced Analysis project/train_balanced.csv", row.names = FALSE)

# Confirm the new dataset is saved
print("Balanced dataset saved as 'train_balanced.csv'.")

num_observe <- nrow(balanced_data)

# Print the number of observations
print(num_observe)
str(balanced_data)
colnames(balanced_data)


# Bar chart for SMOTE data
balanced_data$is_claim <- as.factor(balanced_data$is_claim)

# Calculate counts and percentages
count_data <- as.data.frame(table(balanced_data$is_claim))
count_data$percentage <- count_data$Freq / sum(count_data$Freq) * 100

# Create the bar chart with percentages on top of the bars
ggplot(count_data, aes(x = Var1, y = Freq, fill = Var1)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), vjust = -0.5) +  # Adds percentage on top of bars
  labs(title = "Bar Chart of is_claim Variable (SMOTE)", x = "is_claim", y = "Count") +
  scale_fill_manual(values = c("lightblue", "lightgreen")) +
  theme_minimal()


