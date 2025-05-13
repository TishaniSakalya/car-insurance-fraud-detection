# Load necessary library
library(caret)

# Load your dataset
train_data_new <- read.csv("C:/Users/Hiruni/OneDrive/Desktop/test_set.csv")

dim(train_data_new)

# Define categorical variables
categorical_vars <- c("area_cluster", "model")  # Update with actual categorical variables

# Create a one-hot encoder using dummyVars()
one_hot_encoder <- dummyVars(~ ., data = train_data_new, 
                             levelsOnly = FALSE)  # Keeps all original variables

# Apply transformation
train_data_encoded <- as.data.frame(predict(one_hot_encoder, train_data_new))
write.csv(train_data_encoded, "C:/Users/Hiruni/OneDrive/Desktop/test_encoded.csv", row.names = FALSE)
str(train_data_encoded)

dim(train_data_encoded)
str(train_data_encoded)
