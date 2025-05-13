train_data=read.csv("C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Advanced Analysis project/full_cleaned.csv")
str(train_data)
# Check skewness before transformation
hist(train_data$age_of_car, main = "Age of Car - Before Log Transform", breaks = 20)
hist(train_data$age_of_policyholder, main = "Age of Policyholder - Before Log Transform", breaks = 20)

boxplot(train_data$age_of_car, main = "Age of Car - Before Log Transform")
boxplot(train_data$age_of_policyholder, main = "Age of Policyholder - Before Log Transform")

# Log transformation
train_data$age_of_car_log <- log1p(train_data$age_of_car)
train_data$age_of_policyholder_log <- log1p(train_data$age_of_policyholder)

# Check if skewness is reduced
hist(train_data$age_of_car_log, main = "Age of Car - After Log Transform", breaks = 20)
hist(train_data$age_of_policyholder_log, main = "Age of Policyholder - After Log Transform", breaks = 20)

boxplot(train_data$age_of_car_log, main = "Age of Car - After Log Transform")
boxplot(train_data$age_of_policyholder_log, main = "Age of Policyholder - After Log Transform")



# Square Root Transformation
train_data$age_of_car_sqrt <- sqrt(train_data$age_of_car)
train_data$age_of_policyholder_sqrt <- sqrt(train_data$age_of_policyholder)
hist(train_data$age_of_car_sqrt, main = "Age of Car - After sqrt Transform", breaks = 20)
hist(train_data$age_of_policyholder_sqrt, main = "Age of Policyholder - After sqrt Transform", breaks = 20)

boxplot(train_data$age_of_car_sqrt, main = "Age of Car - After sqrt Transform")
boxplot(train_data$age_of_policyholder_sqrt, main = "Age of Policyholder - After sqrt Transform")


# Cube Root Transformation
train_data$age_of_car_cbrt <- train_data$age_of_car^(1/3)
train_data$age_of_policyholder_cbrt <- train_data$age_of_policyholder^(1/3)
hist(train_data$age_of_car_cbrt, main = "Age of Car - After cubic Transform", breaks = 20)
hist(train_data$age_of_policyholder_cbrt, main = "Age of Policyholder - After cubic Transform", breaks = 20)

boxplot(train_data$age_of_car_cbrt, main = "Age of Car - After cubic Transform")
boxplot(train_data$age_of_policyholder_cbrt, main = "Age of Policyholder - After cubic Transform")

library(MASS)  # Box-Cox transformation

# Box-Cox requires strictly positive values, so shift if needed
train_data$age_of_car_adj <- train_data$age_of_car + 1
train_data$age_of_policyholder_adj <- train_data$age_of_policyholder + 1

# Find the best lambda for transformation
boxcox_model_car <- boxcox(lm(age_of_car_adj ~ 1, data = train_data), lambda = seq(-2, 2, by = 0.1))
best_lambda_car <- boxcox_model_car$x[which.max(boxcox_model_car$y)]

boxcox_model_policy <- boxcox(lm(age_of_policyholder_adj ~ 1, data = train_data), lambda = seq(-2, 2, by = 0.1))
best_lambda_policy <- boxcox_model_policy$x[which.max(boxcox_model_policy$y)]

# Function to apply Box-Cox transformation
boxcox_transform <- function(x, lambda) {
  if (lambda == 0) {
    return(log(x))  # Log transformation if lambda = 0
  } else {
    return((x^lambda - 1) / lambda)
  }
}

# Apply Box-Cox transformation
train_data$age_of_car_boxcox <- boxcox_transform(train_data$age_of_car_adj, best_lambda_car)
train_data$age_of_policyholder_boxcox <- boxcox_transform(train_data$age_of_policyholder_adj, best_lambda_policy)

hist(train_data$age_of_car_boxcox, main = "Age of Car - After boxcox Transform", breaks = 20)
hist(train_data$age_of_policyholder_boxcox, main = "Age of Policyholder - After boxcox Transform", breaks = 20)

boxplot(train_data$age_of_car_boxcox, main = "Age of Car - After boxcox Transform")
boxplot(train_data$age_of_policyholder_boxcox, main = "Age of Policyholder - After boxcox Transform")

###adding box-cox to the age of policyholder removed outliers.
###adding cubic root transformations to the age of car remain only 3-5 outliers

# Replace original values with transformed values
train_data$age_of_car <- train_data$age_of_car_cbrt
train_data$age_of_policyholder <- train_data$age_of_policyholder_boxcox

# Select only required variables
train_data <- train_data[, c("policy_tenure", "age_of_car", "age_of_policyholder", "area_cluster", "model", "is_claim")]

# Verify the structure of the cleaned dataset
str(train_data)

# Save the cleaned dataset
write.csv(train_data, "C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Advanced Analysis project/full_transformed.csv", row.names = FALSE)
str(train_data)


