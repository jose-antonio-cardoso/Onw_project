# ---
# title: "Own_project"
# author: "José Antonio Cardoso"
# date: "2024-11-23"
#
# Note
# This algorithm was developed using R version 4.4.1, and aims to classify security incidents 
# using the Randon Forest algorithm.
# ---

##############
# INSTALLATION
##############

# Load the required libraries
# (Important: This process may take a while and may take a few minutes
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(purrr)) install.packages("purrr", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(Metrics)) install.packages("Metrics", repos = "http://cran.us.r-project.org")

library(caret)
library(tidyverse)
library(dplyr)
library(purrr)
library(randomForest)
library(Metrics)

# Clears previous executions and clears the console
rm(list = ls())
cat("\f")

#################################################
# START PREPARING THE TRAINING AND TEST DATA SETS
#################################################

# Defines the dataset that will be validated (Training)
guide_file <- 'GUIDE_Train.csv'

# Check if the dataset exists.
if (!file.exists(guide_file)) {
  # If it exists, download the repository from GitHub, validating success or error when downloading
  url <- 'https://raw.githubusercontent.com/jose-antonio-cardoso/Own_project/main/GUIDE_Train.csv'
  tryCatch({
    download.file(url, guide_file, mode = "wb")
    cat("Dataset downloaded successfully!\n")
  }, error = function(e) {
    cat("Error downloading dataset:", conditionMessage(e), "\n")
  })
  # If the dataset already exists, just issue a warning.  
} else {
  cat("Dataset already exists in directory.\n")
}

# Define o conjunto de dados que será validado (Test)
guide_file <- 'GUIDE_Test.csv'

# Check if the dataset exists.
if (!file.exists(guide_file)) {
  # If it exists, download the repository from GitHub, validating success or error when downloading
  url <- 'https://raw.githubusercontent.com/jose-antonio-cardoso/Own_project/main/GUIDE_Test.csv'
  tryCatch({
    download.file(url, guide_file, mode = "wb")
    cat("Dataset downloaded successfully!\n")
  }, error = function(e) {
    cat("Error downloading dataset:", conditionMessage(e), "\n")
  })
  # If the dataset already exists, just issue a warning.  
} else {
  cat("Dataset already exists in directory.\n")
}

# Read the training data set
GUIDE_train <- read.csv("GUIDE_Train.csv", header = TRUE)
# Read the test data set
GUIDE_test <- read.csv("GUIDE_Test.csv", header = TRUE)

# Convert the Id field to character from the training dataset
GUIDE_train$Id <-as.character(GUIDE_train$Id)
# Convert the Id field to character from the test data set
GUIDE_test$Id <-as.character(GUIDE_test$Id)

# As you can see in this code, the data set is already divided.
# However, to demonstrate the knowledge about the logic and necessity of this division
# I present here the code related to this process, where for educational purposes I consider the
# complete GUDE file, which I fictitiously called complet_GUIDE.csv
# Just for information...
# Note: The division between Training and Testing is 70% and 30% respectively
# Dividing the data into training and validation
# Defining the seed for generating random numbers
#set.seed(0)
#train_index <- createDataPartition(complet_GUIDE$IncidentGrade, p = 0.7, list = FALSE)
#GUIDE_train <- complet_GUIDE[train_index,]
#GUIDE_test <- complet_GUIDE[-train_index,]

####################################
# PREPARING DATASETS (TRAINING ONLY)
####################################

# Make a copy of the training dataset
df_data_train <- GUIDE_train

# Select only the columns that are important for the evaluation
cols_df <- c("Id","IncidentId","ActionGrouped","ActionGranular","LastVerdict","IncidentGrade")

# Apply the selection on the Training dataset
df_data_train <- df_data_train[, cols_df]

# Filters only the rows that contain SOC responses, thus eliminating any column that contains "Not Available" values
df_data_train <- df_data_train %>%
   filter(ActionGrouped != "" & ActionGranular != "" & LastVerdict != "" & IncidentGrade != "") 

# Presents some quantitative information about the dataset.
cat("Number of records in the Training dataset","\n")

cat("Show the first few rows of the dataset","\n")
head(df_data_train,5)

cat("\n","Show unique values from columns")

# Apply the function to each column and store the results in a list
newcols_df <- c("ActionGrouped","ActionGranular","LastVerdict","IncidentGrade")
list_result <- map(newcols_df, ~df_data_train %>% 
                  pull(.x) %>% 
                  unique())
names(list_result) <- newcols_df
list_result
# In this line above, the code prints the list containing the unique values of
# each column, as explained above,

cat("\n","Shows the structure of the dataset","\n")
str(df_data_train)

cat("\n","Summarizes the dataset")
summary(df_data_train)

# Transform the string column into a factor
df_data_train$IncidentGrade <- as.factor(df_data_train$IncidentGrade)
df_data_train$ActionGrouped <- as.factor(df_data_train$ActionGrouped)
df_data_train$ActionGranular <- as.factor(df_data_train$ActionGranular)
df_data_train$LastVerdict <- as.factor(df_data_train$LastVerdict)

##############################################
# PERFORM TRAINING AND EVALUATION OF THE MODEL
##############################################

# Set the seed for random number generation
set.seed(0)

# Set the control parameters
ctrl <- trainControl(method = "cv", number = 10)

# Define the hyperparameter grid
hyper_grid <- expand.grid(mtry = c(3, 4, 5))

# Train the model
trained_model <- train(IncidentGrade ~ ActionGrouped + ActionGranular + LastVerdict, 
                       data = df_data_train, 
                       method = "rf",
                       tuneGrid = hyper_grid,
                       trControl = ctrl)

# Print the trained model
print(trained_model)

# Presents an overview of the different hyperparameter combinations tested and their respective performance metrics.
trained_model$results

# Show cross-validation results.
trained_model$resample

# Shows the final selected model after hyperparameter sweeping and cross-validation.
trained_model$finalModel

# Shows the relative importance of each predictor variable in the final model
trained_model$finalModel$importance

# This graph shows the distribution of classes in the data set, helping to understand if there is an imbalance.
# Plots the distribution of classes
ggplot(df_data_train, aes(x = IncidentGrade)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Distribution of Classes in the Training Set", x = "Class", y = "Count")

# This graph shows the relative importance of each predictor variable.
# Plots the importance of variables
importance <- varImp(trained_model, scale = FALSE)
ggplot(importance, aes(x = reorder(Overall, Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Importance of Variables", x = "Variable", y = "Importance")

# Make predictions on the TRAINING dataset to evaluate the model's performance
train_predictions <- predict(trained_model, newdata = df_data_train)

# Converts categorical data into factors for classification
actual_class <- df_data_train$IncidentGrade
predicted_class <- train_predictions

# Calculates accuracy, as an assessment of model performance
accuracy_train <- accuracy(actual_class, predicted_class)
cat("Accuracy:", accuracy_train, "\n")

# Calculate the F1-score, as an evaluation of the model's performance
f1_train <- f1(actual_class, predicted_class)
cat("F1-score:", f1_train, "\n")

#####################################
# TEST AND EVALUATE THE TRAINED MODEL
#####################################

# Make a copy of the test dataset
df_data_test <- GUIDE_test

# Apply the selection to the Test dataset
df_data_test <- df_data_test[, cols_df]

# Filters only the rows that contain SOC responses, thus eliminating any column that contains "Not Available" values
df_data_test <- df_data_test %>%
   filter(ActionGrouped != "" & ActionGranular != "" & LastVerdict != "" & IncidentGrade != "") 

# Transform the string column into a factor
df_data_test$IncidentGrade <- as.factor(df_data_test$IncidentGrade)
df_data_test$ActionGrouped <- as.factor(df_data_test$ActionGrouped)
df_data_test$ActionGranular <- as.factor(df_data_test$ActionGranular)
df_data_test$LastVerdict <- as.factor(df_data_test$LastVerdict)

# Make predictions on the TEST dataset to evaluate the model's performance
test_predictions <- predict(trained_model, newdata = df_data_test)

# Converts categorical data into factors for classification
actual_class <- df_data_test$IncidentGrade
predicted_class <- test_predictions

# Calculates accuracy, as an assessment of model performance
accuracy_test <- accuracy(actual_class, predicted_class)
cat("Accuracy:", accuracy_test, "\n")

# Calculate the F1-score, as an evaluation of the model's performance
f1_test <- f1(actual_class, predicted_class)
cat("F1-score:", f1_test, "\n")

####################################
# MODEL OUTPUT AND ITS PERFORMANCE #
####################################

# IMPRIME A Calculates accuracy, as an assessment of model performance
cat("Accuracy:", accuracy_train, "\n")

# Calculate the F1-score, as an evaluation of the model's performance
cat("F1-score:", f1_train, "\n")

# Calculates accuracy, as an assessment of model performance
cat("Accuracy:", accuracy_test, "\n")

# Calculate the F1-score, as an evaluation of the model's performance
cat("F1-score:", f1_test, "\n")
