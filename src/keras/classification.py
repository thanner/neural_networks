#################################################################################
# Part 1 - Data Preprocessing
#################################################################################

# Importing the libraries
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Importing the dataset
dir = os.path.dirname(__file__)
dataset = pd.read_csv(os.path.join(dir, '../../resources/dataset/Churn_Modelling.csv'))
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

transformer = ColumnTransformer([("Geography", OneHotEncoder(), [1])], remainder="passthrough")
X = transformer.fit_transform(X)
X = X[:, 1:]  # avoid dummie variable trap

# Feature Scaling
standard_scaler = StandardScaler()
X_scaled = standard_scaler.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

#################################################################################
# Part 2 - Building Artificial Neural Network
#################################################################################

# Importing the libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Prepare Parameters
amount_input_units = np.size(X, 1)
amount_output_units = 1
amount_hidden_units = np.ceil(np.mean([amount_input_units, amount_output_units])).astype(np.int32)

# Create NN Architecture
classifier = Sequential()
classifier.add(Dense(input_dim=amount_input_units, units=amount_hidden_units, activation="relu"))
classifier.add(Dense(units=amount_hidden_units, activation="relu"))
classifier.add(Dense(units=amount_output_units, activation="sigmoid"))

# Learn Method
# Loss function is binary_crossentropy because the output is binary
# Great relation between activation function and loss function!
# Feedforward activation and Backpropagation derivate
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Show Classifier
classifier.summary()

#################################################################################
# Part 2 - Training
#################################################################################

# Train
batch_size = 10
epochs = 100
classifier.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

#################################################################################
# Part 4 - Test
#################################################################################

# Importing the libraries
from sklearn.metrics import confusion_matrix, accuracy_score

# Get predicion in boolean
threshold = 0.5
y_pred = classifier.predict(X_test)
y_pred = (y_pred > threshold)

# Generate confusion_matrix
matrix = confusion_matrix(y_test, y_pred)

# Get accuracy in set of test
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

#################################################################################
# Part 5 - Predict Case
#################################################################################

"""
Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000
"""

new_prediction = classifier.predict(standard_scaler.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > threshold)

