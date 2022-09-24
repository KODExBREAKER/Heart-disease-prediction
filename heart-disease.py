# Importing Dependicies
import numpy as np
import pandas as pd
import pickle
#Pickle is a package which converts data of python program into a serielized form 
# so that objects can be send in a seriel form to who ever going to use it

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Data Collection and Processing
# loading CSV-data to Pandas DataFrame

heart_data = pd.read_csv('heart.csv')

# number of rows and column in dataset
heart_data.shape

# checking for missing values
heart_data.isnull().sum()

# statistical measures about the data
heart_data.describe()

# checking the distribution of Tartget Variable
# 1 Represents person have Heart Disease
# 0 Represents No Heart Disease
heart_data['target'].value_counts()

# Splitting the Features and Target
X = heart_data.drop(columns='target', axis=1)    # X contains all features
Y = heart_data['target']                         # Y contains all targets
print(X)
print(Y)

# Splitting data into Training & Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 144)

# stratify = Y ->> to evenly distribute the 1 and 0 in Train and Test data else it might
#                  be possible that X_train have all 1 and X_test have all 0 or vice-versa  
print(X.shape, X_train.shape, X_test.shape)

# Model Training: Logistic Regression 
Reg_model = LogisticRegression()

# training the Reg_model with Training Data
Reg_model.fit(X_train, Y_train)


# Model Evaluation
# Accuracy Score

# accuracy on training data
X_train_prediction = Reg_model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy on training data: ", training_data_accuracy)

# accuracy on test data
X_test_prediction = Reg_model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy on test data: ", test_data_accuracy)

# We don't want our python program to run again-again whenever using our website, so we train the Reg_model once and
# import it somewhere else from where we can directly fetch it
# So we are dumping the Reg_model using Pickle into a file Model.pkl 
pickle.dump(Reg_model,open('Reg_model.pkl','wb'))
Reg_model=pickle.load(open('Reg_model.pkl','rb'))

