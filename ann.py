# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[: ,3:13].values
y = dataset.iloc[:, 13].values


#ENCODENG CATEgORICAL DATA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer 
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_1.fit_transform(X[:,2])
ct = ColumnTransformer([("Geography", OneHotEncoder(), [1])],    remainder = 'passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]


# SPLITTING THE DATASET INTO TEST SET AND TRAINING SET
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#   Making the ANN


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# INITiALIZING THE ANN
classifier = Sequential()


# Adding input layer and first hidden layer
classifier.add(Dense(units= 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11 ))
classifier.add(Dropout(p=0.1))
 
#Adding second hidden layer
classifier.add(Dense(units= 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p=0.1))

# OUTPUT LAYER
classifier.add(Dense(units= 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# Compiling the ANN
classifier.compile(optimizer = 'adam', loss= 'binary_crossentropy' , metrics=['accuracy'])

#Fitting the ANN to training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# MAKING PREDICTIONS AND EVALUATING THE MODEL
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


"""Predictinmg a single new observation
Predict if the customer with the following informations will leave the bank:
    geography:France
    Credit Score: 600
    gender:Male
    Age: 40
    Tenure: 3
    Balance:60000
    Number of products: 2
    Has Credit Card: yes
    Is active member: yes
    Estimated salary: 50000
"""
new_prediction = classifier.predict(sc_X.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)


# MAKING THE CONFUSION MATRIX


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Evaluating the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units= 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11 ))
    classifier.add(Dense(units= 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units= 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss= 'binary_crossentropy' , metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X=X_train, y=y_train, cv=10, n_jobs = -1)

mean = accuracies.mean()
variance = accuracies.std()


#IMPROVING THE ANN

# Dropout Regularisation to reduce overfitting if needed

