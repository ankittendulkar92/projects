# data preprocsssing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3: 13].values
y = dataset.iloc[:, 13].values

# encode categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# encode fisrt categorial variable of  country
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# encode second variable of gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# dummy variable 
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()  
X=X[:,1:]              
                
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# making the ann
#import keras
 import keras 

# import sewuential module
from keras.models import Sequential
from keras.layers import Dense
 
 # initializing the ann 
 classifier=Sequential()
 
 #passing the input layer and the first layer
 classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
 
 # add new layer
 classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))

# add the output layer
 classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

# compiling the ann
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# Fitting classifier to the Training set
classifier.fit(X_train,y_train,batch_size=10,epochs=100)

# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred=(y_pred>0.5)

#new prediction
new_prediction=classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction=(new_prediction>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#combine crossvalidation with keras
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier 
if __name__ == "__main__":
  classifier=KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=10) 
  accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=3,n_jobs=-1)
  mean = accuracies.mean()
  variance = accuracies.std()
  print("\nfinished")
  print("Mean: ",mean)
  print("Variance: ",variance)
 # cleanup for TF
import gc; gc.collect()