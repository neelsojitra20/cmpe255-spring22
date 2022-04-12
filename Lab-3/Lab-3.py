import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, confusion_matrix, plot_confusion_matrix, accuracy_score

#Load dataset
dataset = pd.read_csv("fetal_health.csv")
y = dataset['fetal_health'].to_numpy()

x = dataset.drop(columns=["fetal_health"])


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=109) 

#Create a svm Classifier
for x in [7,8,9] :
    clf = svm.SVC(C=x, kernel='linear') # Linear Kernel

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    cf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(clf, X_test, y_test)
    plt.show()
    print(f"for C hyperparameter {x}")
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    cf_matrix = confusion_matrix(y_test, y_pred)
    print("Accuracy    : ", accuracy)
    print("Recall      : ", recall)
    print("Precision   : ", precision)
    print("Confusion Matrix: ", cf_matrix)
    print()
