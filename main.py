import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.io import arff
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

def method(path:str):
    data, meta = arff.loadarff(path)

    print(meta)

    dataframe = pd.DataFrame(data)

    # [initial row:ending row, initial column:ending column]
    features = dataframe.iloc[:,:-1]
    classes = dataframe.iloc[:,-1] # '-1' == series, '-1:' == datafame

    # convert to string if they're byte strings 
    classes = classes.apply(lambda x: x.decode('utf-8') if isinstance(x,bytes) else x)


    X_train, X_test, y_train, y_test = train_test_split(features,classes, test_size=0.33, random_state=42)
   

    ### Knn ###
    classifier = KNeighborsClassifier(n_neighbors=4)
    classifier.fit(X_train,y_train)

    prediction = classifier.predict(X_test)

    # Find num of correct between prediction and y_test
    count = 0
    y_test.reset_index(drop=True,inplace=True)
    for i in range(0,prediction.size):
        if prediction[i] == y_test[i]:
            count += 1

    print(count)

    accuracy = count/prediction.size
    print('Accuracy = ' + str(accuracy))


## load datasets
steel_plates_fault = method("data/steel-plates-fault.arff")
ionosphere = method("data/dataset_59_ionosphere.arff")
banknotes = method("data/banknote-authentication.arff")

