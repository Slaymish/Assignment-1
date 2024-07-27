import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.io import arff
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

def calculate_accuracy(prediction: pd.DataFrame, test_classes:pd.Series) -> float:
    # Find num of correct between prediction and y_test
    count = 0
    test_classes.reset_index(drop=True,inplace=True)
    for i in range(0,prediction.size):
        if prediction[i] == test_classes[i]:
            count += 1

    return count/prediction.size

def load(path:str):
    data, meta = arff.loadarff(path)
    dataframe = pd.DataFrame(data)

    # [initial row:ending row, initial column:ending column]
    features = dataframe.iloc[:,:-1]
    classes = dataframe.iloc[:,-1] # '-1' == series, '-1:' == datafame

    # convert to string if they're byte strings 
    classes = classes.apply(lambda x: x.decode('utf-8') if isinstance(x,bytes) else x)

    return features, classes


"""
Returns the accuracy running Knn
"""
def K_nearest_neighbours(path:str,k:int, state:int) -> float:
    features, classes = load(path)
    
    X_train, X_test, y_train, y_test = train_test_split(features,classes, test_size=0.5, random_state=state)
   
    ### Knn ###
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train,y_train)

    return calculate_accuracy(classifier.predict(X_test),y_test)


def gaussian_NB(path:str, smoothing,state) -> float:
    features, classes = load(path)

    X_train, X_test, y_train, y_test = train_test_split(features,classes, test_size=0.5, random_state=state)

    ### Gaussian NB ###
    classifer = GaussianNB(var_smoothing=smoothing)
    classifer.fit(X_train,y_train)

    return calculate_accuracy(classifer.predict(X_test),y_test)

def decision_tree_classifier(path:str,state:int) -> float:
    features,classes = load(path)

def logistic_regression():
    pass

def gradient_boosting_classifier():
    pass

def random_forest_classifier():
    pass

def MLP_classifier():
    pass




def main():
    np.random.seed(42)

    steel_path = "data/steel-plates-fault.arff"
    ionosphere_path = "data/dataset_59_ionosphere.arff"
    banknotes_path = "data/banknote-authentication.arff"


    ## Scale data

    repeat_amount = 50

    ### KNN ###

    steel_knn = [[] for _ in range(5)]
    ionosphere_knn = [[] for _ in range(5)]
    banknotes_knn = [[] for _ in range(5)]

    values = [1,2,3,4,5]

    for j in range(len(values)):
        k = j+1
        for i in range(repeat_amount):
            steel_knn[j].append(K_nearest_neighbours(steel_path,k,np.random.randint(0,1000)))
            ionosphere_knn[j].append(K_nearest_neighbours(ionosphere_path,k,np.random.randint(0,1000)))
            banknotes_knn[j].append(K_nearest_neighbours(banknotes_path,k,np.random.randint(0,1000)))

    # Plotting Knn
    fig, ax1 = plt.subplots(1,3)

    titles = ['Steell', 'ion', 'banknotes']
    for ax,dataset,title in zip(ax1,[steel_knn,ionosphere_knn,banknotes_knn],titles):
        ax.boxplot(dataset)
        ax.set_title(title)
        ax.set_xticklabels(['K=1','K=2','K=3','K=4','K=5'])

    print("Knn Done")

    ### GaussianNB ###

    steel_gnb = [[] for _ in range(4)]
    ionosphere_gnb = [[] for _ in range(4)]
    banknotes_gnb = [[] for _ in range(4)]

    values = [1e-9,1e-8,1e-7,1e-6]  # todo: change to actuall values

    for j in range(len(values)):
        smoothing = values[j]
        for i in range(repeat_amount):
            steel_gnb[j].append(gaussian_NB(steel_path,smoothing,np.random.randint(0,1000)))
            ionosphere_gnb[j].append(gaussian_NB(ionosphere_path,smoothing,np.random.randint(0,1000)))
            banknotes_gnb[j].append(gaussian_NB(banknotes_path,smoothing,np.random.randint(0,1000)))


    def(steel,ion,banknotes,titles,ticklabels) -> ndarray

    # Plotting Gaussian
    fig, ax2 = plt.subplots(1,3)

    titles = ['Steell', 'ion', 'banknotes']
    for ax,dataset,title in zip(ax2,[steel_gnb,ionosphere_gnb,banknotes_gnb],titles):
        ax.boxplot(dataset)
        ax.set_title(title)
        ax.set_xticklabels(values)

    print("Guassian NB Done")
    



    plt.show()


if __name__ == '__main__':
    main()