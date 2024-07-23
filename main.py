import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.io import arff
from sklearn.neighbors import KNeighborsClassifier
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
    

def K_nearest_neighbours(path:str,k:int, state:int) -> str:
    data, meta = arff.loadarff(path)
    dataframe = pd.DataFrame(data)

    # [initial row:ending row, initial column:ending column]
    features = dataframe.iloc[:,:-1]
    classes = dataframe.iloc[:,-1] # '-1' == series, '-1:' == datafame

    # convert to string if they're byte strings 
    classes = classes.apply(lambda x: x.decode('utf-8') if isinstance(x,bytes) else x)

    X_train, X_test, y_train, y_test = train_test_split(features,classes, test_size=0.5, random_state=state)
   
    ### Knn ###
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train,y_train)

    return calculate_accuracy(classifier.predict(X_test),y_test)
    


def main():
    np.random.seed(42)

    steel_plates_fault = [[] for _ in range(5)]
    ionosphere = [[] for _ in range(5)]
    banknotes = [[] for _ in range(5)]


    ## load datasets
    for k in range(5):
        for i in range(50):
            steel_plates_fault[k].append(K_nearest_neighbours("data/steel-plates-fault.arff",k+1,np.random.randint(0,1000)))
            ionosphere[k].append(K_nearest_neighbours("data/dataset_59_ionosphere.arff",k+1,np.random.randint(0,1000)))
            banknotes[k].append(K_nearest_neighbours("data/banknote-authentication.arff",k+1,np.random.randint(0,1000)))


    # Plotting!!!
    fig, axs = plt.subplots(1,3)

    titles = ['Steell', 'ion', 'banknotes']
    for ax,dataset,title in zip(axs,[steel_plates_fault,ionosphere,banknotes],titles):
        ax.boxplot(dataset)
        ax.set_title(title)
        ax.set_xticklabels(['K=1','K=2','K=3','K=4','K=5'])



    plt.show()




if __name__ == '__main__':
    main()