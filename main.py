import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.io import arff

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


import numpy as np
import pandas as pd

def calculate_accuracy(predictions: pd.DataFrame, test_classes:pd.Series) -> float:
    # Find num of correct between prediction and y_test
    count = 0
    test_classes.reset_index(drop=True,inplace=True)
    for i in range(0,predictions.size):
        if predictions[i] == test_classes[i]:
            count += 1

    return count/predictions.size

def load(path:str):
    data, meta = arff.loadarff(path)
    dataframe = pd.DataFrame(data)

    # [initial row:ending row, initial column:ending column]
    features = dataframe.iloc[:,:-1]
    classes = dataframe.iloc[:,-1] # '-1' == series, '-1:' == datafame

    # convert to string if they're byte strings 
    classes = classes.apply(lambda x: x.decode('utf-8') if isinstance(x,bytes) else x)

    ## SCALEEEEE
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features, classes


def run_classifier(classifier, path, param, state):
    features, classes = load(path)
    X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.5,random_state=state)
    clf = classifier(**param) ## wat does ** do????
    clf.fit(X_train, y_train)
    return calculate_accuracy(clf.predict(X_test), y_test)


def run_experiment(classifier, params, datasets, repeat_amount):
    results = {dataset[0]: [[] for _ in params] for dataset in datasets}

    for i, param in enumerate(params):
        for dataset in datasets:
            for _ in range(repeat_amount):
                accuracy = run_classifier(classifier,dataset[1], param, np.random.randint(0,1000))
                results[dataset[0]][i].append(accuracy)
    return results

def plot_results(results, titles, x_ticks, window_title,param_name):
    fig, axes = plt.subplots(1,3,figsize=(15,5))

    for ax, (title, dataset_results) in zip(axes, results.items()):
        ax.boxplot(dataset_results)
        ax.set_title(title)
        
        # Explicitly set tick locations to match the number of tick labels
        tick_positions = range(1, len(x_ticks) + 1)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{param_name}={p}" for p in x_ticks]) 

    fig.suptitle(window_title, fontsize=16)
    plt.tight_layout()


def main():
    np.random.seed(42) # for reproducability

    classifiers = {
        "KNN": {
            "classifier": KNeighborsClassifier,
            "params": [{"n_neighbors": k} for k in [1,2,3,4,5]],
            "param_name": "n_neighbors"
        },
        "Gaussian NB": {
            "classifier": GaussianNB,
            "params": [{"var_smoothing": s} for s in [1e-9,1e-5,1e-1]],
            "param_name": "var_smoothing"
        },
        "Decision Tree": {
            "classifier": DecisionTreeClassifier,
            "params": [{"max_depth": d} for d in [1,3,5,8,10]],
            "param_name": "max_depth"
        },
        "Logistic Regression": {
            "classifier": LogisticRegression,
            "params": [{"C": c, "max_iter": 5000} for c in [0.1,0.5,1.0,2.0,5.0]],
            "param_name": "C"
        },
        "Gradient Boosting": {
            "classifier": GradientBoostingClassifier,
            "params": [{"max_depth": d} for d in [1,3,5,8,10]],
            "param_name": "max_depth"
        },
        "Random Forest": {
            "classifier": RandomForestClassifier,
            "params": [{"max_depth": d} for d in [1,3,5,8,10]],
            "param_name": "max_depth"
        },
        "MLP": {
            "classifier": MLPClassifier,
            "params": [{"alpha": a, "max_iter": 1000} for a in [1e-5,1e-3,0.1,10.0]],
            "param_name": "alpha"
        }
    }

    datasets = [
        ("Steel Plates Fault","data/steel-plates-fault.arff"),
        ("Ionosphere", "data/dataset_59_ionosphere.arff"),
        ("Banknotes Authentication","data/banknote-authentication.arff")
    ]

    repeat_amount = 5

    for name, clf_info in classifiers.items():
        results = run_experiment(clf_info["classifier"],clf_info["params"],datasets,repeat_amount)
        plot_results(results, [dataset[0] for dataset in datasets], [list(p.values())[0] for p in clf_info["params"]], name, clf_info["param_name"])
        print(f"Finished {clf_info['classifier']}")


    plt.show()


if __name__ == '__main__':
    main()