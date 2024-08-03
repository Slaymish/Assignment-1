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

from tqdm import tqdm

import numpy as np
import pandas as pd

def calculate_accuracy(predictions: pd.DataFrame, test_classes: pd.Series) -> float:
    # Find number of correct predictions
    count = 0
    test_classes.reset_index(drop=True, inplace=True)
    for i in range(predictions.size):
        if predictions[i] == test_classes[i]:
            count += 1
    return count / predictions.size

def load(path: str):
    data, meta = arff.loadarff(path)
    dataframe = pd.DataFrame(data)

    features = dataframe.iloc[:, :-1]
    classes = dataframe.iloc[:, -1]

    # Convert byte strings to strings
    classes = classes.apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    # Scale features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features, classes

def run_classifier(classifier, path, param, state):
    features, classes = load(path)
    X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.5, random_state=state)
    clf = classifier(**param)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = calculate_accuracy(predictions, y_test)
    return predictions, accuracy, param

def run_experiment(classifier, params, datasets, repeat_amount):
    results = {dataset[0]: {tuple(param.items()): [] for param in params} for dataset in datasets}

    for i, param in enumerate(params):
        for dataset in datasets:
            for s in range(repeat_amount):
                _, accuracy, _ = run_classifier(classifier, dataset[1], param, s)
                results[dataset[0]][tuple(param.items())].append(accuracy)
    return results

def print_results(results):
    for dataset, params_results in results.items():
        print(f"\nResults for {dataset}:")
        for param, accuracies in params_results.items():
            mean_accuracy = np.mean(accuracies)
            param_str = ", ".join([f"{k}={v}" for k, v in param])
            print(f"  Param: {param_str}, Mean Accuracy: {mean_accuracy:.4f}")
            for accuracy in accuracies:
                print(f"    Accuracy: {accuracy:.4f}")

def get_lowest_mean_values(results):
    lowest_means = {}
    for classifier_name, datasets in results.items():
        for dataset_name, params_results in datasets.items():
            mean_accuracies = [np.mean(acc) for acc in params_results.values()]
            max_mean = max(mean_accuracies)
            max_index = mean_accuracies.index(max_mean)
            param_value = list(params_results.keys())[max_index]
            lowest_means[(classifier_name, dataset_name)] = (max_mean, param_value)
    return lowest_means

def main():
    classifiers = {
        "KNN": {
            "classifier": KNeighborsClassifier,
            "params": [{"n_neighbors": k} for k in [1, 2, 3, 4, 5]],
            "param_name": "n_neighbors"
        },
        "Gaussian NB": {
            "classifier": GaussianNB,
            "params": [{"var_smoothing": s} for s in [1e-9, 1e-5, 1e-1]],
            "param_name": "var_smoothing"
        },
        "Decision Tree": {
            "classifier": DecisionTreeClassifier,
            "params": [{"max_depth": d} for d in [1, 3, 5, 8, 10]],
            "param_name": "max_depth"
        },
        "Logistic Regression": {
            "classifier": LogisticRegression,
            "params": [{"C": c} for c in [0.1, 0.5, 1.0, 2.0, 5.0]],
            "param_name": "C"
        },
        "Gradient Boosting": {
            "classifier": GradientBoostingClassifier,
            "params": [{"max_depth": d} for d in [1, 3, 5, 8, 10]],
            "param_name": "max_depth"
        },
        "Random Forest": {
            "classifier": RandomForestClassifier,
            "params": [{"max_depth": d} for d in [1, 3, 5, 8, 10]],
            "param_name": "max_depth"
        },
        "MLP": {
            "classifier": MLPClassifier,
            "params": [{"alpha": a} for a in [1e-5, 1e-3, 0.1, 10.0]],
            "param_name": "alpha"
        }
    }

    datasets = [
        ("Steel Plates Fault", "data/steel-plates-fault.arff"),
        ("Ionosphere", "data/dataset_59_ionosphere.arff"),
        ("Banknotes Authentication", "data/banknote-authentication.arff")
    ]

    repeat_amount = 50

    all_results = {}

    classifiers_list = list(classifiers.items())
    for i, (name, clf_info) in enumerate(tqdm(classifiers_list, desc="Classifiers")):
        results = run_experiment(clf_info["classifier"], clf_info["params"], datasets, repeat_amount)
        all_results[name] = results
        print_results(results)
        tqdm.write(f"Finished {name}")

    lowest_means = get_lowest_mean_values(all_results)

    table = "| **Classifier**      | Steel Plates | Ionosphere | Banknote auth |\n"
    table += "| ------------------- | ------------ | ---------- | ------------- |\n"

    for classifier_name in classifiers.keys():
        row = f"| {classifier_name:<19} |"
        for dataset_name in ["Steel Plates Fault", "Ionosphere", "Banknotes Authentication"]:
            mean_value, param_value = lowest_means[(classifier_name, dataset_name)]
            row += f" {mean_value:.4f} ({', '.join([f'{k}={v}' for k, v in param_value])}) |"
        table += row + "\n"

    print(table)

if __name__ == '__main__':
    main()
