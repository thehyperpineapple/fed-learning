from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def ml_subsets(subsets, num_subsets, X_train, Y_train):
    subset_size = len(X_train) // num_subsets
    indices = list(range(len(X_train)))
    
    # Shuffle the indices
    np.random.shuffle(indices)
    
    # Create subsets
    for i in range(num_subsets):
        start_idx = i * subset_size
        end_idx = (i + 1) * subset_size if i != num_subsets - 1 else len(X_train)
        subset_indices = indices[start_idx:end_idx]
        
        # Select the subsets for X_train and Y_train based on the shuffled indices
        X_subset = X_train.iloc[subset_indices]
        Y_subset = Y_train.iloc[subset_indices]
        
        # Append the subset as a nested list [X_subset, Y_subset]
        subsets.append([X_subset, Y_subset])
    
    return subsets

def privacy_checker(classifier, x_train, y_train, x_test, y_test, check_privacy):
    if check_privacy==False:
        y_predict = classifier.predict(x_test)

        return [accuracy_score(y_test, y_predict),
            precision_score(y_test, y_predict),
            recall_score(y_test, y_predict),
            f1_score(y_test, y_predict),
            classification_report(y_test, y_predict),
            confusion_matrix(y_test, y_predict)
            ]
    else:
        y_predict = classifier.predict(x_train)

        return [accuracy_score(y_train, y_predict),
                precision_score(y_train, y_predict),
                recall_score(y_train, y_predict),
                f1_score(y_train, y_predict),
                classification_report(y_train, y_predict),
                confusion_matrix(y_train, y_predict)
            ]

def logistic_regression_pipeline(x_train, y_train, x_test, y_test, check_privacy, federated, nodes):
    if federated==False:
        print("Status: Creating conventional logistic regression model")
        classifier = LogisticRegression(random_state=42).fit(x_train.values, y_train.values)
        return privacy_checker(classifier, x_train, y_train, x_test, y_test, check_privacy)
    else:
        print("Status: Creating federated logistic regression model")
        nodes = nodes
        subsets = []
        coeffs = []
        
        for node in range(nodes):
            subsets = ml_subsets(subsets, num_subsets=nodes, X_train=x_train, Y_train=y_train)
            lr = LogisticRegression(random_state=42).fit(subsets[node][0].values, subsets[node][1].values)
            coeffs.append([lr.coef_, lr.intercept_, lr.classes_])
        coeffs_array = np.array([c[0] for c in coeffs])
        intercepts_array = np.array([c[1] for c in coeffs])
        avg_coeffs = np.mean(coeffs_array, axis=0)
        avg_intercepts = np.mean(intercepts_array, axis=0)
        classes_ = coeffs[0][2]
        # Create a new Logistic Regression object
        final_model = LogisticRegression()

        # Set the averaged coefficients and intercepts to the new model
        final_model.coef_ = avg_coeffs
        final_model.intercept_ = avg_intercepts
        final_model.classes_ = classes_ 
        print("Nodes: ", nodes)
        return privacy_checker(final_model, x_train, y_train, x_test, y_test, check_privacy)



