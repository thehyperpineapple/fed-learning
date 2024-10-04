# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from scipy.stats import mode
import warnings
warnings.filterwarnings("ignore")


# %%
df = pd.read_csv("../../../datasets/predictive_maintenance/balanced_binary_classification.csv", index_col=[0])
X = df.drop(columns= ['Target'])
Y = df['Target']
print("Status: Splitting data")
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# %%
# Define the classifiers
# non-PCA
# classifiers = {
#     "Logistic Regression": LogisticRegression(C=82.39, max_iter=439),
#     "Ridge Classifier": RidgeClassifier(alpha=82.39, max_iter=439),
#     "Decision Tree": DecisionTreeClassifier(max_depth=12),
#     "Random Forest": RandomForestClassifier(n_estimators=188, max_depth=12),
#     "AdaBoost": AdaBoostClassifier(n_estimators=188),
#     "Gradient Boosting": GradientBoostingClassifier(n_estimators=188, max_depth=12)
# }
# PCA
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Ridge Classifier": RidgeClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# %%
# Function to aggregate logistic regression and ridge classifier models (linear models)
def federated_averaging_linear(models):
    avg_model = models[0]  # Initialize with the first model
    avg_model.coef_ = np.mean([model.coef_ for model in models], axis=0)
    avg_model.intercept_ = np.mean([model.intercept_ for model in models], axis=0)
    return avg_model

# Function to simulate local histogram computation for tree-based models (FedTree inspired)
def compute_histograms(clf, x_train, y_train):
    # Simulate local histogram computation (you would compute gradient-based histograms in a real scenario)
    clf.fit(x_train, y_train)
    return clf  # Return the locally trained model (with histograms)

# Function to aggregate histograms (histogram sharing for tree-based models)
def aggregate_histograms(tree_models):
    # In FedTree, histograms would be aggregated here; we simulate this by averaging model predictions
    # In practice, this would be an aggregation of histogram data, which defines the tree split
    return tree_models[0]  # Return the first tree model as a simplified representation


# %%
# Federated Learning for Linear and Tree-Based models
def simulate_federated_learning(x_train, y_train, x_test, y_test, classifiers, num_nodes):
    # Split the training data across the given number of nodes
    node_data = np.array_split(x_train, num_nodes)
    node_labels = np.array_split(y_train, num_nodes)
    
    metrics = []

    for name, clf in classifiers.items():
        node_models = []
        node_predictions = []

        # Train each node and collect models and predictions
        for i in range(num_nodes):
            if name in ["Decision Tree", "Random Forest", "AdaBoost", "Gradient Boosting"]:
                local_clf = clf.__class__()  # Create a fresh model for each node
                local_clf = compute_histograms(local_clf, node_data[i], node_labels[i])
                node_models.append(local_clf)
            else:
                local_clf = clf.__class__()  # Create a fresh model for each node
                local_clf.fit(node_data[i], node_labels[i])
                node_models.append(local_clf)

        # Aggregate models (FedAvg) for linear classifiers
        if name in ["Logistic Regression", "Ridge Classifier"]:
            global_model = federated_averaging_linear(node_models)
            y_pred = global_model.predict(x_test)

        # Use histogram aggregation for tree-based classifiers
        elif name in ["Decision Tree", "Random Forest", "AdaBoost", "Gradient Boosting"]:
            global_model = aggregate_histograms(node_models)
            y_pred = global_model.predict(x_test)

        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Store the results
        metrics.append({
            'Classifier': name,
            'Nodes': num_nodes,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Classification Report': report,
            'Confusion Matrix': cm
        })

    return metrics


# %%
# Simulate federated learning for different node counts
node_counts = [1, 5, 10, 25, 50, 100, 200]

# Create a list to store all metrics for all node counts
all_metrics = []

# %%
for nodes in node_counts:
    metrics = simulate_federated_learning(x_train, y_train, x_test, y_test, classifiers, nodes)
    all_metrics.extend(metrics)

# %%
# for metric in all_metrics:
#     print(f"\nClassifier: {metric['Classifier']}, Nodes: {metric['Nodes']}")
#     print(f"Accuracy: {metric['Accuracy']:.4f}")
#     print(f"Precision: {metric['Precision']:.4f}")
#     print(f"Recall: {metric['Recall']:.4f}")
#     print(f"F1-Score: {metric['F1-Score']:.4f}")
#     print("Classification Report:")
#     print(metric['Classification Report'])
#     print("Confusion Matrix:")
#     print(metric['Confusion Matrix'])

for metric in all_metrics:
    print(f"{metric['Classifier']}\t{metric['Accuracy']:.4f}\t{metric['Precision']:.4f}\t{metric['Recall']:.4f}\t{metric['F1-Score']:.4f}")


# %%



