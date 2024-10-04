<<<<<<< HEAD
# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# %%
df = pd.read_csv("../datasets/predictive_maintenance/predictive_maintenance.csv")

# %% [markdown]
=======
import pandas as pd

df = pd.read_csv("../../Datasets/Machine Predictive Maintenance Classification/predictive_maintenance.csv")

>>>>>>> 56dae2d39cc93cd6b73324bcdf3e57c8b1f32ca0
# ### Feature Columns
# 
# - UID: unique identifier ranging from 1 to 10000
# - productID: consisting of a letter L, M, or H for low (50% of all products), medium (30%), and high (20%) as product quality variants and a variant-specific serial number
# - air temperature [K]: generated using a random walk process later normalized to a standard deviation of 2 K around 300 K
# - process temperature [K]: generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.
# - rotational speed [rpm]: calculated from powepower of 2860 W, overlaid with a normally distributed noise
# - torque [Nm]: torque values are normally distributed around 40 Nm with an Ïƒ = 10 Nm and no negative values.
# - tool wear [min]: The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process. and a
# -'machine failure' label that indicates, whether the machine has failed in this particular data point for any of the following failure modes are true.
# 
# ### Target Columns
# 
# There are two Targets (Do not make the mistake of using one of them as feature, as it will lead to leakage)
# - Target : Failure or Not
# - Failure Type : Type of Failure

<<<<<<< HEAD
# %%
df.info()

# %%
df['Target'].value_counts()

# %%
# scaler = StandardScaler()
=======
>>>>>>> 56dae2d39cc93cd6b73324bcdf3e57c8b1f32ca0
scaling_features = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"] 
# df_scaled = scaler.fit_transform(df_sampled[scaling_features])
for feature in scaling_features:
    df[feature] = df[feature]/df[feature].max()


<<<<<<< HEAD
# %%
# df_sampled.describe()
df.describe()

# %%
df = df.drop(['UDI', 'Product ID', 'Type', 'Failure Type'], axis=1)

# %%
df.head()

# %%
# Separate features (X) and target (y)
X = df.drop('Target', axis=1)  # Features
Y = df['Target']  # Target

# %%
smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X, Y)

# %%
# Concatenate features and target
df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                          pd.DataFrame(Y_resampled, columns=['Target'])], axis=1)

# %%
df_resampled

# %%
# check version number
df_resampled['Target'].value_counts()

# %%
correlation_full_health = X_resampled.corr()

axis_corr = sns.heatmap(
correlation_full_health,
vmin=-1, vmax=1, center=0,
cmap=sns.diverging_palette(50, 500, n=500),
square=True
)

plt.show()

# %%
cov_matrix = X_resampled.cov()

# Plot the covariance matrix using seaborn
sns.heatmap(cov_matrix, annot=True, cmap='coolwarm')
plt.title("Covariance Matrix Heatmap")
plt.show()

# %%
df.describe()

# %%
from sklearn.decomposition import PCA
pca = PCA(n_components=1)  # Specify the number of components you want to keep
pca_result_1 = pca.fit_transform(df_resampled[["Air temperature [K]","Process temperature [K]"]])
pca_result_2 = pca.fit_transform(df_resampled[[ "Rotational speed [rpm]", "Torque [Nm]"]])

# Step 5: Create a DataFrame with the reduced dimensions
pca_df = pd.DataFrame(pca_result_1, columns=['PC1'])
pca_df['PC2'] = pd.DataFrame(pca_result_2, columns=['PC2'])

# Optional: Visualize explained variance
explained_variance = pca.explained_variance_ratio_
print("Explained variance ratio: ", explained_variance)

# Output the DataFrame with reduced dimensions
pca_df = pca_df.round(6)
print(pca_df)


# %%
pca_df['Tool Wear'] = df_resampled['Tool wear [min]']
pca_df['Target'] = df['Target']

# %%
correlation_full_health = pca_df.corr()

axis_corr = sns.heatmap(
correlation_full_health,
vmin=-1, vmax=1, center=0,
cmap=sns.diverging_palette(50, 500, n=500),
square=True
)

plt.show()

# %%
pca_df['Target'] = df_resampled['Target']

# %%
pca_df.to_csv("../datasets/predictive_maintenance/balanced_pca_binary_classification.csv")

# %%
df_resampled.to_csv("../datasets/predictive_maintenance/balanced_binary_classification.csv")

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pyswarms as ps

df = pd.read_csv('../datasets/predictive_maintenance/balanced_pca_binary_classification.csv')
# Split features and target
X = df.drop(columns = ['Target'])
y = df['Target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define objective function for PSO
def objective_function(params):
    params = params[0]
    C = params[0]
    max_iter = int(params[1])
    
    # Train Logistic Regression with PSO-selected hyperparameters
    clf = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict and calculate accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return 1 - accuracy  # Minimize 1 - accuracy (maximize accuracy)

# Define bounds for hyperparameters: (C, max_iter)
bounds = (np.array([0.01, 100]), np.array([100, 1000]))  # C between 0.01 and 100, max_iter between 100 and 1000

# Set up PSO optimizer
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options={'c1': 0.5, 'c2': 0.3, 'w': 0.9}, bounds=bounds)

# Run optimization
best_cost, best_pos = optimizer.optimize(objective_function, iters=10)

# Output best hyperparameters
C_optimal = best_pos[0]
max_iter_optimal = int(best_pos[1])

print(f"Optimal C: {C_optimal}, Optimal max_iter: {max_iter_optimal}")

# %%



=======
df = df.drop(['UDI', 'Product ID', 'Type'], axis=1)

# Save datasets
binary_df = multi_class_df = df
binary_df = binary_df.drop(columns=["Failure Type"])
multi_class_df = multi_class_df.drop(columns=["Target"])
binary_df.to_csv("../../Datasets/Machine Predictive Maintenance Classification/binary_classification.csv")
multi_class_df.to_csv("../../Datasets/Machine Predictive Maintenance Classification/multi_class_classification.csv")
>>>>>>> 56dae2d39cc93cd6b73324bcdf3e57c8b1f32ca0
