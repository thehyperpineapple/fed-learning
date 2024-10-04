# %%
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# %%
broken_path = os.listdir('../datasets/gearbox_fault/archive/BrokenTooth')
healthy_path = os.listdir('../datasets/gearbox_fault/archive/Healthy')
broke_df = pd.DataFrame()
healthy_df = pd.DataFrame()



# %%
for df in broken_path:
    df = pd.read_csv(os.path.join('../datasets/gearbox_fault/archive/BrokenTooth', df))
    broke_df = pd.concat([broke_df, df])

# %%
for df in healthy_path:
    df = pd.read_csv(os.path.join('../datasets/gearbox_fault/archive/Healthy', df))
    healthy_df = pd.concat([healthy_df, df])

# %%
broke_df.info()

# %%
healthy_df.info()

# %%
# broke_df = pd.read_csv('../datasets/gearbox_fault/b30hz50.csv')
# healthy_df = pd.read_csv('../datasets/gearbox_fault/h30hz50.csv', nrows=94208)

# %%
broke_df['condition'] = 'Broken'
healthy_df['condition'] = 'Healthy'


# %%
combined_df = pd.concat([broke_df.sample(n=6000, random_state=42), healthy_df.sample(n=15000, random_state=42)], ignore_index=True)

# %%
combined_df.info()

# %%
category_counts = combined_df['condition'].value_counts()

# %%
category_counts

# %%
le = LabelEncoder()
combined_df['encoded_condition'] = le.fit_transform(combined_df['condition'])

# %%
combined_df['encoded_condition']

# %%
print("Classes:", le.classes_)
print("Encoded values:", le.transform(le.classes_))

# %%
# 1. Z-Score Method
def detect_outliers_zscore(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    return data[z_scores > threshold]

# 2. IQR Method
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data < lower_bound) | (data > upper_bound)]



# %%
columns = ['a1', 'a2', 'a3', 'a4']
# Apply the methods
df_cleaned = combined_df.copy()

# Loop through each column in the list and remove outliers
for column in columns:
    # Apply the Z-Score and IQR methods
    outliers_zscore = detect_outliers_zscore(df_cleaned[column])
    outliers_iqr = detect_outliers_iqr(df_cleaned[column])
    
    # Combine the outliers from both methods
    outliers_combined = pd.concat([outliers_zscore, outliers_iqr]).drop_duplicates()
    
    # Remove the rows with outliers from the dataset
    df_cleaned = df_cleaned[~df_cleaned[column].isin(outliers_combined)]
    
    # Visualize each column after removing outliers
    sns.boxplot(x=df_cleaned[column])
    plt.title(f"Boxplot of {column} after removing outliers")
    plt.show()


# %%
df_cleaned.info()

# %%
counts = df_cleaned['condition'].value_counts()

# %%
counts

# %%
# df_cleaned.to_csv('../datasets/gearbox_fault/gearbox_fault_data.csv')

# %%
scaler = StandardScaler()
df_cleaned[['a1', 'a2', 'a3', 'a4']] = scaler.fit_transform(df_cleaned[['a1', 'a2', 'a3', 'a4']])

# %%
df_cleaned.drop(columns = 'condition')

# %%
df_cleaned = df_cleaned[['a1','a2','a3','a4','encoded_condition']]

# %%
correlation_full_health = df_cleaned.corr()

axis_corr = sns.heatmap(
correlation_full_health,
vmin=-1, vmax=1, center=0,
cmap=sns.diverging_palette(50, 500, n=500),
square=True
)

plt.show()

# %%
combined_df = combined_df[['a1','a2','a3','a4','encoded_condition']]

# %%
correlation_full_health = combined_df.corr()

axis_corr = sns.heatmap(
correlation_full_health,
vmin=-1, vmax=1, center=0,
cmap=sns.diverging_palette(50, 500, n=500),
square=True
)

plt.show()

# %%
df_cleaned.to_csv('../datasets/gearbox_fault/balanced_gearbox_fault_data.csv')

# %%
df_cleaned.head()

# %%
from sklearn.decomposition import PCA

# %%

pca = PCA(n_components=2)  # Specify the number of components you want to keep
pca_result = pca.fit_transform(df_cleaned[["a1","a2","a3","a4"]])

# Step 5: Create a DataFrame with the reduced dimensions
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])

# Optional: Visualize explained variance
explained_variance = pca.explained_variance_ratio_
print("Explained variance ratio: ", explained_variance)

# Output the DataFrame with reduced dimensions
pca_df = pca_df.round(6)
print(pca_df)


# %%
pca_df['encoded_condition'] = combined_df['encoded_condition']

# %%
pca_df.head()

# %%
combined_df.head()

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
pca_df.to_csv('../datasets/gearbox_fault/pca_balanced_gearbox_fault_data.csv')

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pyswarms as ps

df = pd.read_csv('../datasets/gearbox_fault/pca_balanced_gearbox_fault_data.csv', index_col=[0])
# Split features and target
X = df.drop(columns = ['encoded_condition'])
y = df['encoded_condition']

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
# Define objective function for PSO
def objective_function(params):
    params = params[0]
    n_estimators = int(params[0])
    max_depth = int(params[1])
    
    # Train Random Forest with PSO-selected hyperparameters
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict and calculate accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return 1 - accuracy  # Minimize 1 - accuracy (maximize accuracy)

# Define bounds for hyperparameters: (n_estimators, max_depth)
bounds = (np.array([10, 5]), np.array([200, 20]))

# Set up PSO optimizer
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options={'c1': 0.5, 'c2': 0.3, 'w': 0.9}, bounds=bounds)

# Run optimization
best_cost, best_pos = optimizer.optimize(objective_function, iters=10)

# Output best hyperparameters
n_estimators_optimal = int(best_pos[0])
max_depth_optimal = int(best_pos[1])

print(f"Optimal n_estimators: {n_estimators_optimal}, Optimal max_depth: {max_depth_optimal}")


# %%



