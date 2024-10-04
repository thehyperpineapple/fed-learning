# %%
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# %%
errors_df = pd.read_csv('../datasets/telemetry_analysis/raw/PdM_errors.csv')
failures_df = pd.read_csv('../datasets/telemetry_analysis/raw/PdM_failures.csv')
machines_df = pd.read_csv('../datasets/telemetry_analysis/raw/PdM_machines.csv')
maint_df = pd.read_csv('../datasets/telemetry_analysis/raw/PdM_maint.csv')
telemetry_df = pd.read_csv('../datasets/telemetry_analysis/raw/PdM_telemetry.csv')

df_list = [errors_df, failures_df, machines_df, maint_df, telemetry_df]

# %%
telemetry_df = telemetry_df.merge(errors_df, on=['datetime', 'machineID'], how='left')
# telemetry_df['error'].fillna('no error')

# %%
telemetry_df['errorID'] = telemetry_df['errorID'].apply(lambda x: 'error' if pd.notnull(x) else x)
telemetry_df['errorID'] = telemetry_df['errorID'].fillna('no error')

# %%
telemetry_df['errorID'].value_counts()

# %%


# %%
scaling_features = ['volt', 'rotate', 'pressure','vibration']

for feature in scaling_features:
    telemetry_df[feature] = telemetry_df[feature]/telemetry_df[feature].max()

# %%
le = LabelEncoder()
telemetry_df['encoded_errors'] = le.fit_transform(telemetry_df['errorID'])

# %%
telemetry_df = telemetry_df.drop(columns = ["datetime", "errorID"], axis=1)

# %%
telemetry_df.head(10)

# %%
print("Classes:", le.classes_)
print("Encoded values:", le.transform(le.classes_))

# %%


# %%
class_0 = telemetry_df[telemetry_df['encoded_errors'] == 0]
class_1 = telemetry_df[telemetry_df['encoded_errors'] == 1]

# Randomly sample 339 values from each class
class_0_sample = class_0.sample(n=3919, random_state=42)
class_1_sample = class_1.sample(n=3919, random_state=42)

# Combine the samples
balanced_df = pd.concat([class_0_sample, class_1_sample])

# Optionally, shuffle the dataframe to mix the classes
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# %%
balanced_df['encoded_errors'].value_counts()

# %%
# balanced_df.to_csv('balanced_telemetry_analysis.csv')

# %%
correlation_full_health = balanced_df.corr()

axis_corr = sns.heatmap(
correlation_full_health,
vmin=-1, vmax=1, center=0,
cmap=sns.diverging_palette(50, 500, n=500),
square=True
)

plt.show()

# %%
correlation_full_health = telemetry_df.corr()

axis_corr = sns.heatmap(
correlation_full_health,
vmin=-1, vmax=1, center=0,
cmap=sns.diverging_palette(50, 500, n=500),
square=True
)

plt.show()

# %%



