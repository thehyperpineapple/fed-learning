import pandas as pd

df = pd.read_csv("../../Datasets/Machine Predictive Maintenance Classification/predictive_maintenance.csv")

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

# Concatenate the sampled dataframes
df_sampled = pd.concat([df[df['Type'] == 'H'].sample(n=1003, random_state=42),
                        df[df['Type'] == 'L'].sample(n=1003, random_state=42), 
                        df[df['Type'] == 'M'].sample(n=1003, random_state=42)])


# Normalize features
scaling_features = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"] 

for feature in scaling_features:
    df_sampled[feature] = df_sampled[feature]/df_sampled[feature].max()


# Save datasets
binary_df = df.drop(columns=["UDI", "Product ID", "Failure Type"])
multi_class_df = df.drop(columns=["UDI", "Product ID", "Target"])
binary_df.to_csv("../../Datasets/Machine Predictive Maintenance Classification/binary_classification.csv")
multi_class_df.to_csv("../../Datasets/Machine Predictive Maintenance Classification/multi_class_classification.csv")