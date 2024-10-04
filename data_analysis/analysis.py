# %%
import pandas as pd
import matplotlib.pyplot as plt


# %%
# pre_pca = pd.read_csv("data/pre-pca.csv")
pre_pca_pso = pd.read_csv("data/pre-pca-pso.csv")
# post_pca = pd.read_csv("data/post-pca.csv")
post_pca_pso = pd.read_csv("data/post-pca-pso.csv")

# %%
# List of models
models = ['Logistic Regression', 'Ridge Classifier', 'Decision Tree', 'Random Forest', 'AdaBoost', 'Gradient Boosting']

# Create the figure and subplots
fig, axes = plt.subplots(3, 2, figsize=(12, 18))
axes = axes.flatten()  # Flatten the 2x3 grid of subplots into a single array

# Plot for each model
for i, model in enumerate(models):
    ax = axes[i]
    
    # Filter data for the model
    # pre_pca_model = pre_pca[pre_pca['Models'] == model]
    pre_pca_pso_model = pre_pca_pso[pre_pca_pso['Models'] == model]
    # post_pca_model = post_pca[post_pca['Models'] == model]
    post_pca_pso_model = post_pca_pso[post_pca_pso['Models'] == model]
    
    # Plot for each dataset
    # ax.plot(pre_pca_model['Nodes'], pre_pca_model['Accuracy'], label='pre-pca', marker='o')
    ax.plot(pre_pca_pso_model['Nodes'], pre_pca_pso_model['Accuracy'], label='pre-PCA', marker='o')
    # ax.plot(post_pca_model['Nodes'], post_pca_model['Accuracy'], label='post-pca', marker='o')
    ax.plot(post_pca_pso_model['Nodes'], post_pca_pso_model['Accuracy'], label='post-PCA', marker='o')
    
    # Set titles and labels
    ax.set_title(model)
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)

# Adjust the layout and show the plots
plt.tight_layout()
plt.show()

# %%
# List of models in the dataset (modify this if you have different models)
models = ['Logistic Regression', 'Ridge Classifier', 'Decision Tree', 'Random Forest', 'AdaBoost', 'Gradient Boosting']

# Create a plot to compare accuracies across models varying with nodes
plt.figure(figsize=(10, 6))

# Loop over each model and plot accuracy as a function of nodes
for model in models:
    model_data = pre_pca_pso[pre_pca_pso['Models'] == model]  # Filter data for the model
    plt.plot(model_data['Nodes'], model_data['Accuracy'], label=model, marker='o')

# Set plot labels and title
# plt.title('Model Accuracies in Machine Predictive Maintenance Classification Dataset Varying with Nodes', fontsize=14, fontweight='bold')
plt.xlabel('Number of Nodes', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)

# Show legend and grid
plt.legend(title='Models', loc='upper right')
plt.grid(True)

# Display the plot
plt.tight_layout()
plt.show()

# %%



