import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Create a synthetic, imbalanced two-dimensional dataset
X, y = make_classification(n_classes=3, n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=1, weights=[0.05, 0.15, 0.8], n_samples=500, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize samplers
ros = RandomOverSampler(random_state=42)
rus = RandomUnderSampler(random_state=42)

# Apply over-sampling to the training data
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

# Apply under-sampling to the training data
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

# Plotting the results
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Titles for the plots
titles = ['Original Distribution', 'After Over Sampling', 'After Under Sampling']

# Data for plotting
datasets = [(X_train, y_train), (X_train_ros, y_train_ros), (X_train_rus, y_train_rus)]

# Colors for the classes
colors = ['red', 'blue', 'green']

for i, (data, title) in enumerate(zip(datasets, titles)):
    X, y = data
    for label in np.unique(y):
        axs[i].scatter(X[y == label][:, 0], X[y == label][:, 1], c=colors[label], label=f'Class {label}', alpha=0.5)
    axs[i].set_title(title)
    axs[i].set_xlabel('Feature 1')
    axs[i].set_ylabel('Feature 2')
    axs[i].legend()

plt.tight_layout()
plt.show()
