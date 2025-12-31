import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset (assuming you have the csv file from the notebook)
df = pd.read_csv('dataset/heart_cleveland_upload.csv')

print(df)


# Separate Target and Features
target = 'condition'
X = df.drop(columns=[target])  # All columns except 'condition'
print(X)
y = df[target]                 # 'condition' column (0 or 1)

X_scaled = normalize(X, norm='l2', axis=0)
print(X_scaled)


# Apply PCA to reduce to 3 dimensions
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for visualization
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])
pca_df['Condition'] = y

# Calculate how much information (variance) we kept
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
# Example Output: [0.23, 0.12] (Means PC1 holds 23% of info, PC2 holds 12%)



# Set up the figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
# We loop through targets to assign colors manually
targets = [0, 1]
colors = ['b', 'r']
labels = ['No Disease', 'Disease']

for target, color, label in zip(targets, colors, labels):
    indicesToKeep = y == target
    ax.scatter(
        X_pca[indicesToKeep, 0], # PC1
        X_pca[indicesToKeep, 1], # PC2
        X_pca[indicesToKeep, 2], # PC3
        c=color,
        s=50,
        alpha=0.6,
        label=label
    )

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.legend()
plt.title('3D PCA of Heart Disease')
plt.show()