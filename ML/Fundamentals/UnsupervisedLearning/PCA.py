#Prinicipal Component Analysis

import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

scaler = StandardScaler()

X,y = make_blobs(n_samples = 500, n_features = 5, centers = 3, cluster_std = 1.5, random_state = 42)
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=2) # convert the 5 features into 2 features
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['label'] = y

sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='label', palette='viridis')
plt.title('PCA Results')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


