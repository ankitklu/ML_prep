import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=500, centers=3, cluster_std=0.60, random_state=42)

df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])

# print(df)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

intertia = []
K_range = range(1,11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    intertia.append(kmeans.inertia_)

# plt.plot(K_range, intertia, marker='o')
# plt.title('Elbow Method for Optimal k')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Inertia')
# plt.xticks(K_range)
# plt.show()


kmeans_final = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans_final.fit_predict(X_scaled)

df['cluster'] = cluster_labels

sns.scatterplot(data=df, x='Feature1', y='Feature2', hue=df['cluster'], palette='viridis')
plt.title('K-Means Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()