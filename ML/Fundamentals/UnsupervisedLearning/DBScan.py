import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X, y_true = make_moons(n_samples=500, noise=0.1, random_state=42)

df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])

# print(df)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

df['dbscan_cluster'] = dbscan_labels

sns.scatterplot(x=df['Feature1'], y=df['Feature2'], hue=df['dbscan_cluster'], palette='tab10')

plt.title('DB Scan Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()