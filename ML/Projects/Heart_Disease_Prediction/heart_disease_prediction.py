from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import joblib

df = pd.read_csv('heart.csv')

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'SVM' : SVC()
}

result = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = model.score(X_test_scaled, y_test)
    f1 = f1_score(y_test, y_pred)
    result.append(
        {
            'model': name,
            'Accuracy': round(acc,4),
            'f1 score': round(f1,4)
        }
    )

print(pd.DataFrame(result))

joblib.dump(models['Decision Tree'],'Decision_Tree_heart.pkl')
joblib.dump(scaler,'Scaler_heart.pkl')
joblib.dump(X.columns.tolist(),'columns.pkl')