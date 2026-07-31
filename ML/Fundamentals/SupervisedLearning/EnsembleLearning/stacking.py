import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

df = sns.load_dataset('iris')
# print(df.head())

X = df.drop(['species'], axis=1)
y = df['species']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y_encoded, test_size= 0.2, random_state=42, stratify=y_encoded)

base_learners = [
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('svc', SVC(probability=True, kernel='rbf', random_state=42)),
    ('lr', LogisticRegression(random_state=42))
]

meta_model = LogisticRegression(random_state=42)

stacking_clf = StackingClassifier(estimators=base_learners, final_estimator=meta_model, cv=5)

stacking_clf.fit(X_train, y_train)

y_pred = stacking_clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))