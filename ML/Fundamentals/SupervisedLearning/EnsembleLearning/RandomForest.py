import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df = sns.load_dataset('iris')
# print(df.head())

X = df.drop(['species'], axis=1)
y = df['species']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y_encoded, test_size= 0.2, random_state=42, stratify=y_encoded)

rf_model = RandomForestClassifier(
    n_estimators = 100,   # No of trees
    max_depth= None,   # let the trees grow fully
    random_state=42
)

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)

print(classification_report(y_test, y_pred))

