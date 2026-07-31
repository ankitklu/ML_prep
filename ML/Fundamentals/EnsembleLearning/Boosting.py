import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

df = sns.load_dataset('iris')
# print(df.head())

X = df.drop(['species'], axis=1)
y = df['species']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y_encoded, test_size= 0.2, random_state=42, stratify=y_encoded)

ada_model = AdaBoostClassifier(
    n_estimators = 100,   # No of trees
    learning_rate=1.0,   # learning rate
    random_state=42
)

ada_model.fit(X_train, y_train)

y_pred = ada_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# print(accuracy)
 
# print(classification_report(y_test, y_pred)) 


## Gradient Boosting Model
gb_model = GradientBoostingClassifier(
    n_estimators = 100, # No of trees
    learning_rate = 0.1, # learning rate
    random_state=42
)

gb_model.fit(X_train, y_train)
y_pred = gb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# print(accuracy)
 
# print(classification_report(y_test, y_pred))


## XGBoost Model
xgb_model = XGBClassifier(
    n_estimators = 100,
    learning_rate = 0.1,
    ax_depth = 3,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)

xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
print(classification_report(y_test, y_pred))
