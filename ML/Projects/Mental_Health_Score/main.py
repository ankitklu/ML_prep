import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('Student_Social_Media_And_Mental_Health_Impact.csv')

#looking fro number of rows and columns in the dataset
# print(df.shape)

#looking at the first few rows of the dataset
# print(df.head())


# print(df['Stress_Level'].unique())
order = ['Low', 'Medium', 'High', 'Very High']

sns.boxplot(x=df['Stress_Level'], y=df['Mental_Health_Score'], order=order)
# plt.show()

# df.duplicated().sum()


# df.info()

# Reading this: Most columns look reasonable — but look closely at Physical_Activity_Hours: the minimum value is -0.4. Negative hours aren't physically possible, so this is a data-entry glitch, not a real value. We'll fix this properly in the Data Cleaning step below instead of ignoring it.
# df.describe()

## Before predicting anything, we need to understand the shape of what we're predicting.
sns.histplot(df['Mental_Health_Score'], kde=True)

# 4.2 — Correlation heatmap
# Which numeric features actually move together with the target?

sns.heatmap(df.corr(numeric_only=True), annot=True)

#Does higher stress genuinely come with a lower score?
order = ['Low', 'Medium', 'High', 'Very High']
sns.boxplot(x='Stress_Level', y='Mental_Health_Score',data=df, order=order)

# Does more time on social media relate to a lower score?
sns.scatterplot(x='Avg_Daily_Usage_Hours', y='Mental_Health_Score',data=df)


# Sleep Hours vs Mental Health Score
# Sleep is one of the most commonly cited mental health factors — does the data back that up?

sns.scatterplot(x='Sleep_Hours_Per_Night', y='Mental_Health_Score', data=df)

#4.6 — Most Used Platform (count)
# A quick look at which platforms are most common in our dataset.

# df['Most_Used_Platform'].value_counts()

# plt.figure(figsize=(8, 4))
# sns.countplot(x=df['Most_Used_Platform'], order=df['Most_Used_Platform'].value_counts().index)


## Checking Outliers

num_features = df.select_dtypes(include='number') #int64 or float64
# print(num_features)

Q1 = num_features.quantile(0.25)
Q3 = num_features.quantile(0.75)
IQR = Q3 - Q1 # Interquartile range

lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR

# The 1.5 * IQR rule is a convention (from box plots) — it's not magic, just a widely-accepted threshold for "unusually far from the typical range."

outliers = (num_features < lower_bound) | (num_features > upper_bound)
# print(outliers.sum())


## Data Cleaning
## Drop the Duplicates

# print(df.describe())

#               Age  Avg_Daily_Usage_Hours  Daily_Unlocks  Study_Hours  Physical_Activity_Hours  Sleep_Hours_Per_Night  Mental_Health_Score
# count  5000.00000            5000.000000    5000.000000  5000.000000              5000.000000            5000.000000          5000.000000
# mean     20.82180               5.078460     171.452600     3.008420                 1.751000               6.634580             6.230980
# std       1.73662               1.653913      42.858254     1.637018                 0.668398               1.221391             1.278701
# min      18.00000               1.000000      62.000000     0.300000                -0.400000               3.600000             3.600000
# 25%      19.00000               3.800000     140.000000     1.500000                 1.300000               5.600000             5.100000
# 50%      21.00000               5.000000     171.000000     2.800000                 1.700000               6.600000             6.100000
# 75%      22.00000               6.300000     204.000000     4.200000                 2.200000               7.500000             7.100000
# max      24.00000               8.800000     273.000000     8.300000                 4.100000               9.900000             9.400000

## we are getting a negative value for Physical_Activity_Hours which is not possible. So we will clip the values and check the data again.

df = df.drop_duplicates()
df['Physical_Activity_Hours'] = df['Physical_Activity_Hours'].clip(lower=0) #clip the values below 0 to 0

# print(df.describe())

#               Age  Avg_Daily_Usage_Hours  Daily_Unlocks  Study_Hours  Physical_Activity_Hours  Sleep_Hours_Per_Night  Mental_Health_Score
# count  4998.000000            4998.000000    4998.000000  4998.000000              4998.000000            4998.000000          4998.000000
# mean     20.822129               5.078491     171.455582     3.008403                 1.751160               6.634654             6.231152
# std       1.736774               1.654097      42.859829     1.636831                 0.667282               1.221561             1.278476
# min      18.000000               1.000000      62.000000     0.300000                 0.000000               3.600000             3.600000
# 25%      19.000000               3.800000     140.000000     1.500000                 1.300000               5.600000             5.100000
# 50%      21.000000               5.000000     171.000000     2.800000                 1.700000               6.600000             6.100000
# 75%      22.000000               6.300000     204.000000     4.200000                 2.200000               7.500000             7.100000
# max      24.000000               8.800000     273.000000     8.300000                 4.100000               9.900000             9.400000


## Skewness

num_cols = df.select_dtypes(include='number')
num_cols.skew()
# near to 0 -> Centralized
# Greater than 0 -> Right Skewed
# Less than 0 -> Left Skewed


## Feature Engineering
# One meaningful engineered feature here: grouping Country.

# Country has 111 unique values in this dataset — one-hot encoding that directly would add 110+ mostly-empty columns, which hurts the model far more than it helps (this is called high cardinality).
# Dropping Country entirely throws away real signal — a student's country genuinely correlates with things like internet access, culture, and sleep norms.
# The fix: keep the top 10 most frequent countries as their own category, and bucket everything else into "Other". We keep the signal that matters and lose the noise that doesn't.

top_countries = df['Country'].value_counts().index[:10].tolist()
     

# print(df['Country'].value_counts().index[:10].tolist())
top_countries = df['Country'].value_counts().index[:10].tolist()


def group_countries(country):
    if country in top_countries:
        return country
    else:
        return 'Other'

df['Grouped_country'] = df['Country'].apply(group_countries)
# print(df['Grouped_country'].value_counts())

## 8. Encoding Strategy
# Before we jump into code, let's decide how each categorical column should be encoded — this decision matters more than the code itself.

# Stress_Level → Ordinal Encoding. Its categories have a real, meaningful order: Low < Medium < High < Very High. We already saw in EDA (section 4.3) that the score drops step by step as stress increases — encoding it as 0, 1, 2, 3 preserves that order for the model.
# Gender, Academic_Level, Most_Used_Platform, Purpose_Of_Use, Country_Grouped → One-Hot Encoding. These categories have no natural order — "Instagram" isn't "greater than" "LinkedIn". One-hot encoding creates a separate 0/1 column per category so the model doesn't accidentally assume a false ranking.

skewed_cols = ['Study_Hours']           ## We apply log transformation for skewed columns to make them more normally distributed. This can help some models perform better.
other_numeric_cols = ['Age', 'Avg_Daily_Usage_Hours', 'Daily_Unlocks', 'Physical_Activity_Hours', 'Sleep_Hours_Per_Night']      ## These columns are already fairly normally distributed, so we will perform standard scaler
ordinal_cols = ['Stress_Level']     ## Ordinal encoding for Stress_Level column as it has a natural order.
normal_cols = ['Gender', 'Academic_Level', 'Most_Used_Platform', 'Purpose_Of_Use', 'Grouped_country']       ## One-hot encoding for these columns as they don't have a natural order.

feature_col = skewed_cols + other_numeric_cols + ordinal_cols + normal_cols
X = df[feature_col]
y = df['Mental_Health_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

## 10. Preprocessing using ColumnTransformer
# Our columns need different treatment:

# Study_Hours (the skewed one) → impute → log1p transform → scale
# The other numeric columns → impute → scale (no skew to fix)
# Stress_Level → impute → OrdinalEncoder with an explicit order
# The nominal columns → impute → OneHotEncoder
# We include a SimpleImputer in every branch even though this dataset has zero missing values right now — it's a safety net. Real-world data (and our future API's incoming requests) won't always be this clean, and a pipeline that assumes "no missing values ever" is a pipeline that breaks in production.

# ColumnTransformer glues all of this into one object that applies the right transformation to the right column type in a single .fit() / .transform() call — no manual column-by-column juggling.

# for each of the split we did in the above lines we will create seperate pipelines for each of them and then combine them using ColumnTransformer.

# kewed Features
skew_pipeline = Pipeline(steps=[
    ('log_transform', FunctionTransformer(np.log1p)),
    ('scaler', StandardScaler())
])

# Numeric Features
plain_numeric_pipeline= Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Ordinal Features
ordinal_pipeline = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(categories=[["Low", "Medium", "High", "Very High"]]))
])
# Normal features
normal_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

print("All pipelines created")


## (which_pipeline, which_feature)
preprocessor = ColumnTransformer(transformers=[
    ('skewed', skew_pipeline, skewed_cols),
    ('numeric', plain_numeric_pipeline, other_numeric_cols),
    ('ordinal', ordinal_pipeline, ordinal_cols),
    ('normal', normal_pipeline, normal_cols)
])

# 11. Build a Pipeline
# Why Pipeline matters: a Pipeline chains preprocessing and the model into a single object. Calling .fit() once does both steps in the correct order, and calling .predict() on brand-new raw data automatically applies the exact same preprocessing that was used during training — no risk of forgetting a step or applying it inconsistently.

# Why companies prefer this: when this model gets deployed (which we're doing in Part 2 with FastAPI), the API doesn't need to know anything about scaling, encoding, or log transforms — it just loads one saved pipeline object and calls .predict() on raw input. That's a huge reduction in what can go wrong in production.

# We'll build two pipelines below — one per model — so we can fairly compare them.

lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_test)
lr_preds_train = lr_pipeline.predict(X_train)

lr_r2_testing= r2_score(y_test, lr_preds)
lr_r2_train = r2_score(y_train, lr_preds_train)

lr_mae = mean_absolute_error(y_test, lr_preds)
lr_mse = mean_squared_error(y_test, lr_preds)

print(f"Linear Regression R2 Score on Testing Data: {lr_r2_testing:.4f}")
print(f"Linear Regression R2 Score on Training Data: {lr_r2_train:.4f}")
print(f"Linear Regression Mean Absolute Error: {lr_mae:.4f}")
print(f"Linear Regression Mean Squared Error: {lr_mse:.4f}")

## Results for Linear Regression
# Linear Regression R2 Score on Testing Data: 0.7398
# Linear Regression R2 Score on Training Data: 0.7237
# Linear Regression Mean Absolute Error: 0.5362
# Linear Regression Mean Squared Error: 0.4570


rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('random forest', RandomForestRegressor(random_state=42))
])

rf_pipeline.fit(X_train, y_train)

rf_preds = rf_pipeline.predict(X_test)
rf_preds_train = rf_pipeline.predict(X_train)

rf_r2_testing= r2_score(y_test, rf_preds)
rf_r2_train = r2_score(y_train, rf_preds_train)

print(f"Random Forest R2 Score on Testing Data: {rf_r2_testing:.4f}")
print(f"Random Forest R2 Score on Training Data: {rf_r2_train:.4f}")
print(f"Random Forest Mean Absolute Error: {mean_absolute_error(y_test, rf_preds):.4f}")
print(f"Random Forest Mean Squared Error: {mean_squared_error(y_test, rf_preds):.4f}")

## Results for Random Forest
# Random Forest R2 Score on Testing Data: 0.8780
# Random Forest R2 Score on Training Data: 0.9809
# Random Forest Mean Absolute Error: 0.3465
# Random Forest Mean Squared Error: 0.2142


