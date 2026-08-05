import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Student_Social_Media_And_Mental_Health_Impact.csv')

#looking fro number of rows and columns in the dataset
# print(df.shape)

#looking at the first few rows of the dataset
# print(df.head())


# print(df['Stress_Level'].unique())
order = ['Low', 'Medium', 'High', 'Very High']

sns.boxplot(x=df['Stress_Level'], y=df['Mental_Health_Score'], order=order)
# plt.show()

df.duplicated().sum()


df.info()

# Reading this: Most columns look reasonable — but look closely at Physical_Activity_Hours: the minimum value is -0.4. Negative hours aren't physically possible, so this is a data-entry glitch, not a real value. We'll fix this properly in the Data Cleaning step below instead of ignoring it.
df.describe()

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

df['Most_Used_Platform'].value_counts()

plt.figure(figsize=(8, 4))
sns.countplot(x=df['Most_Used_Platform'], order=df['Most_Used_Platform'].value_counts().index)


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
print(df['Grouped_country'].value_counts())