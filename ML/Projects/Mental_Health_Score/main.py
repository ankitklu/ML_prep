import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Student_Social_Media_And_Mental_Health_Impact.csv')

#looking fro number of rows and columns in the dataset
# print(df.shape)

#looking at the first few rows of the dataset
# print(df.head())


print(df['Stress_Level'].unique())
order = ['Low', 'Medium', 'High', 'Very High']

sns.boxplot(x=df['Stress_Level'], y=df['Mental_Health_Score'], order=order)
plt.show()

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
