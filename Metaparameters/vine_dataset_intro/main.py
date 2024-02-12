# imports
# for dl modeling
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# for data processing
import pandas as pd

# for numerical processing
import numpy as np
import scipy.stats as stats

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# read in data
df = pd.read_csv("data\winequality-red.csv", sep=';')
print(df.shape)
print(df.head())
print()
print(df.describe())

# list of unique values per column
for col in df.columns:
    print(col, len(df[col].unique()))

# pair plot
cols2train = ['fixed acidity', 'volatile acidity', 'citric acid', "quality"]
sns.pairplot(df[cols2train], kind='reg', hue='quality')
plt.show()

# plot data
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax = sns.boxplot(data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.title('before removing outliers')
plt.show()

# remove outliers
df = df[df['total sulfur dioxide'] < 200]

# plot data
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax = sns.boxplot(data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.title('After removing outliers')
plt.show()

# zscore all cols except quality
cols2train = df.keys()
cols2train = cols2train.drop('quality')
for col in cols2train:
    df[col] = stats.zscore(df[col])

# plot data
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax = sns.boxplot(data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.title('After zscore')
plt.show()

# distribution of quality
counts = df['quality'].value_counts()
plt.bar(counts.index, counts.values)
plt.title('Distribution of quality before making to binary classification')
plt.xlabel('quality')
plt.ylabel('count')
plt.show()

# converting to binary classification
df['quality'] = df['quality'].apply(lambda x: 1 if x >= 6 else 0)

# distribution of quality
counts = df['quality'].value_counts()
plt.bar(counts.index, counts.values)
plt.title('Distribution of quality after making to binary classification')
plt.xlabel('quality')
plt.ylabel('count')
plt.show()

# quality in now binary
print(f"unique values in quality: {df['quality'].unique()}")

# organise data
data = torch.tensor(df[cols2train].values, dtype=torch.float32)
labels = torch.tensor(df['quality'].values, dtype=torch.float32)
print(data.shape)
print(labels.shape)

# making the labels 2D
labels = labels.unsqueeze(1)
print(labels.shape)

# train test split
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
print(train_data.shape)
print(test_data.shape)

# making train and test datasets
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)

# making train and test dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
print(len(train_dataloader))
print("batch size: ", train_dataloader.batch_size)
print(len(test_dataloader))
