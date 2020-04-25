import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Import data set
df = pd.read_csv('./DATA/fake_reg.csv')

# Define features / input from data set
X = df[['feature1', 'feature2']].values

# Define label / output from data set
y = df['price'].values

# Split data set in training and test groups
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale input data to be between 0 and 1
MinMaxScaler().fit(X_train)
X_train = MinMaxScaler().transform(X_train)
X_test = MinMaxScaler().transform(X_test)

# Define layers with neurons and activation function
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(4, activation="relu"))
model.add(tf.keras.layers.Dense(4, activation="relu"))
model.add(tf.keras.layers.Dense(4, activation="relu"))
model.add(tf.keras.layers.Dense(1))