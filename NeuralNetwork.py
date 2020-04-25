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
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Define layers with neurons and activation function
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(4, activation="relu"))
model.add(tf.keras.layers.Dense(4, activation="relu"))
model.add(tf.keras.layers.Dense(4, activation="relu"))
model.add(tf.keras.layers.Dense(1))

# Compile model with learning rate optimizer and loss function
model.compile(optimizer="rmsprop", loss="mse")

# Train the model
model.fit(x=X_train, y=y_train, epochs=250, verbose=1)

# Plot error vs. epochs
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
plt.show()
