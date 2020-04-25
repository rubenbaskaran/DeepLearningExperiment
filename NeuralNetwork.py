import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
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
plt.title("Training Loss per Epoch")
plt.show()

# Evaluation (training and test scores should be close to each other)
# Loss (MSE) on training set
training_score = model.evaluate(X_train, y_train, verbose=0)
# Loss (MSE) on test set
test_score = model.evaluate(X_test, y_test, verbose=0)
print("Training MSE score: " + str(training_score))
print("Test MSE score: " + str(test_score))

# Make predictions on test data
# Plot predictions and ground truth values of test set next to each other
test_predictions = model.predict(X_test)
test_predictions = pd.Series(test_predictions.reshape(300, ))
test_groundtruths = pd.DataFrame(y_test, columns=["Test set groundtruths"])
test_groundtruths = pd.concat([test_groundtruths, test_predictions], axis=1)
test_groundtruths.columns = ["Test set groundtruths", "Test set predictions"]
print(test_groundtruths)

# Show error data
# Should be (close to) a straight line
sns.scatterplot(x="Test set groundtruths", y="Test set predictions", data=test_groundtruths)
plt.show()

print("MAE: " + str(
    mean_absolute_error(test_groundtruths["Test set groundtruths"], test_groundtruths["Test set predictions"])))
print("MSE: " + str(
    mean_squared_error(test_groundtruths["Test set groundtruths"], test_groundtruths["Test set predictions"])))

# Test model on future data
new_data = [[998, 1000]]
new_data = scaler.transform(new_data)
print("Price of new data: " + str(model.predict(new_data)))

# Save model
model.save("my_price_calculator_model")

# Load and use saved model
loaded_model = load_model("my_price_calculator_model")
print("Loaded model prediction: " + str(loaded_model.predict(new_data)))
