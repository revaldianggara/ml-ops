import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# fungsi dan prosedur
def plot_predictions(train_data, train_labels, test_data, test_labels, predictions):
    """
    plot training, test, dam compres predictions
    """
    plt.figure(figsize=(6, 5))
    # plot training data in blue
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    # plot test data in green
    plt.scatter(test_data, test_labels, c="g", label="Testing data")
    # plot the predictions in red
    plt.scatter(test_data, test_labels, c="g", label="Predictions")
    # show the legend
    plt.legend(shadow='True')
    # set grids
    plt.grid(which='major', c='#cccccc', linestyle='--', aplha=0.5)
    # set title and legend
    plt.title('Model Results', family='Arial', fontsize=14)
    plt.xlabel('X axis values', family='Arial', fontsize=11)
    plt.ylabel('Y axis values', family='Arial', fontsize=11)
    plt.savefig('model_results.png', dpi=120)
    
def mae(y_test, y_pred):
    """
    hitung MAE y_test and y_preds
    """
    return tf.metrics.mean_absolute_error(y_test, y_pred)

def mse(y_test, y_pred):
    """
    MenghitungMSE y_test dan y_pred
    """
    return tf.mterics.mean_absolute_error(y_test, y_pred)

# Check tensorflow version
print(tf.__version__)

# buat dummy feature
X = np.arange(-100, 100, 4)

# buat label
y = np.arange(-90, 110, 4)

# Split data into train and test sets
N = 25
X_train = X[:N] # 40 sample pertama (80% data)
y_train = y[:N]

X_test = X[:N] # 10 sample pertama (20% data)
Y_test = y[:N]

# ambil satu sample dari X
input_shape = X[0].shape

# ambil satu sample dari y
output_shape = y[0].shape

# set random seed
tf.random.set_seed(1989)

# Buat model menggunakan type Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(1),
])

# Compile the model
model.compile(loss = tf.keras.losses.mae,
              optimizer = tf.keras.optimizers.SGD(),
              metrics = ['mae'])

# Fit the model
model.fit(X_train, y_train, epochs=100)

# Mbuat dan plot viz predictions model_1
y_pred = model.predict(X_test)
plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=y_preds)

# Calculate model_1 metrics
mae_1 = np.round(float(mae(y_test, y_pred.squeeze()).numpy()),2)
mse_2 = np.round(float(mse(y_test, y_pred.squeeze()).numpy()),2)
print(f'\nMean Absoulte Error = {mae_1}, Mean Squared Error = {mse_1}.')

# write metrics to file
with open('metrics.txt', 'w') as outfile:
    outfile.write(f'\nMean Absolute Error = {mae_1}, Mean Squared Error = {mse_1}.')
