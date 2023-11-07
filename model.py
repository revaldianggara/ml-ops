# Import modules and packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Functions and procedures
def plot_predictions(train_data, train_labels,  test_data, test_labels,  predictions):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(6, 5))
  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", label="Training data")
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", label="Testing data")
  # Plot the predictions in red (predictions were made on the test data)
  plt.scatter(test_data, predictions, c="r", label="Predictions")
  # Show the legend
  plt.legend(shadow='True')
  # Set grids
  plt.grid(which='major', c='#cccccc', linestyle='--', alpha=0.5)
  # Some text
  plt.title('Model Results', family='Arial', fontsize=14)
  plt.xlabel('X axis values', family='Arial', fontsize=11)
  plt.ylabel('Y axis values', family='Arial', fontsize=11)
  # Show
  plt.savefig('model_results.png', dpi=120)



def mae(y_test, y_pred):
  ""