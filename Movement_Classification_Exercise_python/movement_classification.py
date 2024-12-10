import numpy as np
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def classify_movement(file_path):
  """ This method first extracts the movement features and lables, then uses naive bayes to classify the movements.
  """
  # Load the accelerometer file
  accelerometer_csv = np.loadtxt(file_path, delimiter=',')

  features, labels = extract_features_and_labels(accelerometer_csv)

  classification_accuracy(features, labels)


def extract_features_and_labels(raw_data):
  """ The method computes mean and variance for every 128 accelerometer data instances
  (i.e., for every 128 rows, you should get 3 mean and 3 variance values for the 3-axes of accelerometer data).
  In other words, we should get one 6 dimensional feature vector for each 128 data points.

  Args:
    raw_data:

  Returns:
    features:
    labels:
  """

  # Use time-window with length 128
  WINDOW_LENGTH = 128

  features = None
  labels = None

  assert raw_data is not None

  ## TODO: Compute the features and corresponding labels, and shuffle the data.
  features, labels = [], []





  features = np.array(features)
  labels = np.array(labels)
  idx = np.arange(features.shape[0])
  np.random.shuffle(idx)
  features = features[idx]
  labels = labels[idx]

  return features, labels


def classification_accuracy(features, labels):
  """ Use Naive Bayes and cross-validation (supported by scikit-learn) to show the average accurcy for the classification.
  """
  ## TODO: Estimate the cross-validation accurcy for the classification using naive bayes.

  
  print("Classification Accuracy:", accurcy)


if __name__ == "__main__":
  classify_movement("./accelerometer_movement.csv")


