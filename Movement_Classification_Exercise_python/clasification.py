import numpy as np
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd

def classify_movement(file_path):
  """ This method first extracts the movement features and lables, then uses naive bayes to classify the movements.
  """
  # Load the accelerometer file
  accelerometer_csv = np.loadtxt(file_path, delimiter=',')
  header_list = ["X", "Y", "Z", "Label"]

  df = pd.read_csv(file_path, delimiter=',', names=header_list)
  #features, labels = extract_features_and_labels(accelerometer_csv)
  features, labels = extract_features_and_labels(df)
  classification_accuracy(features, labels)

def get_window_bounds(self, num_values, min_periods, center, closed):
        start = np.empty(num_values, dtype=np.int64)
        end = np.empty(num_values, dtype=np.int64)
        for i in range(num_values):
            if self.use_expanding[i]:
                start[i] = 0
                end[i] = i + 1
            else:
                start[i] = i
                end[i] = i + self.window_size
        return start, end
    


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
  print(raw_data)
  ## TODO: Compute the features and corresponding labels, and shuffle the data.
  features, labels = [], []

  numbers_series = pd.Series(raw_data['X'])
  # calculating the mean of segmented data with the window lenght of 128 without any overlap
  a = raw_data.rolling(window=WINDOW_LENGTH, min_periods=1).mean()[::WINDOW_LENGTH].reset_index(drop=True)
  # calculating the variance of segmented data with the window lenght of 128 without any overlap
  b = raw_data.rolling(window=WINDOW_LENGTH, min_periods=1).std()[::WINDOW_LENGTH].reset_index(drop=True)
  # Finding the mean of the 128 labels and rounding it to 0 or 1 (if more than 0.5 then most of the labels were 1 then rounde to 1 and vice versa.)
  c = raw_data['Label'].rolling(window=WINDOW_LENGTH, min_periods=1).mean()[::WINDOW_LENGTH].reset_index(drop=True)
  #print("test:", a['X'])
  #print("test:", b['X'])
  #print("test:", c)

  features = {'XM': a['X'], 'XV': b['X'],'YM': a['Y'], 'YV': b['Y'],'ZM': a['Z'], 'ZV': b['Z'], 'L': c}
  features = pd.DataFrame(data=features)


  features = np.array(features)
  print("features", features)
  labels = round(c)
  labels = np.array(labels)
  idx = np.arange(features.shape[0])
  np.random.shuffle(idx)
  features = features[idx]
  labels = labels[idx]

  return features, labels


def classification_accuracy(features, labels):
  """ Use Naive Bayes and cross-validation (supported by scikit-learn) to show the average accurcy for the classification.
  """
  features_new = features[~np.isnan(features).any(axis=1)]
  labels_new = labels[~np.isnan(features).any(axis=1)]
  print("fea",features)

  X_train, X_test, y_train, y_test = train_test_split(features_new, labels_new, test_size=0.3, random_state=0)
  gnb = GaussianNB()
  y_pred = gnb.fit(X_train, y_train).predict(X_test)
  error =  (y_test != y_pred).sum() / X_test.shape[0]
  print("Classification Accuracy:", 1-error)


if __name__ == "__main__":
  classify_movement("./accelerometer_movement.csv")


