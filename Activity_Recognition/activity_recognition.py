import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import precision_recall_fscore_support
import sys
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Index for each activity
activity_indices = {
  'Stationary': 0,
  'Walking-flat-surface': 1,
  'Walking-up-stairs': 2,
  'Walking-down-stairs': 3,
  'Elevator-up': 4,
  'Running': 5,
  'Elevator-down': 6
}


def compute_raw_data(dir_name):

  raw_data_features = None
  raw_data_labels = None
  interpolated_timestamps = None

  sessions = set()
  # Categorize files containing different sensor sensor data
  file_dict = dict()
  # List of different activity names
  activities = set()
  file_names = os.listdir(dir_name)

  for file_name in file_names:
    if '.txt' in file_name:
      tokens = file_name.split('-')
      identifier = '-'.join(tokens[4: 6])
      activity = '-'.join(file_name.split('-')[6:-2])
      sensor = tokens[-1]
 
      sessions.add((identifier, activity))

      if (identifier, activity, sensor) in file_dict:
        file_dict[(identifier, activity, sensor)].append(file_name)
      else:
        file_dict[(identifier, activity, sensor)] = [file_name]


  for session in sessions:
    accel_file = file_dict[(session[0], session[1], 'accel.txt')][0]
    accel_df = pd.read_csv(dir_name + '/' + accel_file)
    accel = accel_df.drop_duplicates(accel_df.columns[0], keep='first').values
 
    # Spine-line interpolataion for x, y, z values (sampling rate is 32Hz).
    # Remove data in the first and last 3 seconds.
    timestamps = np.arange(accel[0, 0]+3000.0, accel[-1, 0]-3000.0, 1000.0/32)

    accel = np.stack([np.interp(timestamps, accel[:, 0], accel[:, 1]),
                      np.interp(timestamps, accel[:, 0], accel[:, 2]),
                      np.interp(timestamps, accel[:, 0], accel[:, 3])],
                     axis=1)

    bar_file = file_dict[(session[0], session[1], 'pressure.txt')][0]
    bar_df = pd.read_csv(dir_name + '/' + bar_file)
    bar = bar_df.drop_duplicates(bar_df.columns[0], keep='first').values
    bar = np.interp(timestamps, bar[:, 0], bar[:, 1]).reshape(-1, 1)

    # Apply lowess to smooth the barometer data with window-size 128
    # bar = np.convolve(bar[:, 0], np.ones(128)/128, mode='same').reshape(-1, 1)
    bar = sm.nonparametric.lowess(bar[:, 0], timestamps, return_sorted=False).reshape(-1, 1)

    # Keep data with dimension multiple of 128
    length_multiple_128 = 128*int(bar.shape[0]/128)
    accel = accel[0:length_multiple_128, :]
    bar = bar[0:length_multiple_128, :]
    labels = np.array(bar.shape[0]*[int(activity_indices[session[1]])]).reshape(-1, 1)
    timestamps = timestamps[0:length_multiple_128]

    if raw_data_features is None:
      raw_data_features = np.append(accel, bar, axis=1)
      raw_data_labels = labels
      interpolated_timestamps = timestamps
    else:
      raw_data_features = np.append(raw_data_features, np.append(accel, bar, axis=1), axis=0)
      raw_data_labels = np.append(raw_data_labels, labels, axis=0)
      interpolated_timestamps = np.append(interpolated_timestamps, timestamps, axis=0)

  return raw_data_features, raw_data_labels, interpolated_timestamps


def plot_raw_data(raw_data_features, raw_data_labels):
  accel_magnitudes = np.sqrt((raw_data_features[:, 0]**2).reshape(-1, 1)+
                             (raw_data_features[:, 1]**2).reshape(-1, 1)+
                             (raw_data_features[:, 2]**2).reshape(-1, 1))

  plt.subplot(3, 1, 1)
  plt.plot(accel_magnitudes)
  plt.ylabel("ACC Mag")
  
  plt.subplot(3, 1, 2)
  plt.plot(raw_data_features[:, 3])
  plt.ylabel("Barometer")
  
  plt.subplot(3, 1, 3)
  plt.plot(raw_data_labels)
  plt.ylabel("Activity")
  plt.grid(True)

  plt.show()

def feature_extraction(raw_data_features, raw_data_labels, timestamps):
  """
  Args:
    raw_data_features: The fourth column is the barometer data.

  Returns:
    features: Features extracted from the data features, where
              features[:, 0] is the mean magnitude of acceleration;
              features[:, 1] is the variance of acceleration;
              features[:, 2:6] is the fft power spectrum of equally-spaced frequencies;
              features[: 6:12] is the fft power spectrum of frequencies in logarithmic sacle;
              features[:, 13] is the slope of pressure.
  """
  features = None
  labels = None

  accel_magnitudes = np.sqrt((raw_data_features[:, 0]**2).reshape(-1, 1)+
                             (raw_data_features[:, 1]**2).reshape(-1, 1)+
                             (raw_data_features[:, 2]**2).reshape(-1, 1))

  # The window size for feature extraction
  segment_size = 128

  for i in range(0, accel_magnitudes.shape[0]-segment_size, 64):

  # TO DO Compute mean and variance of acceleration for each segment          
     
    segment = accel_magnitudes[i:i+segment_size]
    
    accel_mean = np.mean(segment)
    accel_var = np.std(segment)
    
    segment_fft_powers = np.abs(np.fft.fft(segment))**2
    #print(segment_fft_powers)

    # Aggreate band power within frequency range, with equal space (window size=32) or logarithmic scale
    # Band power of equally-sapced bands: 4 features
    equal_band_power = list()
    window_size = 32
    for j in range(0, len(segment_fft_powers), window_size):
      equal_band_power.append(sum(segment_fft_powers[j: j+32]).tolist()[0])

    # Band power of bands in logarithmic scale: 7 features
    log_band_power = list()
    freqs = [0, 2, 4, 8, 16, 32, 64, 128]
    for j in range(len(freqs)-1):
      log_band_power.append(sum(segment_fft_powers[freqs[j]: freqs[j+1]]).tolist()[0])

    # Slope of barometer data
    # bar_slope = raw_data_features[i+segment_size-1, 3] - raw_data_features[i, 3]
    bar_slope = np.polyfit(timestamps[i:i+segment_size], raw_data_features[i:i+segment_size, 3], 1)[0]
    # bar_slope = np.polyfit([x*0.1 for x in range(segment_size)], raw_data_features[i:i+segment_size, 3], 1)[0]

    feature = [accel_mean, accel_var] + equal_band_power + log_band_power + [bar_slope]

    if features is None:
      features = np.array([feature])
    else:
      features = np.append(features, [feature], axis=0)

    label = Counter(raw_data_labels[i:i+segment_size][:, 0].tolist()).most_common(1)[0][0]

    if labels is None:
      labels = np.array([label])
    else:
      labels = np.append(labels, [label], axis=0)
    
    #print(feature)
  return features, labels


def plot_extracted_features(features, labels):
  # Plot ACC mean and var features
  plt.subplot(2, 1, 1)
  plt.plot(features[:, 0])
  plt.ylabel("ACC mean")

  plt.subplot(2, 1, 2)
  plt.plot(features[:, 1])
  plt.ylabel("ACC Var")

  plt.show()

  #Plot the equally segmented power spectrum power
  plt.subplot(4, 1, 1)
  plt.plot(features[:, 2])
  plt.ylabel("ACC Eq 1")

  plt.subplot(4, 1, 2)
  plt.plot(features[:, 3])
  plt.ylabel("ACC Eq 2")

  plt.subplot(4, 1, 3)
  plt.plot(features[:, 4])
  plt.ylabel("ACC Eq 3")

  plt.subplot(4, 1, 4)
  plt.plot(features[:, 5])
  plt.ylabel("ACC Eq 4")
  plt.show()
  
  
  # Plot the barometer slope
  plt.subplot(2, 1, 1)
  plt.plot(features[:, -1])
  plt.ylabel("Bar Slope")

  plt.subplot(2, 1, 2)
  plt.plot(labels)
  plt.grid(True)
  plt.ylabel("Activity")


  plt.show()


def five_fold_cross_validation(features, labels):
  # decision_stump = DecisionTreeClassifier(max_depth=2)
  # clf = AdaBoostClassifier(base_estimator=decision_stump)
  # clf = AdaBoostClassifier()
  # print(cross_val_score(clf, features, labels, cv=5))

  clf = DecisionTreeClassifier()

  true_labels = list()
  predicted_labels = list()

  for train_index, test_index in StratifiedKFold(n_splits=5).split(features, labels):
    X_train = features[train_index, :]
    Y_train = labels[train_index]

    X_test = features[test_index, :]
    Y_test = labels[test_index]

    clf = DecisionTreeClassifier()
    clf.fit(X_train, Y_train)
    predicted_label = clf.predict(X_test)
    #print(clf.score(X_test, Y_test))

    predicted_labels += predicted_label.flatten().tolist()
    true_labels += Y_test.flatten().tolist()

  # print(predicted_labels)


  # precision tp / (tp + fp)
  #precision = precision_score(true_labels, predicted_labels)
  #print('Precision: %f' % precision)
  # recall: tp / (tp + fn)
  #recall = recall_score(true_labels, predicted_labels)
  #print('Recall: %f' % recall)
  # f1: 2 tp / (2 tp + fp + fn)
  #f1 = f1_score(true_labels, predicted_labels)
  #print('F1 score: %f' % f1)

  confusion_matrix = np.zeros((len(activity_indices), len(activity_indices)))

  for i in range(len(true_labels)):
    confusion_matrix[true_labels[i], predicted_labels[i]] += 1
  
#    per = precision_recall_fscore_support(true_labels[i], predicted_labels[i], average='macro')
 #   reca = precision_recall_fscore_support(true_labels[i], predicted_labels[i], average='micro')
  #  acc = precision_recall_fscore_support(true_labels[i], predicted_labels[i], average='weighted')
  
    
  print("\n===== Confusion Matrix (within subject's) =====")
  print(confusion_matrix)

  for i in range(confusion_matrix.shape[0]):
    # print(sum(confusion_matrix[i, :]))
    #confusion_matrix[i, :] = confusion_matrix[i, :]/sum(confusion_matrix[i, :])
    per = confusion_matrix[i ,i] / (sum(confusion_matrix[:, i]))
    print("precision:", per)
    reca = confusion_matrix[i ,i] / (sum(confusion_matrix[i, :]))
    print("recall:", reca)
    acc = (2 * confusion_matrix[i ,i]) / (sum(confusion_matrix[i, :]) + sum(confusion_matrix[:, i]))
    print("accuracy:", acc)

  print("\n===== Normalized Confusion Matrix (within subject's) =====")
  print(confusion_matrix)
  
  print("\n===== Colormap of Normalized Confusion Matrix (within subject's) =====")
  plt.imshow(confusion_matrix)
  plt.colorbar()
  plt.show()


def evaluate_generalized_model(X_train, Y_train, X_test, Y_test):
  clf = DecisionTreeClassifier().fit(X_train, Y_train)
  #print(clf.score(X_test, Y_test))
  Y_pred = clf.predict(X_test)

  # Plot the true labels and predicted labels
  plt.subplot(2, 1, 1)
  plt.plot(Y_test)
  plt.ylabel("True Activity")

  plt.subplot(2, 1, 2)
  plt.plot(Y_pred)
  plt.ylabel("Predicted Activity")

  plt.show()


  # Plot confusion matrix
  confusion_matrix = np.zeros((len(activity_indices), len(activity_indices)))

  for i in range(len(Y_test)):
    confusion_matrix[int(Y_test[i]), int(Y_pred[i])] += 1
    
  print("\n===== Confusion Matrix (between subject's) =====")
  print(confusion_matrix)

  #for i in range(confusion_matrix.shape[0]):
    # print(sum(confusion_matrix[i, :]))
    #confusion_matrix[i, :] = confusion_matrix[i, :]/sum(confusion_matrix[i, :])

    
  print("\n===== Normalized Confusion Matrix (between subject's) =====")
  print(confusion_matrix)

  print("\n===== Colormap of Normalized Confusion Matrix (between subject's) =====")
  plt.imshow(confusion_matrix)
  plt.colorbar()
  # plt.show()

  # Print the top-5 features using recursive feature selection
  # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
  selector = RFE(clf, 5)
  selector.fit(X_train, Y_train)
  print("\n===== Mask of Top-5 Features =====")
  print(selector.support_)


if __name__ == "__main__":

  data_path = './back_up_data/'
  # your user id
  netid = 'mlc299'

  raw_data_features, raw_data_labels, timestamps = compute_raw_data(data_path + netid)

  # STEP1. Plot the raw data to get a sense about what features might work
  # You can comment out this line of code if you don't want to see the plots
  plot_raw_data(raw_data_features, raw_data_labels)

  # STEP2. extract features
  features, labels = feature_extraction(raw_data_features, raw_data_labels, timestamps);

  # STEP3. Plot the top features
  plot_extracted_features(features, labels)

  # STEP4. Personal-dependent Model
  five_fold_cross_validation(features, labels)


  # Generalized model (i.e. train on other's data and test on your own data)
  X_train = None
  Y_train = None
  X_test = None
  Y_test = None

  dirs = os.listdir(data_path)
  print("loading other people's data....")
  for dir in dirs:
    print(dir)
    if dir[0] == '.':
      continue
    raw_data_features, raw_data_labels, timestamps = compute_raw_data(data_path + dir);
    features, labels = feature_extraction(raw_data_features, raw_data_labels, timestamps);

    if dir == netid:
      X_test = features
      Y_test = labels

    else:
      if X_train is None:
        X_train = features
        Y_train = labels
      else:
        X_train = np.append(X_train, features, axis=0)
        Y_train = np.append(Y_train, labels, axis=0)

  #print(X_train)
  #print(X_test)

  evaluate_generalized_model(X_train, Y_train, X_test, Y_test)

