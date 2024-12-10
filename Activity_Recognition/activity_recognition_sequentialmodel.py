import os
#uncomment this line and skip gpu if running issues with model training in gpu 
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import numpy as np
import random
import statsmodels.api as sm
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
sns.set(font_scale=2,style='whitegrid')
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categories='auto')
from sklearn.metrics import accuracy_score
#set random seed across different library/components
seed_val = 42
np.random.seed(seed=seed_val)
try:
    tf.random.set_seed(seed_val)
except:
    tf.set_random_seed(seed_val)
    
os.environ['PYTHONHASHSEED']=str(seed_val)
random.seed(seed_val)

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


standard_scaler = StandardScaler()
minmax_scaler   = MinMaxScaler()

def compute_raw_data(dir_name):
  '''
  Given a directory location, this function returns the raw data and activity labels
  for data in that directory location
  '''
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
    #timestamps in the file are in milliseconds
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

def feature_extraction_lstm_raw(raw_data_features, raw_data_labels, timestamps):
  """
  
  Takes in the raw data and labels information and returns data formatted
  for processing in a lstm model (batch, timesteps, features) format
  
  Args:
    raw_data_features: raw data returns from the directory The fourth column is the barometer data.
    raw_data_labels: labels associated with a data row
    timestamps: timestamp of the given row of observation

  Returns:
    features_np: features (re-arrange raw data or other derived observation) according to lstm format
    labels_np: labels associated with each of the features
  """
  features = []
  labels   = []

  #accel_magnitudes = butter_lowpass_filter(accel_magnitudes,4,32)
  accel_magnitudes = np.sqrt((raw_data_features[:, 0]**2).reshape(-1, 1)+
                             (raw_data_features[:, 1]**2).reshape(-1, 1)+
                             (raw_data_features[:, 2]**2).reshape(-1, 1))

  # The window size for feature extraction
  segment_size = 128

  for i in range(0, raw_data_features.shape[0]-segment_size, 64):
    segment    = raw_data_features[i:i+segment_size]
    seg_accmag = accel_magnitudes[i:i+segment_size]
    #compute difference of successive barometer values
    seg_bardiff    = np.append([0],np.diff(segment[:,3]))[:,None]

    #add acc magnitude and barometer difference feature to segment
    segment    = np.hstack([segment,seg_accmag])
    segment    = np.hstack([segment,seg_bardiff])
    
    features.append(segment[:,:])
    label = Counter(raw_data_labels[i:i+segment_size][:, 0].tolist()).most_common(1)[0][0]
    labels.append(label)
  
  #re-arrange feature to (batch, timestep,features format)
  features_np = np.einsum('ijk->kij',np.dstack(features))    
  labels_np   = np.array(labels)
  return features_np, labels_np

def return_lstm_model(num_input_feat,num_classes):
    '''
    Returns a lstm model architecture

    Parameters
    ----------
    num_input_feat : integer
        Number of input features in the model
    num_classes : integer
        Number of output labels to be predicted

    Returns
    -------
    model : tf model
        
    '''
    _input   = tf.keras.layers.Input(shape=(None, num_input_feat))
    lstm1    = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128))(_input)
    dropout1 = tf.keras.layers.Dropout(rate=0.4)(lstm1)
    fc1      = tf.keras.layers.Dense(units=128,activation='relu')(dropout1)
    dropout2 = tf.keras.layers.Dropout(rate=0.4)(fc1)
    fc2      = tf.keras.layers.Dense(units=64,activation='relu')(dropout2)
    out      = tf.keras.layers.Dense(units=num_classes,activation='softmax')(fc2)
    model    = tf.keras.Model(outputs=out, inputs=_input)
    return model
    
if __name__ == "__main__":

  #data_path = './A2/back_up_data/'
  data_path = 'back_up_data/'
  # your user id
  netid = 'mlc299'

  raw_data_features, raw_data_labels, timestamps = compute_raw_data(data_path + netid)

  # Generalized model (i.e. train on other's data and test on your own data)
  X_train = None
  Y_train = None
  X_test = None
  Y_test = None

  X_train_lstm = []
  X_test_lstm  = [] 
  Y_train_lstm = []
  Y_test_lstm  = []

  dirs = os.listdir(data_path)
  print("loading other people's data....")
  for dir in dirs:
    print(dir)
    if dir[0] == '.':
      continue
    #obtain raw data and arrange data/features according to format required for lstm model
    raw_data_features, raw_data_labels, timestamps = compute_raw_data(data_path + dir);
    features_lstm, labels_lstm                     = feature_extraction_lstm_raw(raw_data_features, 
                                                                                 raw_data_labels,
                                                                                 timestamps)    
    if dir == netid:      
      X_test_lstm.append(features_lstm)
      Y_test_lstm.append(labels_lstm)
    else:
      X_train_lstm.append(features_lstm)
      Y_train_lstm.append(labels_lstm)

  X_test_lstm_np  = np.dstack(X_test_lstm)
  Y_test_lstm_np  = np.array(Y_test_lstm).squeeze()
  X_train_lstm_np = np.concatenate(X_train_lstm,axis=0)
  Y_train_lstm_np = np.concatenate(Y_train_lstm).squeeze()
  
  #normalize features
  scaler_fit        = StandardScaler().fit(X_train_lstm_np.reshape(-1,X_train_lstm_np.shape[-1]))
  X_train_lstm_np_  = scaler_fit.transform(X_train_lstm_np.reshape(-1,X_train_lstm_np.shape[-1])).reshape(X_train_lstm_np.shape)
  X_test_lstm_np_   = scaler_fit.transform(X_test_lstm_np.reshape(-1,X_test_lstm_np.shape[-1])).reshape(X_test_lstm_np.shape)
  
   
  #obtain a lstm model (re-define architecture in the function to try different architectures)
  lstm_model = return_lstm_model(X_train_lstm_np_.shape[-1],
                                 len(np.unique(Y_train_lstm_np)))
  #set learning rate for the model (hyperparamter not tuned yet)
  lr_val      = 0.001
  #let's use ADAM optimizer for training our model with the specified learning rate.. see other parameters from function's docstring
  adam_opt   = tf.keras.optimizers.Adam(lr=lr_val) 
  #depending upon the system configuration/library version you might need to provide metrics=['accuracy'] here
  lstm_model.compile(loss='categorical_crossentropy', optimizer=adam_opt,metrics=['accuracy'])
  #one hot encoding of labels
  Y_train_lstm_np_ohe = ohe.fit_transform(Y_train_lstm_np[:,None]).toarray()
  #model training
  #obtain class weight inversely proportional to respective sizes in the trianing set
  class_weight = dict(zip(np.arange(Y_train_lstm_np_ohe.shape[1]),
                          1/Y_train_lstm_np_ohe.sum(axis=0)))
  lstm_model.fit(X_train_lstm_np_,Y_train_lstm_np_ohe,epochs=100,batch_size=64,
                 class_weight=class_weight)
  
  #predict the labels on the test set
  pred_label    = lstm_model.predict(X_test_lstm_np_)
  #inverse transform from one hot encoding to ordinal/categorical encoding
  pred_label_np = ohe.inverse_transform(pred_label)
  #evaluate accuracy scores
  lstm_accu     = accuracy_score(Y_test_lstm_np,pred_label_np)
  print('LSTM accu is: {0}'.format(lstm_accu))
  
  target_names = pd.Series(activity_indices).to_frame().sort_values(by=0).index.values
  generalizedmodel_classification_report= classification_report(Y_test_lstm_np,
                                                             pred_label_np,
                                                             target_names=target_names)
  