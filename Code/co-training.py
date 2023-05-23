import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, learning_curve
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# Load your data
data = pd.read_csv('/content/drive/MyDrive/Training Data/default_train_data.csv')

y = data['transportation_mode']
data.drop(columns=['new_id', 'transportation_mode', 'points_inside_buffer_bus_50', 'total_gps_per_trip', 'points_inside_buffer_station_50'], inplace=True)
X = data

test_data = pd.read_csv('/content/drive/MyDrive/Testing Data/default_test_data.csv')

y_test = test_data['transportation_mode']
test_data.drop(columns=['new_id', 'transportation_mode', 'points_inside_buffer_bus_50', 'total_gps_per_trip', 'points_inside_buffer_station_50'], inplace=True)
x_test = test_data

X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X, y, test_size=0.1, stratify=y)

# Split labeled data based on column names
X1_labeled = X_labeled.loc[:, ['distance_m', 'mean_speed_ms', 'max_speed_ms', 'min_speed_ms',
       'mean_accs', 'std_speed', 'std_accs', '95_speed', '75_speed', '50_speed']]
X2_labeled = X_labeled.loc[:, ['trip_duration',
       'trip_start_hour', 'buffer_ratio_bus_50', 'buffer_ratio_station_50',
       'origin_bus_density', 'destination_bus_density',
       'origin_station_density', 'destination_station_density', 'rain',
       'temp', 'age', 'gender', 'occupation',  'speed_ms_inside_buffer_bus_50',
                    'speed_ms_outside_buffer_bus_50', 'speed_ms_inside_buffer_station_50',
                    'speed_ms_outside_buffer_station_50']]

rf1 = KNeighborsClassifier(n_neighbors=5)
rf2 = RandomForestClassifier(n_estimators=100, random_state=42)

rf1.fit(X1_labeled, y_labeled)
rf2.fit(X2_labeled, y_labeled)

# Iterate until convergence or max iterations
max_iterations = 10
k = 10
for i in range(max_iterations):
    # Use the classifiers to predict labels for the unlabeled data
    pred_1 = rf1.predict(X_unlabeled.loc[:, ['distance_m', 'mean_speed_ms', 'max_speed_ms', 'min_speed_ms',
       'mean_accs', 'std_speed', 'std_accs', '95_speed', '75_speed', '50_speed']])
    pred_2 = rf2.predict(X_unlabeled.loc[:, ['trip_duration',
       'trip_start_hour', 'buffer_ratio_bus_50', 'buffer_ratio_station_50',
       'origin_bus_density', 'destination_bus_density',
       'origin_station_density', 'destination_station_density', 'rain',
       'temp', 'age', 'gender', 'occupation',  'speed_ms_inside_buffer_bus_50',
                    'speed_ms_outside_buffer_bus_50', 'speed_ms_inside_buffer_station_50',
                    'speed_ms_outside_buffer_station_50']])
    pred = np.column_stack((pred_1, pred_2))



    # Add the most confident predictions to the labeled data
    confidence = np.max(rf1.predict_proba(X_unlabeled.loc[:, ['distance_m', 'mean_speed_ms', 'max_speed_ms', 'min_speed_ms',
       'mean_accs', 'std_speed', 'std_accs', '95_speed', '75_speed', '50_speed']]), axis=1) * np.max(rf2.predict_proba(X_unlabeled.loc[:, ['trip_duration',
       'trip_start_hour', 'buffer_ratio_bus_50', 'buffer_ratio_station_50',
       'origin_bus_density', 'destination_bus_density',
       'origin_station_density', 'destination_station_density', 'rain',
       'temp', 'age', 'gender', 'occupation',  'speed_ms_inside_buffer_bus_50',
                    'speed_ms_outside_buffer_bus_50', 'speed_ms_inside_buffer_station_50',
                    'speed_ms_outside_buffer_station_50']]), axis=1)
    indices = np.argsort(confidence)[::-1][:k]
    max_index = len(X_labeled) - 1
    indices = [i for i in indices if i <= max_index]
    labeled_data = X_labeled.append(X_labeled.iloc[indices, :])
    unlabeled_data = X_unlabeled.drop(X_unlabeled.index[indices])

    # Retrain the classifiers on the updated labeled data
    X1_labeled = X_labeled.loc[:, ['distance_m', 'mean_speed_ms', 'max_speed_ms', 'min_speed_ms',
       'mean_accs', 'std_speed', 'std_accs', '95_speed', '75_speed', '50_speed']]
    X2_labeled = X_labeled.loc[:, ['trip_duration',
       'trip_start_hour', 'buffer_ratio_bus_50', 'buffer_ratio_station_50',
       'origin_bus_density', 'destination_bus_density',
       'origin_station_density', 'destination_station_density', 'rain',
       'temp', 'age', 'gender', 'occupation',  'speed_ms_inside_buffer_bus_50',
                    'speed_ms_outside_buffer_bus_50', 'speed_ms_inside_buffer_station_50',
                    'speed_ms_outside_buffer_station_50']]
    rf1.fit(X1_labeled, y_labeled)
    rf2.fit(X2_labeled, y_labeled)

test_view_1 = test_data.loc[:, ['distance_m', 'mean_speed_ms', 'max_speed_ms', 'min_speed_ms',
       'mean_accs', 'std_speed', 'std_accs', '95_speed', '75_speed', '50_speed']]
test_view_2 = test_data.loc[:,['trip_duration',
       'trip_start_hour', 'buffer_ratio_bus_50', 'buffer_ratio_station_50',
       'origin_bus_density', 'destination_bus_density',
       'origin_station_density', 'destination_station_density', 'rain',
       'temp', 'age', 'gender', 'occupation',  'speed_ms_inside_buffer_bus_50',
                    'speed_ms_outside_buffer_bus_50', 'speed_ms_inside_buffer_station_50',
                    'speed_ms_outside_buffer_station_50']]

pred_test_1 = rf1.predict(test_view_1)
pred_test_2 = rf2.predict(test_view_2)
pred_test = np.column_stack((pred_test_1, pred_test_2))

# pred_final = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)).astype(int), axis=1, arr=pred_test)
pred_final = np.apply_along_axis(lambda x: np.argmax(np.bincount(x.astype(int))), axis=1, arr=pred_test)

accuracy = accuracy_score(y_test, pred_final)
print('Test accuracy:', accuracy)
