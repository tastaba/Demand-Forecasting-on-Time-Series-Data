import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df_train = pd.read_csv("train.csv", parse_dates=['date'])
df = df_train.copy()
df_test = pd.read_csv("test.csv", parse_dates=['date'])

def create_dataset(X, y, time_steps=1,pred_steps=0):
    Xs, ys = [], []
    for i in range(len(X) - time_steps-pred_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps+pred_steps])
    return np.array(Xs), np.array(ys)


########### Create the time steps ##############
def create_predset(X, y, time_steps = 1):
    Xs = []
    for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)
    return np.array(Xs)

time_steps = 90
pred_step = 90

df_train['day'] = df_train['date'].dt.day
df_train['month'] = df_train['date'].dt.month
df_train['year'] = df_train['date'].dt.year
df_train['dayofweek']=df_train['date'].dt.dayofweek

df_sub_test = pd.DataFrame(columns=df_train.columns)
for index, row in df_test.iterrows():
    # find test set start date and corresponding training date for predicting submission test set
    pred_test_start_date = row['date'] - pd.to_timedelta(time_steps+pred_step, unit='d')
    #print(pred_test_start_date)
    df_pred_test_start = df_train[(df_train['date']==pred_test_start_date) & (df_train['store']==row['store']) & (df_train['item']==row['item'])]
    #print(df_pred_test_start)
    ind = df_pred_test_start.index.values[0]
    #print(ind)
    df_sub_test = df_sub_test.append(df_pred_test_start)

print(len(df_train.iloc[ind:(ind+time_steps+pred_step),]))

# Append time_steps + pred_step data for last sequence
df_sub_test= df_sub_test.append(df_train.iloc[ind:(ind+time_steps+pred_step-1),])

# take all rows from that index to the end of training set
df_sub_test.reset_index()
print(len(df_sub_test))
# print(df_sub_test)

# splitting dataset
train_size = int(len(df_train) * 0.9)
train, test = df_train.iloc[0:train_size], df_train.iloc[train_size:len(df_train)]

# Defining features
label_column = ['sales']
feature_columns = ['store', 'item', 'day', 'month', 'year', 'dayofweek']
all_columns = ['store', 'item', 'day', 'month', 'year', 'dayofweek', 'sales']
label_scalar = MinMaxScaler(feature_range=(0, 1))
feature_scalar = MinMaxScaler(feature_range=(0, 1))

# Normalizing Features for training set
train[label_column] = label_scalar.fit_transform(train[label_column])
train[feature_columns] = feature_scalar.fit_transform(train[feature_columns])

# Normalizing Features for test set
df_sub_test[label_column] = label_scalar.fit_transform(df_sub_test[label_column])
df_sub_test[feature_columns] = feature_scalar.fit_transform(df_sub_test[feature_columns])

# Normalizing Features for test+pred(df_sub_test set)
test[label_column] = label_scalar.fit_transform(test[label_column])
test[feature_columns] = feature_scalar.fit_transform(test[feature_columns])

print(len(train))
print(len(test))
print(len(df_sub_test))
# print('train: ',train.tail())
# print('test: ',test.tail())
# print('df_sub_test-tail: ',df_sub_test.tail())
# print('df_sub_test-head: ',df_sub_test.head())

# Creating input sequence for input to the LSTM - we'll use 4 weeks of data to predict the 5th week
X_train,y_train = create_dataset(train[all_columns],train[label_column],time_steps,pred_step)
X_test, y_test = create_dataset(test[all_columns], test[label_column], time_steps,pred_step)
X_pred_test = create_predset(df_sub_test[all_columns], df_sub_test[label_column], time_steps)
print('train shape: ',X_train.shape, y_train.shape)
print('test shape: ',X_test.shape, y_test.shape)
print('sub_test shape: ',X_pred_test.shape)

def custom_smape(x, x_): # From the Public Kernel https://www.kaggle.com/rezas26/simple-keras-starter
    return keras.backend.mean(2*keras.backend.abs(x-x_)/(keras.backend.abs(x)+keras.backend.abs(x_)))
def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


########## Define  LSTM model #############
model = keras.Sequential()
model.add(keras.layers.LSTM(200, input_shape=(time_steps, X_train.shape[2]), activation='relu'))
model.add(keras.layers.Dropout(rate=0.1))
model.add(keras.layers.Dense(units=1))
model.compile(loss=custom_smape, optimizer='adam')
model.summary()
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=15, batch_size=90, shuffle = False, callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

# predictions
y_pred = model.predict(X_test)
X_test_pred = model.predict(X_pred_test)

# inverse transform
y_pred = label_scalar.inverse_transform(y_pred)
inverse_y_test = label_scalar.inverse_transform(y_test)
X_test_pred = label_scalar.inverse_transform(X_test_pred)

print('Actual Values:', inverse_y_test)
print('Predicted Values:',y_pred)
print('Submission prediction: ', X_test_pred)

# calculate SMAPE
SMAPE = smape(y_pred, inverse_y_test)
print('validation SMAPE:', SMAPE)


# write to submission file
PATH = "C:/Users/Tassi/PycharmProjects/BigDataAnalytics/demand-forecasting-data/"
sub_df = pd.DataFrame(np.array(X_test_pred).flatten(), columns=['sales'])
sub_df.to_csv(PATH + 'sample_submission.csv', header=False)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# plot prediction
# plt.plot(np.arange(0, len(y_train)), train_y.flatten(), 'g', label="history")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), inverse_y_test.flatten(), marker='.', label="true")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_pred, 'r', label="prediction")
plt.ylabel('Sales')
plt.xlabel('Time Step')
plt.legend()
plt.show()


