import pandas as pd
import numpy as np
from numpy import array
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
%matplotlib inline

df=pd.read_csv('/content/AAPL.csv')
df.head()

df.shape

df1=df.reset_index()['Close'];df1

caler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1));df1

training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
print(training_size,test_size)

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

print(X_train.shape), print(y_train.shape),print(X_test.shape), print(ytest.shape)

X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
     
print(X_train.shape), print(X_test.shape)

model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)))  # 100 hidden units in the first layer
model.add(LSTM(100, return_sequences=True))  # 100 hidden units in the second layer
model.add(LSTM(100))  # 100 hidden units in the third layer
model.add(Dense(1))  # Single output unit
model.compile(loss='mse', optimizer='adam')

from keras.callbacks import EarlyStopping

from keras.callbacks import EarlyStopping

# ... (load and preprocess data as before)
# ... (define your LSTM model as before)

# Add EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=1024,  # Or try smaller sizes like 32 or 16 if needed
    validation_data=(X_test, ytest),
    callbacks=[early_stopping],
    verbose=1
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot();

train_predict=model.predict(X_train) #prediction
test_predict=model.predict(X_test)

from sklearn.metrics import mean_squared_error
print(math.sqrt(mean_squared_error(y_train,train_predict)))
print(math.sqrt(mean_squared_error(ytest,test_predict)))

look_back=100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.legend()
plt.show()
     
x_input=test_data[483:].reshape(1,-1) #len(test_data)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

# Invert the predictions
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculate RMSE
from sklearn.metrics import mean_squared_error
print(f"Train RMSE: {np.sqrt(mean_squared_error(scaler.inverse_transform(y_train.reshape(-1, 1)), train_predict))}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(scaler.inverse_transform(ytest.reshape(-1, 1)), test_predict))}")

# Shift train predictions for plotting
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back : len(train_predict) + look_back, :] = train_predict

# Shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[
    len(train_predict) + (look_back * 2) + 1 : len(df1) - 1, :
] = test_predict

# Plot baseline and predictions
plt.plot(scaler.inverse_transform(df1), label="Actual")
plt.plot(trainPredictPlot, label="Train Prediction")
plt.plot(testPredictPlot, label="Test Prediction")
plt.legend()
plt.show()

day_new=np.arange(1,101)
day_pred=np.arange(101,131)

plt.plot(day_new, scaler.inverse_transform(df1[1564:1664]))

day_new = np.arange(1, len(df1[1564:]) + 1)

df3=df1.tolist()
df3.extend(lst_output)
df3=scaler.inverse_transform(df3).tolist()
plt.plot(df3[1600:])

