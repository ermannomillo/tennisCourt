#-------------------------------------------------------------------------------
    Python AI API for predicting number of bookings in next two weeks
#-------------------------------------------------------------------------------


from flask import Flask
from flask import jsonify
import json, mysql.connector
import pandas as pd
import numpy
from numpy import newaxis
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import datetime
from subprocess import check_output
import time


app = Flask(__name__)

def getBookingDictionary():
    dbconnection = mysql.connector.connect(
      host="localhost",
      user="root",
      passwd="",
      database="tennis"
    )
    dbcursor = dbconnection.cursor()
    query = "SELECT date FROM bookings"
    dbcursor.execute(query)
    rawData = dbcursor.fetchall()
    bookings = []
    for booking in set(rawData):
        count = rawData.count(booking)
        bookings.append([str(booking)[2:-3], count])
    return bookings

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

def predict_sequences_multiple(model, firstValue,length):
    prediction_seqs = []
    curr_frame = firstValue

    for i in range(length):
        predicted = []
        print(model.predict(curr_frame[newaxis,:,:]))
        predicted.append(int(model.predict(curr_frame[newaxis,:,:])[0,0]))
        curr_frame = curr_frame[0:]
        curr_frame = numpy.insert(curr_frame[0:], i+1, predicted[-1], axis=0)
        prediction_seqs.append(int(predicted[-1]))

    return prediction_seqs

def trainLSTM():
    bookings = getBookingDictionary()
    numpy.random.seed(7)
    dataframe = pd.DataFrame(bookings, columns=['date', 'count'])
    dataframe['date'] = pd.to_datetime(dataframe['date'], format='%d/%m/%Y')
    dataframe = dataframe.sort_values('date')
    sparseDataframe = dataframe
    maxDate = max(dataframe['date'])
    minDate = min(dataframe['date'])
    dataframe.set_index('date', inplace=True)
    print(dataframe)
    index = pd.date_range(minDate, maxDate)
    #dataframe.index = pd.DatetimeIndex(dataframe.index)
    dataframe = dataframe.reindex(index, fill_value=0)
    dataframe = dataframe.reset_index()

    dataset = dataframe['count'].values
    print(dataset)
    dataset = dataset.reshape(len(dataset), 1)
    print(dataset.shape)
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    train_size = int(len(dataset) * 0.80)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    #Step 2 Build Model
    model = Sequential()

    model.add(LSTM(
        input_dim=1,
        output_dim=50,
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        100,
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=1))
    model.add(Activation('linear'))

    start = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    print ('compilation time : ', time.time() - start)

    model.fit(
    trainX,
    trainY,
    batch_size=128,
    nb_epoch=100,
    validation_split=0.05)

    predict_length=14
    predictions = predict_sequences_multiple(model, testX[0], predict_length)
    print(scaler.inverse_transform(numpy.array(predictions).reshape(-1, 1)))

    predictionsScaled = (scaler.inverse_transform(numpy.array(predictions).reshape(-1, 1))).astype(int)
    return predictionsScaled

@app.route("/test")
def hello():
    return "<h1 style='font-family:Segoe UI'>Hello to Tennis Court price API</h1>"

@app.route('/api/getall', methods=['GET'])
def getAll():
    bookings = getBookingDictionary()
    return json.dumps(bookings)

@app.route('/api/getfuture', methods=['GET'])
def getNextDay():
    predictions = trainLSTM()
    print(predictions)
    noBookings = "["
    for i in range(0, 14):
        noBookings = noBookings + str(predictions.item(i)) + ', '
    noBookings = noBookings[:-2] +  ']'
    return jsonify(noBookings)


if __name__ == "__main__":
    app.run()
