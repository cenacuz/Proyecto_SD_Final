#Importación de las librerias para poder realizar los calculos
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
%matplotlib inline
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# fix random seed for reproducibility
numpy.random.seed(7)
# Lectura de los datos
dataset = pd.read_csv('DB1.csv', index_col='time', parse_dates=['time'])
#Mostrando la serie de tiempo de los datos
dataset.head()
plt.figure(figsize=(18,6))
plt.plot(dataset,label='Real Data')
plt.xlabel('t Time[ms]')
plt.ylabel('CPU [usage]')
plt.title("Time series CPU")
plt.legend()
plt.show()
#viendo los percentiles del proyecto
dataset.describe()
# realizando un cuadro estadistico de la muestra
sns.boxplot(dataset)
                                #Algoritmo LSTM
# normalize the dataset
datasetY = dataset.index
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# reshape into X=t and Y=t+1
look_back = 10
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1],1))
testX = numpy.reshape(testX, (testX.shape[0],testX.shape[1],1))
print(trainX.shape,testX.shape)
# create and fit the LSTM network
dim_entrada = (trainX.shape[1],1)
dim_salida = 1
neuronas_bloque =50
model = Sequential()
model.add(LSTM(neuronas_bloque, input_shape=dim_entrada))
model.add(Dense(dim_salida))
model.compile(loss='mean_squared_error', optimizer='rmsprop')
model.fit(trainX, trainY, epochs=1000, batch_size=13719, verbose=2)
model.save('Train_V2')
#model=load_model('Train_V2') 
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# Puntaje de Varianza. El mejor puntaje es un 1.0
print('Variance score of Train: %.2f' % r2_score(trainY[0], trainPredict[:,0]))
print('Variance score of Test: %.2f' % r2_score(testY[0], testPredict[:,0]))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.figure(figsize=(18,6))
plt.plot(datasetY,scaler.inverse_transform(dataset),label='Real Data',color='blue')
plt.plot(datasetY,trainPredictPlot,label='Entrenamiento')
plt.plot(datasetY,testPredictPlot,label='Prediccion', color='green')

plt.xlabel('t Time[ms]')
plt.ylabel('CPU [MHz]')

plt.title("Time series CPU")

plt.legend()
plt.show()
