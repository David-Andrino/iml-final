from cProfile import label
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
from lstm_load import Pred
# from lstm import LSTM

# Parameters
training_periods = 5
training_samples = 40
predict_samples = 500
noise_amplitude = 0
noise_period = 1
n_neighbors = 5

def generateNoisySin():
    # x = np.sort(2 * training_periods * np.pi * rng.random((training_samples*training_periods, 1)), axis=0)
    x = np.linspace((training_periods - 1) * 2 * np.pi, training_periods*2*np.pi, predict_samples)[:, np.newaxis]
    y = np.sin(x).ravel()
    # y[::noise_period] += noise_amplitude * (0.5 - rng.random(int(training_samples*training_periods/noise_period)))
    return (x, y)

rng = np.random.default_rng()
# Random spacing
x = np.sort(2 * training_periods * np.pi * rng.random((training_samples*training_periods, 1)), axis=0)
y = np.sin(x).ravel()
T = np.linspace((training_periods - 1) * 2 * np.pi, training_periods*2*np.pi, predict_samples)[:, np.newaxis]
y[::noise_period] += noise_amplitude * (0.5 - rng.random(int(training_samples*training_periods/noise_period)))

# Uniform knn
knn_uniform = neighbors.KNeighborsRegressor(n_neighbors, weights="uniform")
model_uniform = knn_uniform.fit(x, y)
y_predict_uniform = model_uniform.predict(T)

# Distance
knn_distance = neighbors.KNeighborsRegressor(n_neighbors, weights="distance")
model_distance = knn_distance.fit(x, y)
y_predict_distance = model_distance.predict(T)

T -= 2 * np.pi * (training_periods - 1)
x -= 2 * np.pi * (training_periods - 1)

# model = LSTM().build()
# model.train_model(generateNoisySin)
# y_lstm = model.predict(T).detach().numpy()

signal = np.empty((100, 1000), 'int64')
signal[:] = np.array(range(1000))

noisy = np.sin(signal / 1.0 / (2 * np.pi)).astype('float64') + np.random.uniform(-noise_amplitude/2,noise_amplitude/2)
clean = np.sin(signal / 1.0 / 20).astype('float64')

y_pred_lstm = Pred.predict(noisy, 1000, clean)

y_pred_lstm = y_pred_lstm[0].tolist()[-len(T):]

plt.figure()
plt.plot(T, np.sin(T), '--', color='gray', label='Ideal Sin')
# plt.plot(T, y_predict_uniform, color='blue', label='Uniform KNN')
# plt.plot(T, y_predict_distance, color='red', label='Distance KNN')
# plt.plot(T, y_lstm, color='pink', label="LSTM")
plt.plot(T, y_pred_lstm, color='pink', label="LSTM")
plt.legend()
plt.title("Nearest Neighbor Regression")
plt.xlim((0, 2*np.pi))
plt.show()

