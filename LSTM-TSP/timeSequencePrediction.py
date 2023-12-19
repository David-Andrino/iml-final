import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors

# Parameters
training_periods = 5
training_samples = 40
predict_samples = 500
noise_amplitude = 1
noise_period = 5
n_neighbors = 5
weight_mode = "distance" # Uniform/distance

rng = np.random.default_rng()
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

plt.figure()
plt.plot(T, np.sin(T), '--', color='gray', label='Ideal Sin')
plt.plot(T, y_predict_uniform, color='blue', label='Uniform KNN')
plt.plot(T, y_predict_distance, color='red', label='Distance KNN')
plt.legend()
plt.title("Nearest Neighbor Regression")
plt.xlim((0, 2*np.pi))
plt.show()
