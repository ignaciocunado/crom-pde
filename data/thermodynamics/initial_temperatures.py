import numpy as np

def initial_temp_sinusoidal(x):
    return np.sin(10 * np.pi * x)

def initial_temp_gaussian(x):
    A = 1.0
    x0 = 0.5
    sigma = 0.1
    return A * np.exp(-((x - x0)**2) / (2 * sigma**2))

def initial_temp_piecewise_linear(x):
    return np.where(x < 0.5, 2 * x, 2 * (1 - x))

def initial_temp_combined_sinusoids(x):
    return np.sin(10 * np.pi * x) + 0.5 * np.sin(3 * np.pi * x)

def initial_temp_polynomial(x):
    return x * (1 - x)

def initial_temp_sine_gaussian(x):
    A = 1.0
    x0 = 0.5
    sigma = 0.1
    k = 4
    return A * np.exp(-((x - x0)**2) / (2 * sigma**2)) * np.sin(k * np.pi * x)