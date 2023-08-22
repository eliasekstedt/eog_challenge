from skopt import gp_minimize
from skopt.space import Real
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d




def model(wd):
    def get_map():
        # defining the map
        x = np.linspace(0, 4, 10)
        y = np.array([6, 5, 3, 1, 1, 3, 4, 4, 3, 3])
        f = interp1d(x, y, kind='cubic')
        return f

    f = get_map()
    return f(wd)

def objective(params):
    weight_decay = params
    performance = model(weight_decay)
    # We want to maximize performance, so we return its negative
    print(performance)
    return float(performance)

# Define hyperparameter space
space = [Real(0, 4, name='wd')]

# Run Bayesian optimization
result = gp_minimize(objective, space, n_calls=7, acq_func='EI', n_random_starts=5)

# Best hyperparameters
best_weight_decay = result.x[0]
print(best_weight_decay)

# plotting
x = np.linspace(0, 4, 10)
y = np.array([6, 5, 3, 1, 1, 3, 4, 4, 3, 3])
f = interp1d(x, y, kind='cubic')

plt.plot(np.arange(0, 4, 1/100), f(np.arange(0, 4, 1/100)))
plt.scatter(best_weight_decay, f(best_weight_decay))
plt.show()
