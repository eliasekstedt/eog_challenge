
from skopt import gp_minimize
from skopt.space import Real
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d




def model(wd):
    def get_map():
        # defining the map
        a = np.linspace(0, 4, 10)
        y = np.array([6, 5, 3, 1, 1, 3, 4, 4, 3, 3])
        f = interp1d(a, y, kind='cubic')
        return f

    f = get_map()
    return f(wd)

def objective(params):
    print('* *')
    weight_decay = params
    performance = model(weight_decay)
    return float(performance)

def noise_objective(params):
    #print(np.random.normal(0, 2))
    weight_decay = params
    performance = model(weight_decay) + np.random.normal(0, 2)
    return float(performance)


# Define hyperparameter space
space = [Real(0, 4, name='wd')]

# Run Bayesian optimization
result = gp_minimize(objective, space, n_calls=10, acq_func='EI', n_random_starts=5)
noise_result = gp_minimize(noise_objective, space, n_calls=10, acq_func='EI', n_random_starts=5)

# Best hyperparameters
print(f'***: {result}')
1/0
best_param = result.x[0]
noise_best_param = noise_result.x[0]
print(f'      best_param: {best_param}')
print(f'noise_best_param: {noise_best_param}')



# plotting
a = np.linspace(0, 4, 10)
y = np.array([6, 5, 3, 1, 1, 3, 4, 4, 3, 3])
f = interp1d(a, y, kind='cubic')

plt.plot(np.arange(0, 4, 1/100), f(np.arange(0, 4, 1/100)))
plt.scatter(best_param, f(best_param))
plt.scatter(noise_best_param, f(noise_best_param))
plt.show()
