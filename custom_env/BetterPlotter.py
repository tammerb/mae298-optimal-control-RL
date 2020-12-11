import numpy as np
import matplotlib.pyplot as plt

class BetterPlotter():
    def __init__(self, bag_path):

        default_data = np.genfromtxt(bag_path + 'default/monitor.csv', delimiter=',', skip_header=1, names=True, comments='#')
        optuna_data = np.genfromtxt(bag_path + 'optuna/monitor.csv', delimiter=',', skip_header=1, names=True, comments='#')

        def_len = len(default_data)
        opt_len = len(optuna_data)
        def_list = default_data['l']
        opt_list = optuna_data['l']
        default_ts = [sum(def_list[0:x:1]) for x in range(0, def_len+1)]
        default_ts = default_ts[1:]
        optuna_ts = [sum(opt_list[0:x:1]) for x in range(0, opt_len+1)]
        optuna_ts = optuna_ts[1:]
        # Plot the data
        plt.scatter(default_ts, default_data['r'])
        plt.scatter(optuna_ts, optuna_data['r'])
        plt.xlabel('Timestep')
        plt.ylabel('Reward')
        plt.show()