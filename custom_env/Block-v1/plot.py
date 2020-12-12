ENV_ID='Block-v1'
bag_dir = 'bag/'
TOTAL_TIMESTEPS = 5e6

legend_names = []
legend_names.append('Default')
legend_names.append('Optuna')

results = []
results.append(bag_dir + "default/")
results.append(bag_dir + "optuna/")
from stable_baselines3.common import results_plotter
import matplotlib.pyplot as plt

results_plotter.plot_results(results, TOTAL_TIMESTEPS, results_plotter.X_TIMESTEPS, ENV_ID)
plt.legend(legend_names)
plt.show()