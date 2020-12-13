PLOT_NAME = "MODEL NAME"
ENV_ID = 'Block-v2'
bag_dir = 'bag/'
TOTAL_TIMESTEPS = 2e6

legend_names = []
legend_names.append('Default')
legend_names.append('Optuna')

results = []
results.append(bag_dir + "default/")
results.append(bag_dir + "optuna/")
from stable_baselines3.common import results_plotter
import matplotlib.pyplot as plt

results_plotter.plot_results(results, TOTAL_TIMESTEPS, results_plotter.X_TIMESTEPS, PLOT_NAME)
plt.legend(legend_names)
plt.show()