import optuna
import matplotlib.pyplot as plt
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler



ENV_ID = "Block-v1"


N_STARTUP_TRIALS = 5 ### Originally 5
N_EVALUATIONS = 100 ### Originally 2

sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)


study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize", study_name=ENV_ID + '_study',
storage='sqlite:///study.db',
load_if_exists=True)

optuna.visualization.matplotlib.plot_intermediate_values(study)
plt.show()
