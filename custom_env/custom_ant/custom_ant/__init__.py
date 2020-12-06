from gym.envs.registration import register

register(id='CustomAnt-v0', # id by which to refer to the new environment; the string is passed as an argument to gym.make() to create a copy of the environment
	entry_point='custom_ant.envs:CustomAntEnvV0' # points to the class that inherits from gym.Env and defines the four basic functions, i.e. reset, step, render, close
)


register(id='CustomAnt-v2', # id by which to refer to the new environment; the string is passed as an argument to gym.make() to create a copy of the environment
	entry_point='custom_ant.envs:CustomAntEnvV2' # points to the class that inherits from gym.Env and defines the four basic functions, i.e. reset, step, render, close
)

register(id='CustomAnt-v3', # id by which to refer to the new environment; the string is passed as an argument to gym.make() to create a copy of the environment
	entry_point='custom_ant.envs:CustomAntEnvV3' # points to the class that inherits from gym.Env and defines the four basic functions, i.e. reset, step, render, close
)

register(id='CustomAnt-v4', # id by which to refer to the new environment; the string is passed as an argument to gym.make() to create a copy of the environment
	entry_point='custom_ant.envs:CustomAntEnvV4' # points to the class that inherits from gym.Env and defines the four basic functions, i.e. reset, step, render, close
)
