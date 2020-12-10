from gym.envs.registration import register, registry

env_dict = registry.env_specs.copy()
if 'CustomAnt-v0' not in env_dict:
    register(id='CustomAnt-v0', # id by which to refer to the new environment; the string is passed as an argument to gym.make() to create a copy of the environment
    	entry_point='custom_ant.envs:CustomAntEnvV0' # points to the class that inherits from gym.Env and defines the four basic functions, i.e. reset, step, render, close
    )

if 'CustomAnt-v2' not in env_dict:
    register(id='CustomAnt-v2', # id by which to refer to the new environment; the string is passed as an argument to gym.make() to create a copy of the environment
    	entry_point='custom_ant.envs:CustomAntEnvV2' # points to the class that inherits from gym.Env and defines the four basic functions, i.e. reset, step, render, close
    )

if 'CustomAnt-v3' not in env_dict:
    register(id='CustomAnt-v3', # id by which to refer to the new environment; the string is passed as an argument to gym.make() to create a copy of the environment
    	entry_point='custom_ant.envs:CustomAntEnvV3' # points to the class that inherits from gym.Env and defines the four basic functions, i.e. reset, step, render, close
    )

if 'CustomAnt-v4' not in env_dict:
    register(id='CustomAnt-v4', # id by which to refer to the new environment; the string is passed as an argument to gym.make() to create a copy of the environment
    	entry_point='custom_ant.envs:CustomAntEnvV4' # points to the class that inherits from gym.Env and defines the four basic functions, i.e. reset, step, render, close
    )

if 'Block-v0' not in env_dict:
    register(id='Block-v0', # id by which to refer to the new environment; the string is passed as an argument to gym.make() to create a copy of the environment
    	entry_point='custom_ant.envs:BlockV0' # points to the class that inherits from gym.Env and defines the four basic functions, i.e. reset, step, render, close
    )

if 'Block-v1' not in env_dict:
    register(id='Block-v1', # id by which to refer to the new environment; the string is passed as an argument to gym.make() to create a copy of the environment
    	entry_point='custom_ant.envs:BlockV1' # points to the class that inherits from gym.Env and defines the four basic functions, i.e. reset, step, render, close
    )

if 'Block-v2' not in env_dict:
    register(id='Block-v2', # id by which to refer to the new environment; the string is passed as an argument to gym.make() to create a copy of the environment
    	entry_point='custom_ant.envs:BlockV2' # points to the class that inherits from gym.Env and defines the four basic functions, i.e. reset, step, render, close
    )

if 'Block-v3' not in env_dict:
    register(id='Block-v3', # id by which to refer to the new environment; the string is passed as an argument to gym.make() to create a copy of the environment
    	entry_point='custom_ant.envs:BlockV3' # points to the class that inherits from gym.Env and defines the four basic functions, i.e. reset, step, render, close
    )
