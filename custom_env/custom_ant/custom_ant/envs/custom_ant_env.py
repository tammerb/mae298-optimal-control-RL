import os
import mujoco_py as mjp
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class CustomAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        # mujoco_env.MujocoEnv.__init__(self, os.getcwd() + '/custom_ant/models/custom_ant_v2.xml', 5)
        mujoco_env.MujocoEnv.__init__(self, os.getcwd() + '/custom_ant/models/original_ant.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        
        
        #######
        mjp.functions.mj_rnePostConstraint(self.sim.model, self.sim.data) #### Forces mujoco to calculate the contact forces. Does not do so by default
        cf = self.sim.data.cfrc_ext #contact force data 14X6
        
        # located the sections in the matrix cooresponding to the feet
        # the 6 colums are = force x,y,z + torque x,y,z
        
        print(" ")
        print("foot 1  ")
        cf_f1 = cf[4,:]
        print(cf_f1)
        
        print("foot 2  ")
        cf_f2 = cf[7,:]
        print(cf_f2) 
          
        print("foot 3  ")
        cf_f3 = cf[10,:]
        print(cf_f3) 
         
        print("foot 4  ")
        cf_f4 = cf[13,:]
        print(cf_f4)
        ####### 
         
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 1.5
